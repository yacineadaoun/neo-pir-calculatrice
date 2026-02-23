# app.py
from __future__ import annotations

import io
import os
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =========================
# Configuration
# =========================
APP_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "neo_pir.db"

ITEMS_CSV = DATA_DIR / "items.csv"          # optionnel: colonnes: item_id, text
SCORING_KEY_CSV = DATA_DIR / "scoring_key.csv"  # optionnel: colonnes mini: item_id, reverse (0/1) ou direction (+1/-1)

TOTAL_ITEMS = 240
RESP_LABELS = ["FD", "D", "N", "A", "FA"]   # indices 0..4
BLANK_IDX = -1                               # pas répondu

# Règles (ajuste si besoin)
MAX_BLANKS_INVALID = 15
MAX_NEUTRAL_INVALID = 42
IMPUTE_IF_BLANKS_LEQ = 10
IMPUTE_TO_IDX = 2  # "N"

RULES_VERSION = "v1.0 (blanc>=15 invalide; N>=42 invalide; si blancs<=10 => impute N)"

# SQLite pragmas
DEFAULT_BUSY_TIMEOUT_MS = 5000

# =========================
# Helpers / Validation
# =========================
def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def facet_index_for_item(item_id: int) -> int:
    # 30 facettes; facette k = {k, k+30, ..., k+210}
    return ((item_id - 1) % 30) + 1  # 1..30

def domain_for_facet(facet_idx: int) -> str:
    # 5 domaines x 6 facettes: 1-6 N, 7-12 E, 13-18 O, 19-24 A, 25-30 C
    if 1 <= facet_idx <= 6:
        return "N"
    if 7 <= facet_idx <= 12:
        return "E"
    if 13 <= facet_idx <= 18:
        return "O"
    if 19 <= facet_idx <= 24:
        return "A"
    return "C"

# =========================
# DB Layer
# =========================
@contextmanager
def db_conn():
    # Connexion courte par transaction = safe en Streamlit multi-sessions
    conn = sqlite3.connect(
        DB_PATH,
        timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000.0,
        isolation_level=None,  # autocommit, on gère BEGIN/COMMIT
    )
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS};")
        conn.execute("PRAGMA journal_mode = WAL;")
        # Choix perf vs durabilité:
        conn.execute("PRAGMA synchronous = NORMAL;")
        yield conn
    finally:
        conn.close()

def db_init():
    with db_conn() as conn:
        conn.execute("BEGIN;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                patient_id TEXT NOT NULL,
                item_id INTEGER NOT NULL CHECK(item_id BETWEEN 1 AND 240),
                response_idx INTEGER NOT NULL CHECK(response_idx BETWEEN -1 AND 4),
                updated_at TEXT NOT NULL,
                PRIMARY KEY (patient_id, item_id),
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            );
        """)
        conn.execute("COMMIT;")

    run_migrations()

def get_user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version;").fetchone()[0])

def set_user_version(conn: sqlite3.Connection, v: int) -> None:
    conn.execute(f"PRAGMA user_version = {int(v)};")

def run_migrations():
    """
    Migrations incrémentales. Exemple: si tu avais une ancienne colonne 'response'
    -> 'response_idx'. Ici on met un squelette prêt à étendre.
    """
    with db_conn() as conn:
        uv = get_user_version(conn)
        if uv >= 1:
            return

        # Migration v0 -> v1 (placeholder)
        conn.execute("BEGIN;")
        # ... ajouter ici des ALTER TABLE / data backfills si nécessaire ...
        set_user_version(conn, 1)
        conn.execute("COMMIT;")

def list_patients() -> List[Tuple[str, str, str]]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT patient_id, full_name, created_at FROM patients ORDER BY created_at DESC;"
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]

def create_patient(patient_id: str, full_name: str) -> None:
    patient_id = patient_id.strip()
    full_name = full_name.strip()
    if not patient_id or not full_name:
        raise ValueError("patient_id et full_name requis.")
    with db_conn() as conn:
        conn.execute("BEGIN;")
        conn.execute(
            "INSERT INTO patients(patient_id, full_name, created_at) VALUES(?,?,?);",
            (patient_id, full_name, now_iso()),
        )
        conn.execute("COMMIT;")

def delete_patient(patient_id: str, make_backup: bool = True) -> None:
    if make_backup and DB_PATH.exists():
        backup_dir = DATA_DIR / "backups"
        backup_dir.mkdir(exist_ok=True)
        safe_pid = "".join(c for c in patient_id if c.isalnum() or c in ("-", "_"))[:40]
        backup_path = backup_dir / f"neo_pir_backup_{safe_pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(DB_PATH, backup_path)

    with db_conn() as conn:
        conn.execute("BEGIN;")
        # ON DELETE CASCADE supprime automatiquement responses
        conn.execute("DELETE FROM patients WHERE patient_id = ?;", (patient_id,))
        conn.execute("COMMIT;")

def upsert_response(patient_id: str, item_id: int, response_idx: int) -> None:
    item_id = clamp_int(int(item_id), 1, TOTAL_ITEMS)
    response_idx = int(response_idx)
    if response_idx not in (-1, 0, 1, 2, 3, 4):
        raise ValueError("response_idx invalide.")
    with db_conn() as conn:
        conn.execute("BEGIN;")
        conn.execute(
            """
            INSERT INTO responses(patient_id, item_id, response_idx, updated_at)
            VALUES(?,?,?,?)
            ON CONFLICT(patient_id, item_id) DO UPDATE SET
                response_idx=excluded.response_idx,
                updated_at=excluded.updated_at;
            """,
            (patient_id, item_id, response_idx, now_iso()),
        )
        conn.execute("COMMIT;")

def load_responses(patient_id: str) -> Dict[int, int]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT item_id, response_idx FROM responses WHERE patient_id = ?;",
            (patient_id,),
        ).fetchall()
    return {int(item_id): int(idx) for item_id, idx in rows}

def next_unanswered_item(resps: Dict[int, int]) -> int:
    for i in range(1, TOTAL_ITEMS + 1):
        if resps.get(i, BLANK_IDX) == BLANK_IDX:
            return i
    return TOTAL_ITEMS + 1  # terminé

# =========================
# Items / Scoring Key loading
# =========================
@st.cache_data(show_spinner=False)
def load_items(items_path: str, mtime: float) -> Dict[int, str]:
    p = Path(items_path)
    if not p.exists():
        return {i: f"Item {i}" for i in range(1, TOTAL_ITEMS + 1)}
    df = pd.read_csv(p)
    if "item_id" not in df.columns:
        raise ValueError("items.csv doit contenir une colonne 'item_id'.")
    text_col = "text" if "text" in df.columns else df.columns[-1]
    mapping = {}
    for _, r in df.iterrows():
        iid = int(r["item_id"])
        if 1 <= iid <= TOTAL_ITEMS:
            mapping[iid] = str(r[text_col])
    # fallback
    for i in range(1, TOTAL_ITEMS + 1):
        mapping.setdefault(i, f"Item {i}")
    return mapping

@st.cache_data(show_spinner=False)
def load_scoring_key(scoring_path: str, mtime: float) -> Dict[int, int]:
    """
    Retourne direction par item: +1 normal, -1 reverse.
    Supporte:
      - colonne reverse (0/1)
      - ou colonne direction (+1/-1)
    """
    p = Path(scoring_path)
    if not p.exists():
        # fallback: aucune inversion
        return {i: +1 for i in range(1, TOTAL_ITEMS + 1)}

    df = pd.read_csv(p)
    if "item_id" not in df.columns:
        raise ValueError("scoring_key.csv doit contenir 'item_id'.")

    direction = {}
    if "direction" in df.columns:
        for _, r in df.iterrows():
            iid = int(r["item_id"])
            val = int(r["direction"])
            direction[iid] = +1 if val >= 0 else -1
    elif "reverse" in df.columns:
        for _, r in df.iterrows():
            iid = int(r["item_id"])
            rev = int(r["reverse"])
            direction[iid] = -1 if rev == 1 else +1
    else:
        # fallback: pas d’inversion
        direction = {i: +1 for i in range(1, TOTAL_ITEMS + 1)}

    for i in range(1, TOTAL_ITEMS + 1):
        direction.setdefault(i, +1)
    return direction

# =========================
# Scoring + Protocol rules
# =========================
@dataclass(frozen=True)
class ProtocolResult:
    status: str  # "valid" | "invalid"
    blanks: int
    neutral: int
    imputed: int
    reason: str

def apply_protocol_rules(resps: Dict[int, int]) -> Tuple[Dict[int, int], ProtocolResult]:
    blanks = sum(1 for i in range(1, TOTAL_ITEMS + 1) if resps.get(i, BLANK_IDX) == BLANK_IDX)
    neutral = sum(1 for i in range(1, TOTAL_ITEMS + 1) if resps.get(i, BLANK_IDX) == IMPUTE_TO_IDX)
    imputed = 0

    if blanks >= MAX_BLANKS_INVALID:
        return resps, ProtocolResult(
            status="invalid",
            blanks=blanks,
            neutral=neutral,
            imputed=0,
            reason=f"Invalide: trop de blancs (>= {MAX_BLANKS_INVALID}).",
        )

    if neutral >= MAX_NEUTRAL_INVALID:
        return resps, ProtocolResult(
            status="invalid",
            blanks=blanks,
            neutral=neutral,
            imputed=0,
            reason=f"Invalide: trop de réponses 'N' (>= {MAX_NEUTRAL_INVALID}).",
        )

    new_resps = dict(resps)
    if blanks <= IMPUTE_IF_BLANKS_LEQ and blanks > 0:
        for i in range(1, TOTAL_ITEMS + 1):
            if new_resps.get(i, BLANK_IDX) == BLANK_IDX:
                new_resps[i] = IMPUTE_TO_IDX
                imputed += 1

    return new_resps, ProtocolResult(
        status="valid",
        blanks=blanks,
        neutral=neutral,
        imputed=imputed,
        reason="OK" if imputed == 0 else f"OK (imputation de {imputed} blancs vers 'N').",
    )

def score_item(response_idx: int, direction: int) -> int:
    """
    Transforme idx 0..4 en score 0..4 puis applique reverse si direction=-1.
    (Ajuste si ton barème réel diffère.)
    """
    if response_idx not in (0, 1, 2, 3, 4):
        return 0
    val = response_idx
    if direction == -1:
        val = 4 - val
    return val

def compute_scores(resps: Dict[int, int], scoring_dir: Dict[int, int]) -> Tuple[Dict[int, int], Dict[str, int]]:
    """
    Returns:
      facet_scores: {facet_idx: raw_sum}
      domain_scores: {"N":sum facets 1..6, ...}
    """
    facet_scores: Dict[int, int] = {f: 0 for f in range(1, 31)}
    for item_id in range(1, TOTAL_ITEMS + 1):
        idx = resps.get(item_id, BLANK_IDX)
        if idx == BLANK_IDX:
            continue
        f = facet_index_for_item(item_id)
        facet_scores[f] += score_item(idx, scoring_dir.get(item_id, +1))

    domain_scores = {"N": 0, "E": 0, "O": 0, "A": 0, "C": 0}
    for f, sc in facet_scores.items():
        domain_scores[domain_for_facet(f)] += sc

    return facet_scores, domain_scores

# =========================
# PDF
# =========================
def build_pdf_bytes(
    patient_id: str,
    patient_name: str,
    protocol: ProtocolResult,
    facet_scores: Dict[int, int],
    domain_scores: Dict[str, int],
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "NEO PI-R — Scores bruts")
    y -= 24

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Patient ID: {patient_id}")
    y -= 14
    c.drawString(50, y, f"Nom: {patient_name}")
    y -= 14
    c.drawString(50, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Protocole")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Statut: {protocol.status.upper()} — {protocol.reason}")
    y -= 14
    c.drawString(50, y, f"Blancs: {protocol.blanks} | N: {protocol.neutral} | Imputés: {protocol.imputed}")
    y -= 14
    c.drawString(50, y, f"Règles: {RULES_VERSION}")
    y -= 22

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Domaines (brut)")
    y -= 14
    c.setFont("Helvetica", 10)
    for k in ["N", "E", "O", "A", "C"]:
        c.drawString(60, y, f"{k}: {domain_scores.get(k, 0)}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Facettes (brut)")
    y -= 14
    c.setFont("Helvetica", 9)

    # 30 facettes: 2 colonnes
    left_x, right_x = 60, 320
    y0 = y
    for f in range(1, 16):
        c.drawString(left_x, y0, f"F{f:02d}: {facet_scores.get(f, 0)}")
        y0 -= 12
    y1 = y
    for f in range(16, 31):
        c.drawString(right_x, y1, f"F{f:02d}: {facet_scores.get(f, 0)}")
        y1 -= 12

    c.showPage()
    c.save()
    return buf.getvalue()

# =========================
# UI State + Callbacks
# =========================
def ensure_state():
    st.session_state.setdefault("patient_id", "")
    st.session_state.setdefault("patient_name", "")
    st.session_state.setdefault("current_item", 1)
    st.session_state.setdefault("sound_ok", False)
    st.session_state.setdefault("pdf_bytes", None)
    st.session_state.setdefault("last_saved_at", None)

def load_patient_into_state(patient_id: str, full_name: str):
    st.session_state.patient_id = patient_id
    st.session_state.patient_name = full_name
    resps = load_responses(patient_id)
    st.session_state.current_item = next_unanswered_item(resps)

def on_answer(selected_idx: int):
    pid = st.session_state.patient_id
    if not pid:
        st.warning("Sélectionne/crée un patient d'abord.")
        return

    item = int(st.session_state.current_item)
    if item > TOTAL_ITEMS:
        st.toast("Questionnaire déjà terminé.")
        return

    upsert_response(pid, item, int(selected_idx))
    st.session_state.last_saved_at = now_iso()
    st.session_state.pdf_bytes = None  # invalider PDF cache session
    st.session_state.current_item = clamp_int(item + 1, 1, TOTAL_ITEMS + 1)

def on_prev():
    st.session_state.current_item = clamp_int(int(st.session_state.current_item) - 1, 1, TOTAL_ITEMS + 1)

def on_jump(item_id: int):
    st.session_state.current_item = clamp_int(int(item_id), 1, TOTAL_ITEMS + 1)

# =========================
# App
# =========================
def main():
    st.set_page_config(page_title="NEO PI-R (Streamlit)", layout="wide")
    ensure_state()
    db_init()

    # Load items + scoring (cache invalidable par mtime)
    items_mtime = ITEMS_CSV.stat().st_mtime if ITEMS_CSV.exists() else 0.0
    scoring_mtime = SCORING_KEY_CSV.stat().st_mtime if SCORING_KEY_CSV.exists() else 0.0
    items = load_items(str(ITEMS_CSV), items_mtime)
    scoring_dir = load_scoring_key(str(SCORING_KEY_CSV), scoring_mtime)

    st.title("NEO PI-R — Passation & Scores bruts")

    # Sidebar patients
    with st.sidebar:
        st.header("Patients")

        patients = list_patients()
        if patients:
            labels = [f"{pid} — {name}" for pid, name, _ in patients]
            default_idx = 0
            current = st.session_state.patient_id
            if current:
                for i, (pid, _, _) in enumerate(patients):
                    if pid == current:
                        default_idx = i
                        break

            choice = st.selectbox("Sélection", labels, index=default_idx)
            selected_pid = patients[labels.index(choice)][0]
            selected_name = patients[labels.index(choice)][1]

            if st.button("Charger patient", use_container_width=True):
                load_patient_into_state(selected_pid, selected_name)
                st.rerun()

        st.divider()
        st.subheader("Créer")
        new_pid = st.text_input("Patient ID", placeholder="ex: P001")
        new_name = st.text_input("Nom complet", placeholder="ex: Nom Prénom")

        if st.button("Créer", type="primary", use_container_width=True):
            try:
                create_patient(new_pid, new_name)
                st.success("Patient créé.")
                load_patient_into_state(new_pid.strip(), new_name.strip())
                st.rerun()
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.subheader("Suppression")
        del_backup = st.checkbox("Faire un backup .db avant suppression", value=True)
        if st.button("Supprimer patient chargé", use_container_width=True):
            pid = st.session_state.patient_id
            if not pid:
                st.warning("Aucun patient chargé.")
            else:
                delete_patient(pid, make_backup=del_backup)
                st.session_state.patient_id = ""
                st.session_state.patient_name = ""
                st.session_state.current_item = 1
                st.session_state.pdf_bytes = None
                st.success("Supprimé.")
                st.rerun()

        st.divider()
        st.caption("Astuce: place `items.csv` et `scoring_key.csv` dans le dossier `data/`.")

    # Main area
    pid = st.session_state.patient_id
    pname = st.session_state.patient_name

    if not pid:
        st.info("Crée ou charge un patient depuis la barre latérale.")
        return

    tabs = st.tabs(["Passation", "Résultats & PDF"])

    # ===== Passation =====
    with tabs[0]:
        resps = load_responses(pid)
        current_item = int(st.session_state.current_item)

        colA, colB, colC = st.columns([2, 1, 1])
        with colA:
            st.subheader(f"Patient: {pname} ({pid})")
            st.caption(f"Dernière sauvegarde: {st.session_state.last_saved_at or '—'}")
        with colB:
            jump_to = st.number_input("Aller à l’item", min_value=1, max_value=TOTAL_ITEMS + 1, value=current_item)
            if st.button("Go", use_container_width=True):
                on_jump(int(jump_to))
                st.rerun()
        with colC:
            st.metric("Progression", f"{min(current_item, TOTAL_ITEMS)}/{TOTAL_ITEMS}")

        if current_item > TOTAL_ITEMS:
            st.success("✅ Passation terminée.")
        else:
            st.markdown(f"### Item {current_item}")
            st.write(items.get(current_item, f"Item {current_item}"))

            answered = resps.get(current_item, BLANK_IDX)
            if answered != BLANK_IDX:
                st.info(f"Réponse existante: **{RESP_LABELS[answered]}** (tu peux écraser en cliquant un bouton)")

            bcols = st.columns(5)
            for i, lab in enumerate(RESP_LABELS):
                with bcols[i]:
                    st.button(
                        lab,
                        use_container_width=True,
                        on_click=on_answer,
                        args=(i,),
                        # shortcuts utiles en passation:
                        shortcut=f"{i+1}",
                    )

            nav1, nav2, nav3 = st.columns([1, 2, 1])
            with nav1:
                st.button("⬅️ Précédent", use_container_width=True, on_click=on_prev)
            with nav2:
                st.caption("Raccourcis: 1..5 pour FD..FA")
            with nav3:
                st.button("Recharger", use_container_width=True, on_click=lambda: st.rerun())

    # ===== Résultats & PDF =====
    with tabs[1]:
        # fragment pour éviter de recalculer inutilement en passation
        @st.fragment
        def results_fragment():
            resps = load_responses(pid)
            resps2, protocol = apply_protocol_rules(resps)
            facet_scores, domain_scores = compute_scores(resps2, scoring_dir)

            st.subheader("Statut protocole")
            if protocol.status == "valid":
                st.success(protocol.reason)
            else:
                st.error(protocol.reason)

            c1, c2, c3 = st.columns(3)
            c1.metric("Blancs", protocol.blanks)
            c2.metric("N", protocol.neutral)
            c3.metric("Imputés", protocol.imputed)

            st.subheader("Domaines (brut)")
            st.write(domain_scores)

            st.subheader("Facettes (brut)")
            df_facets = pd.DataFrame(
                [{"facet": f"F{f:02d}", "domain": domain_for_facet(f), "score": facet_scores[f]} for f in range(1, 31)]
            )
            st.dataframe(df_facets, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("PDF")
            if st.button("Générer PDF", type="primary"):
                st.session_state.pdf_bytes = build_pdf_bytes(
                    patient_id=pid,
                    patient_name=pname,
                    protocol=protocol,
                    facet_scores=facet_scores,
                    domain_scores=domain_scores,
                )
                st.toast("PDF généré.")

            if st.session_state.pdf_bytes:
                st.download_button(
                    "Télécharger PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"NEO_PIR_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.caption("Clique sur “Générer PDF” puis télécharge.")

        results_fragment()

if __name__ == "__main__":
    main()
