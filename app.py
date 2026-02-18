# app.py — NEO PI-R Calculatrice Clinique PRO 2026 (Stable)
# Workflow: 1 item -> boutons XXL -> item suivant (200 copies friendly)
# DB SQLite + Backup suppression + Exports CSV/PDF/PNG + Theme + Flash + Sound
# ------------------------------------------------------------

import io
import os
import csv
import sqlite3
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas


# =========================
# CONFIG
# =========================
APP_TITLE = "NEO PI-R — Calculatrice Pro 2026 (Cabinet) — ADAOUN YACINE"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # index 0..4
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}
IDX_TO_OPT = {i: k for k, i in OPT_TO_IDX.items()}


# =========================
# SCORING KEY
# =========================
@st.cache_data
def load_scoring_key(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier '{path}' introuvable. Mets scoring_key.csv à la racine."
        )
    key: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"item", "FD", "D", "N", "A", "FA"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"scoring_key.csv: colonnes requises {sorted(required_cols)} "
                f"mais trouvé {reader.fieldnames}"
            )
        for row in reader:
            item = int(row["item"])
            key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])]

    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplet. Items manquants: {missing[:30]} ...")
    return key


# =========================
# MAPPING ITEMS -> FACETTES
# =========================
facet_bases = {
    "N1": [1],  "N2": [6],  "N3": [11], "N4": [16], "N5": [21], "N6": [26],
    "E1": [2],  "E2": [7],  "E3": [12], "E4": [17], "E5": [22], "E6": [27],
    "O1": [3],  "O2": [8],  "O3": [13], "O4": [18], "O5": [23], "O6": [28],
    "A1": [4],  "A2": [9],  "A3": [14], "A4": [19], "A5": [24], "A6": [29],
    "C1": [5],  "C2": [10], "C3": [15], "C4": [20], "C5": [25], "C6": [30],
}

item_to_facette: Dict[int, str] = {}
for fac, bases in facet_bases.items():
    for b in bases:
        for k in range(0, 240, 30):
            item_to_facette[b + k] = fac

facettes_to_domain = {
    **{f"N{i}": "N" for i in range(1, 7)},
    **{f"E{i}": "E" for i in range(1, 7)},
    **{f"O{i}": "O" for i in range(1, 7)},
    **{f"A{i}": "A" for i in range(1, 7)},
    **{f"C{i}": "C" for i in range(1, 7)},
}

facette_labels = {
    "N1": "N1 — Anxiété", "N2": "N2 — Hostilité", "N3": "N3 — Dépression",
    "N4": "N4 — Timidité", "N5": "N5 — Impulsivité", "N6": "N6 — Vulnérabilité",
    "E1": "E1 — Chaleur", "E2": "E2 — Grégarité", "E3": "E3 — Affirmation de soi",
    "E4": "E4 — Activité", "E5": "E5 — Recherche d'excitation", "E6": "E6 — Émotions positives",
    "O1": "O1 — Imagination", "O2": "O2 — Esthétique", "O3": "O3 — Sentiments",
    "O4": "O4 — Actions", "O5": "O5 — Idées", "O6": "O6 — Valeurs",
    "A1": "A1 — Confiance", "A2": "A2 — Franchise", "A3": "A3 — Altruisme",
    "A4": "A4 — Compliance", "A5": "A5 — Modestie", "A6": "A6 — Tendresse",
    "C1": "C1 — Compétence", "C2": "C2 — Ordre", "C3": "C3 — Sens du devoir",
    "C4": "C4 — Effort", "C5": "C5 — Autodiscipline", "C6": "C6 — Délibération",
}
domain_labels = {"N": "Névrosisme", "E": "Extraversion", "O": "Ouverture", "A": "Agréabilité", "C": "Conscience"}


# =========================
# PROTOCOLE
# =========================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2  # N


# =========================
# DB (SQLite) — robuste + migration
# =========================
@contextmanager
def db_ctx():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def init_db():
    with db_ctx() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                patient_id TEXT,
                item_id INTEGER,
                response_idx INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (patient_id, item_id)
            )
        """)

        # Migration: anciennes versions pouvaient avoir colonne "response"
        cols = table_cols(conn, "responses")
        if "response" in cols and "response_idx" not in cols:
            conn.execute("ALTER TABLE responses ADD COLUMN response_idx INTEGER;")
            conn.execute("UPDATE responses SET response_idx = response WHERE response_idx IS NULL;")


def upsert_patient(patient_id: str, name: str):
    with db_ctx() as conn:
        conn.execute(
            "INSERT INTO patients(patient_id, name) VALUES(?, ?) "
            "ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name",
            (patient_id, name),
        )


def list_patients(search: str = "") -> List[Tuple[str, str]]:
    with db_ctx() as conn:
        if search.strip():
            q = f"%{search.strip()}%"
            rows = conn.execute(
                "SELECT patient_id, COALESCE(name,'') FROM patients "
                "WHERE patient_id LIKE ? OR name LIKE ? ORDER BY created_at DESC",
                (q, q),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT patient_id, COALESCE(name,'') FROM patients ORDER BY created_at DESC"
            ).fetchall()
    return [(r[0], r[1]) for r in rows]


def load_responses(patient_id: str) -> Dict[int, int]:
    with db_ctx() as conn:
        cols = table_cols(conn, "responses")
        col = "response_idx" if "response_idx" in cols else "response"
        rows = conn.execute(
            f"SELECT item_id, {col} FROM responses WHERE patient_id=?",
            (patient_id,),
        ).fetchall()

    resp = {int(item): int(idx) for item, idx in rows}
    for i in range(1, 241):
        resp.setdefault(i, -1)
    return resp


def save_response(patient_id: str, item_id: int, response_idx: int):
    with db_ctx() as conn:
        cols = table_cols(conn, "responses")
        col = "response_idx" if "response_idx" in cols else "response"
        conn.execute(
            f"INSERT INTO responses(patient_id, item_id, {col}) VALUES(?,?,?) "
            f"ON CONFLICT(patient_id, item_id) DO UPDATE SET {col}=excluded.{col}, updated_at=CURRENT_TIMESTAMP",
            (patient_id, item_id, response_idx),
        )


def reset_response(patient_id: str, item_id: int):
    save_response(patient_id, item_id, -1)


def delete_patient(patient_id: str) -> str:
    # backup auto
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backup_{patient_id}_{ts}.db"
    if os.path.exists(DB_PATH):
        shutil.copy2(DB_PATH, backup_path)

    with db_ctx() as conn:
        conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
        conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
    return backup_path


# =========================
# CALCULS
# =========================
def apply_protocol_rules(responses: Dict[int, int], rules: ProtocolRules) -> Tuple[Dict[int, int], dict]:
    blanks = [i for i, v in responses.items() if v == -1]
    n_blank = len(blanks)
    n_count = sum(1 for v in responses.values() if v == 2)

    status = {
        "valid": True,
        "reasons": [],
        "n_blank": n_blank,
        "n_count": n_count,
        "imputed": 0,
        "blank_items": blanks,
    }

    if n_blank >= rules.max_blank_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop d'items vides: {n_blank} (>= {rules.max_blank_invalid})")

    if n_count >= rules.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop de réponses 'N': {n_count} (>= {rules.max_N_invalid})")

    new_resp = dict(responses)
    if status["valid"] and 0 < n_blank <= rules.impute_blank_if_leq:
        for it in blanks:
            new_resp[it] = rules.impute_option_index
            status["imputed"] += 1

    return new_resp, status


def compute_scores(scoring_key: Dict[int, List[int]], responses: Dict[int, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    facette_scores = {fac: 0 for fac in facette_labels.keys()}

    for item_id, idx in responses.items():
        if idx == -1:
            continue
        fac = item_to_facette.get(item_id)
        if fac is None:
            continue
        facette_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc

    return facette_scores, domain_scores


# =========================
# GRAPHIQUES (matplotlib)
# =========================
def plot_domains_radar(domain_scores: Dict[str, int]):
    labels = ["N", "E", "O", "A", "C"]
    values = [domain_scores[k] for k in labels]
    values = values + values[:1]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Domaines (scores bruts)")
    return fig


def plot_facets_line(facette_scores: Dict[str, int]):
    order = [f"{d}{i}" for d in "NEOAC" for i in range(1, 7)]
    y = [facette_scores[k] for k in order]
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.plot(range(len(order)), y, marker="o", linewidth=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=60, ha="right")
    ax.set_title("Facettes (scores bruts)")
    ax.set_ylabel("Score brut")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_to_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# =========================
# PDF REPORT (scores bruts)
# =========================
def build_pdf_report_bytes(
    patient_id: str,
    patient_name: str,
    status: dict,
    facette_scores: Dict[str, int],
    domain_scores: Dict[str, int],
) -> bytes:
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "RAPPORT NEO PI-R — Scores bruts")
    y -= 24

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Patient ID: {patient_id}")
    y -= 14
    c.drawString(40, y, f"Nom: {patient_name}")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"STATUT PROTOCOLE: {'VALIDE' if status['valid'] else 'INVALIDE'}")
    y -= 16

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Items vides: {status['n_blank']} | N observés: {status['n_count']} | Imputations: {status['imputed']}")
    y -= 14

    if status["reasons"]:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Raisons:")
        y -= 12
        c.setFont("Helvetica", 10)
        for r in status["reasons"]:
            c.drawString(50, y, f"- {r}")
            y -= 12
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "DOMAINES")
    y -= 14
    c.setFont("Helvetica", 10)
    for d in ["N", "E", "O", "A", "C"]:
        c.drawString(40, y, f"{domain_labels[d]} ({d}): {domain_scores[d]}")
        y -= 12
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "FACETTES")
    y -= 14
    c.setFont("Helvetica", 9)
    for fac in sorted(facette_labels.keys()):
        c.drawString(40, y, f"{facette_labels[fac]}: {facette_scores[fac]}")
        y -= 11
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)

    c.save()
    buf.seek(0)
    return buf.getvalue()


# =========================
# FEEDBACK — beep discret
# =========================
def play_beep(volume: float = 0.25):
    # très court beep wav (base64 minimal)
    wav_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="
    components.html(
        f"""
        <script>
        try {{
          const audio = new Audio("data:audio/wav;base64,{wav_b64}");
          audio.volume = {volume};
          audio.play();
        }} catch(e) {{}}
        </script>
        """,
        height=0,
    )


# =========================
# CSS (boutons XXL 3+2)
# =========================
def inject_css(theme: str):
    if theme == "dark":
        bg = "#0e1117"
        panel = "#111827"
        text = "#f9fafb"
        subtle = "#9ca3af"
        border = "rgba(255,255,255,0.10)"
        btn_bg = "#0b1220"
    else:
        bg = "#f8fafc"
        panel = "#ffffff"
        text = "#0f172a"
        subtle = "#64748b"
        border = "rgba(0,0,0,0.10)"
        btn_bg = "#f1f5f9"

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {text}; }}
        .neo-panel {{
          background: {panel};
          border: 1px solid {border};
          border-radius: 18px;
          padding: 18px;
        }}
        .neo-subtle {{ color: {subtle}; }}

        /* Bandeau stats */
        .neo-band {{
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 10px;
          margin-top: 10px;
        }}
        .neo-kpi {{
          background: {panel};
          border: 1px solid {border};
          border-radius: 14px;
          padding: 12px;
          text-align: center;
        }}
        .neo-kpi .t {{ font-size: 12px; color: {subtle}; }}
        .neo-kpi .v {{ font-size: 22px; font-weight: 900; letter-spacing: -0.02em; }}

        @media (max-width: 900px) {{
          .neo-band {{ grid-template-columns: repeat(2, 1fr); }}
        }}

        /* Answer buttons XXL */
        .neo-answer div.stButton > button {{
          width: 100%;
          height: 120px;
          font-size: 44px;
          font-weight: 900;
          border-radius: 20px;
          border: 2px solid {border};
          background: {btn_bg};
          color: {text};
        }}
        .neo-answer div.stButton > button:hover {{
          transform: translateY(-2px);
          filter: brightness(1.06);
        }}

        /* Nav buttons smaller */
        .neo-nav div.stButton > button {{
          height: 52px;
          font-size: 18px;
          font-weight: 800;
          border-radius: 14px;
        }}

        /* Flash animation */
        .neo-flash {{
          animation: neoFlash 380ms ease-out;
          border: 2px solid rgba(34,197,94,0.55);
          box-shadow: 0 0 0 10px rgba(34,197,94,0.15);
        }}
        @keyframes neoFlash {{
          0% {{ transform: scale(0.995); opacity: 0.6; }}
          100% {{ transform: scale(1.0); opacity: 1; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# APP
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")
init_db()

# session state defaults
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "flash_ok" not in st.session_state:
    st.session_state.flash_ok = True
if "sound_ok" not in st.session_state:
    st.session_state.sound_ok = True
if "current_item" not in st.session_state:
    st.session_state.current_item = 1
if "just_saved" not in st.session_state:
    st.session_state.just_saved = False

inject_css(st.session_state.theme)

# Load scoring key
scoring_key = load_scoring_key(SCORING_KEY_CSV)

# Sidebar
with st.sidebar:
    st.markdown("## 👤 Patient")
    theme_choice = st.radio("Thème", ["Sombre", "Clair"], index=0 if st.session_state.theme == "dark" else 1)
    st.session_state.theme = "dark" if theme_choice == "Sombre" else "light"

    st.markdown("---")
    st.session_state.flash_ok = st.toggle("Flash vert", value=st.session_state.flash_ok)
    st.session_state.sound_ok = st.toggle("Son discret", value=st.session_state.sound_ok)

    st.markdown("---")
    mode = st.radio("Mode", ["Ouvrir", "Créer"], horizontal=True)

    search = st.text_input("Recherche (ID ou nom)", value="")
    patients = list_patients(search)

    patient_id = ""
    patient_name = ""

    if mode == "Ouvrir":
        if patients:
            labels = [f"{pid} — {name}" if name else pid for pid, name in patients]
            pick = st.selectbox("Sélection", labels, index=0)
            patient_id = pick.split(" — ")[0].strip()
            patient_name = dict(patients).get(patient_id, "")
        else:
            st.info("Aucun patient. Crée un patient.")
    else:
        patient_id = st.text_input("ID patient (unique)", value="")
        patient_name = st.text_input("Nom (optionnel)", value="")
        if st.button("✅ Enregistrer", type="primary", disabled=(not patient_id.strip())):
            upsert_patient(patient_id.strip(), patient_name.strip())
            st.success("Patient enregistré.")
            st.rerun()

    st.markdown("---")
    st.markdown("## ⚙️ Protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses N ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2,
    )

    st.markdown("---")
    st.markdown("## 🗑️ Gestion patient")
    confirm_delete = st.checkbox("Confirmer suppression", value=False)
    if patient_id.strip() and st.button("Supprimer patient (backup auto)", disabled=not confirm_delete):
        backup = delete_patient(patient_id.strip())
        st.success(f"Patient supprimé. Backup: {backup}")
        st.rerun()

# Need patient
if not patient_id.strip():
    st.title(APP_TITLE)
    st.caption("Sélectionne ou crée un patient pour commencer.")
    st.stop()

# Re-inject css after theme update
inject_css(st.session_state.theme)

# Main header
st.title(APP_TITLE)
st.caption("Workflow rapide: 1 item → boutons XXL → item suivant • Calculs instantanés • Exports")

# Load responses & compute
responses = load_responses(patient_id)
answered = sum(1 for i in range(1, 241) if responses[i] != -1)
remaining = 240 - answered

final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(scoring_key, final_resp)

proto_label = "INVALIDE" if (answered == 240 and not status["valid"]) else "EN COURS" if answered < 240 else "VALIDE"

# Live band
st.markdown(
    f"""
    <div class="neo-band">
      <div class="neo-kpi"><div class="t">Patient</div><div class="v">{patient_id[:12]}</div></div>
      <div class="neo-kpi"><div class="t">Saisis</div><div class="v">{answered}</div></div>
      <div class="neo-kpi"><div class="t">Restants</div><div class="v">{remaining}</div></div>
      <div class="neo-kpi"><div class="t">Items vides</div><div class="v">{status["n_blank"]}</div></div>
      <div class="neo-kpi"><div class="t">Protocole</div><div class="v">{proto_label}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.progress(answered / 240.0)

if not status["valid"]:
    st.error("Protocole INVALIDE")
    for r in status["reasons"]:
        st.write("•", r)

tabs = st.tabs(["🧮 Saisie", "📊 Résultats", "📦 Exports"])

# =========================
# TAB 1 — SAISIE
# =========================
with tabs[0]:
    flash_class = "neo-flash" if (st.session_state.just_saved and st.session_state.flash_ok) else ""
    st.markdown(f"<div class='neo-panel {flash_class}'>", unsafe_allow_html=True)

    # Navigation row
    navA, navB, navC = st.columns([1.0, 1.2, 1.0])
    with navA:
        cur_item = st.number_input("Item", 1, 240, int(st.session_state.current_item), step=1)
        st.session_state.current_item = int(cur_item)

    with navB:
        go = st.text_input("Aller à l'item", value="", placeholder="ex: 120")
        if st.button("Aller", key="go_btn"):
            try:
                v = int(go.strip())
                if 1 <= v <= 240:
                    st.session_state.current_item = v
                    st.rerun()
                else:
                    st.warning("Item doit être entre 1 et 240.")
            except Exception:
                st.warning("Entrez un nombre valide (1..240).")

    with navC:
        if st.button("➡️ Prochain VIDE", key="next_blank"):
            found = None
            start = int(st.session_state.current_item)
            for i in list(range(start, 241)) + list(range(1, start)):
                if responses[i] == -1:
                    found = i
                    break
            if found is not None:
                st.session_state.current_item = found
                st.rerun()

    cur_item = int(st.session_state.current_item)
    cur_idx = responses[cur_item]
    cur_label = "VIDE" if cur_idx == -1 else IDX_TO_OPT[cur_idx]
    fac = item_to_facette.get(cur_item, "?")
    dom = facettes_to_domain.get(fac, "?") if fac != "?" else "?"
    st.markdown(f"**Item {cur_item}/240** • **Réponse : {cur_label}** • **Facette : {fac}** • **Domaine : {dom}**")

    # Reset button
    if st.button("🧹 Réinitialiser la réponse (VIDE)", key="reset_item"):
        reset_response(patient_id, cur_item)
        st.session_state.just_saved = True
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Choisir la réponse")

    # Answer buttons (2 lines: 3 + 2)
    st.markdown("<div class='neo-answer'>", unsafe_allow_html=True)
    r1 = st.columns(3)
    clicked: Optional[int] = None
    with r1[0]:
        if st.button("FD", key="ans_fd"):
            clicked = 0
    with r1[1]:
        if st.button("D", key="ans_d"):
            clicked = 1
    with r1[2]:
        if st.button("N", key="ans_n"):
            clicked = 2

    r2 = st.columns(2)
    with r2[0]:
        if st.button("A", key="ans_a"):
            clicked = 3
    with r2[1]:
        if st.button("FA", key="ans_fa"):
            clicked = 4
    st.markdown("</div>", unsafe_allow_html=True)

    # Apply click
    if clicked is not None:
        save_response(patient_id, cur_item, int(clicked))
        st.session_state.just_saved = True
        if st.session_state.sound_ok:
            play_beep(0.22)

        # auto next
        if cur_item < 240:
            st.session_state.current_item = cur_item + 1
        st.toast("Enregistré ✓", icon="✅")
        st.rerun()

    # Nav small
    st.markdown("<div class='neo-nav'>", unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        if st.button("⬅️ -1", key="nav_m1"):
            st.session_state.current_item = max(1, cur_item - 1)
            st.rerun()
    with n2:
        if st.button("➡️ +1", key="nav_p1"):
            st.session_state.current_item = min(240, cur_item + 1)
            st.rerun()
    with n3:
        if st.button("⏭️ +10", key="nav_p10"):
            st.session_state.current_item = min(240, cur_item + 10)
            st.rerun()
    with n4:
        if st.button("⏮️ -10", key="nav_m10"):
            st.session_state.current_item = max(1, cur_item - 10)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # reset flash after render
    if st.session_state.just_saved:
        st.session_state.just_saved = False


# =========================
# TAB 2 — RESULTATS
# =========================
with tabs[1]:
    st.markdown("<div class='neo-panel'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Items vides", status["n_blank"])
    c2.metric("N observés", status["n_count"])
    c3.metric("Imputations", status["imputed"])

    st.markdown("### Domaines")
    dom_table = [{"Code": d, "Domaine": domain_labels[d], "Score brut": domain_scores[d]} for d in ["N", "E", "O", "A", "C"]]
    st.dataframe(dom_table, hide_index=True, use_container_width=True)
    st.pyplot(plot_domains_radar(domain_scores))

    st.markdown("### Facettes")
    fac_rows = [{"Code": fac, "Facette": facette_labels[fac], "Score brut": facette_scores[fac]} for fac in sorted(facette_labels.keys())]
    st.dataframe(fac_rows, hide_index=True, use_container_width=True)
    st.pyplot(plot_facets_line(facette_scores))

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB 3 — EXPORTS
# =========================
with tabs[2]:
    st.markdown("<div class='neo-panel'>", unsafe_allow_html=True)

    # CSV
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["patient_id", patient_id])
    w.writerow(["name", patient_name])
    w.writerow([])
    w.writerow(["PROTOCOLE", "VALIDE" if status["valid"] else "INVALIDE"])
    w.writerow(["items_vides", status["n_blank"]])
    w.writerow(["n_observes", status["n_count"]])
    w.writerow(["imputations", status["imputed"]])
    w.writerow([])
    w.writerow(["DOMAINES"])
    w.writerow(["code", "label", "score_brut"])
    for d in ["N", "E", "O", "A", "C"]:
        w.writerow([d, domain_labels[d], domain_scores[d]])
    w.writerow([])
    w.writerow(["FACETTES"])
    w.writerow(["code", "label", "score_brut"])
    for fac in sorted(facette_labels.keys()):
        w.writerow([fac, facette_labels[fac], facette_scores[fac]])

    st.download_button("📥 Télécharger CSV", out.getvalue(), f"{patient_id}_neo_pir.csv", "text/csv")

    # PDF
    pdf_bytes = build_pdf_report_bytes(patient_id, patient_name, status, facette_scores, domain_scores)
    st.download_button("📥 Télécharger PDF", pdf_bytes, f"{patient_id}_neo_pir_report.pdf", "application/pdf")

    # PNG
    st.download_button("📥 Radar domaines (PNG)", fig_to_bytes(plot_domains_radar(domain_scores), "png"),
                       f"{patient_id}_domains.png", "image/png")
    st.download_button("📥 Profil facettes (PNG)", fig_to_bytes(plot_facets_line(facette_scores), "png"),
                       f"{patient_id}_facettes.png", "image/png")

    st.markdown("</div>", unsafeโ_allow_html=True)
