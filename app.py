# app.py — NEO PI-R Calculatrice Pro 2026 (Cabinet)
# Workflow: 1 item -> 5 boutons -> item suivant
# Auteur: ADAOUN YACINE

from __future__ import annotations

import io
import os
import csv
import sqlite3
import shutil
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
import streamlit.components.v1 as components


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "🧮 NEO PI-R — Calculatrice Pro 2026 (Cabinet) | ADAOUN YACINE"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # idx 0..4
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}
IDX_TO_OPT = {i: k for k, i in OPT_TO_IDX.items()}


# ============================================================
# SCORING KEY
# ============================================================
@st.cache_data(show_spinner=False)
def load_scoring_key(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' introuvable. Ajoute scoring_key.csv à la racine du repo."
        )
    key: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"item", "FD", "D", "N", "A", "FA"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Colonnes attendues: {sorted(required_cols)}")

        for row in reader:
            item = int(row["item"])
            key[item] = [
                int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])
            ]

    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplet. Items manquants: {missing[:20]}")
    return key


# ============================================================
# MAPPINGS NEO PI-R (5 domaines + 30 facettes)
# Le manuel présente la structure N E O A C et les facettes N1..C6. :contentReference[oaicite:1]{index=1}
# (tableau visible p.9 du PDF rendu)
# ============================================================
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

domain_labels = {
    "N": "Névrosisme",
    "E": "Extraversion",
    "O": "Ouverture",
    "A": "Agréabilité",
    "C": "Conscience",
}

facette_labels = {
    "N1": "Anxiété",
    "N2": "Hostilité",
    "N3": "Dépression",
    "N4": "Timidité / Gêne sociale",
    "N5": "Impulsivité",
    "N6": "Vulnérabilité",

    "E1": "Chaleur",
    "E2": "Grégarité",
    "E3": "Assertivité",
    "E4": "Activité",
    "E5": "Recherche d'excitation",
    "E6": "Émotions positives",

    "O1": "Imagination",
    "O2": "Esthétique",
    "O3": "Sentiments",
    "O4": "Actions",
    "O5": "Idées",
    "O6": "Valeurs",

    "A1": "Confiance",
    "A2": "Franchise",
    "A3": "Altruisme",
    "A4": "Compliance",
    "A5": "Modestie",
    "A6": "Tendresse",

    "C1": "Compétence",
    "C2": "Ordre",
    "C3": "Sens du devoir",
    "C4": "Effort / Réussite",
    "C5": "Autodiscipline",
    "C6": "Délibération",
}


# ============================================================
# PROTOCOLE (paramétrable)
# ============================================================
@dataclass(frozen=True)
class ProtocolRules:
    max_blank_invalid: int = 15   # >= 15 blancs => invalide
    max_N_invalid: int = 42       # >= 42 réponses "N" => invalide
    impute_blank_if_leq: int = 10 # <= 10 blancs => imputation en N
    impute_option_index: int = 2  # N


# ============================================================
# DB (SQLite) — robuste + migration
# ============================================================
@contextmanager
def db_ctx():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
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
            patient_id TEXT NOT NULL,
            item_id INTEGER NOT NULL,
            response_idx INTEGER NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (patient_id, item_id)
        )
        """)

        # Migration depuis anciennes versions (ex: colonne "response")
        cols = table_columns(conn, "responses")
        if "response" in cols and "response_idx" not in cols:
            conn.execute("ALTER TABLE responses ADD COLUMN response_idx INTEGER;")
            conn.execute("UPDATE responses SET response_idx = response WHERE response_idx IS NULL;")
            conn.execute("UPDATE responses SET response_idx = -1 WHERE response_idx IS NULL;")


def upsert_patient(patient_id: str, name: str):
    with db_ctx() as conn:
        conn.execute(
            "INSERT INTO patients(patient_id, name) VALUES(?, ?) "
            "ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name",
            (patient_id, name),
        )


def delete_patient(patient_id: str) -> str:
    if os.path.exists(DB_PATH):
        backup_name = f"backup_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(DB_PATH, backup_name)
    else:
        backup_name = ""

    with db_ctx() as conn:
        conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
        conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
    return backup_name


def list_patients(search: str = "") -> List[Tuple[str, str]]:
    with db_ctx() as conn:
        if search.strip():
            q = f"%{search.strip()}%"
            rows = conn.execute(
                "SELECT patient_id, COALESCE(name,'') "
                "FROM patients WHERE patient_id LIKE ? OR name LIKE ? "
                "ORDER BY created_at DESC",
                (q, q),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT patient_id, COALESCE(name,'') FROM patients ORDER BY created_at DESC"
            ).fetchall()
    return [(r[0], r[1]) for r in rows]


def load_responses(patient_id: str) -> Dict[int, int]:
    with db_ctx() as conn:
        rows = conn.execute(
            "SELECT item_id, response_idx FROM responses WHERE patient_id=?",
            (patient_id,),
        ).fetchall()

    resp = {int(item): int(idx) for item, idx in rows}
    for i in range(1, 241):
        resp.setdefault(i, -1)
    return resp


def save_response(patient_id: str, item_id: int, response_idx: int):
    with db_ctx() as conn:
        conn.execute(
            "INSERT INTO responses(patient_id, item_id, response_idx) VALUES(?,?,?) "
            "ON CONFLICT(patient_id, item_id) DO UPDATE SET "
            "response_idx=excluded.response_idx, updated_at=CURRENT_TIMESTAMP",
            (patient_id, item_id, response_idx),
        )


def reset_item(patient_id: str, item_id: int):
    save_response(patient_id, item_id, -1)


# ============================================================
# CALCULS
# ============================================================
def apply_protocol_rules(responses: Dict[int, int], rules: ProtocolRules) -> Tuple[Dict[int, int], dict]:
    blanks = [i for i, v in responses.items() if v == -1]
    n_count = sum(1 for v in responses.values() if v == 2)

    status = {
        "valid": True,
        "reasons": [],
        "n_blank": len(blanks),
        "n_count": n_count,
        "imputed": 0,
        "blank_items": blanks,
    }

    if status["n_blank"] >= rules.max_blank_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop d'items vides: {status['n_blank']} (>= {rules.max_blank_invalid})")

    if status["n_count"] >= rules.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop de réponses 'N': {status['n_count']} (>= {rules.max_N_invalid})")

    new_resp = dict(responses)
    if status["valid"] and 0 < status["n_blank"] <= rules.impute_blank_if_leq:
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
        if not fac:
            continue
        facette_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc

    return facette_scores, domain_scores


# ============================================================
# EXPORTS / GRAPHIQUES
# ============================================================
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
    ax.set_thetagrids(np.degrees(angles[:-1]), [domain_labels[l] for l in labels])
    ax.set_title("Domaines (scores bruts)")
    return fig


def plot_facets_line(facette_scores: Dict[str, int]):
    order = [f"{d}{i}" for d in "NEOAC" for i in range(1, 7)]
    y = [facette_scores[k] for k in order]

    fig = plt.figure(figsize=(16, 4.8))
    ax = plt.gca()
    ax.plot(range(len(order)), y, marker="o", linewidth=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_title("Facettes (scores bruts)")
    ax.set_ylabel("Score brut")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    return fig


def fig_to_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=160)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


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

    y = height - 48
    c.setFont("Helvetica-Bold", 15)
    c.drawString(40, y, "RAPPORT NEO PI-R — Scores bruts")
    y -= 22

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
    y -= 16

    if status["reasons"]:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Raisons:")
        y -= 12
        c.setFont("Helvetica", 9)
        for r in status["reasons"]:
            c.drawString(50, y, f"- {r}")
            y -= 11
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "DOMAINES (scores bruts)")
    y -= 14
    c.setFont("Helvetica", 10)
    for d in ["N", "E", "O", "A", "C"]:
        c.drawString(40, y, f"{domain_labels[d]} ({d}): {domain_scores[d]}")
        y -= 12
    y -= 8

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "FACETTES (scores bruts)")
    y -= 14
    c.setFont("Helvetica", 9)
    for fac in sorted(facette_labels.keys()):
        c.drawString(40, y, f"{fac} — {facette_labels[fac]}: {facette_scores[fac]}")
        y -= 11
        if y < 60:
            c.showPage()
            y = height - 48
            c.setFont("Helvetica", 9)

    c.save()
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# FEEDBACK (son discret)
# ============================================================
def play_beep_once(volume: float = 0.25):
    # bip ultra court (silence-compatible)
    wav_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="
    components.html(
        f"""
        <script>
        try {{
          const a = new Audio("data:audio/wav;base64,{wav_b64}");
          a.volume = {volume};
          a.play();
        }} catch(e) {{}}
        </script>
        """,
        height=0,
    )


# ============================================================
# CSS (boutons XXL 3 + 2 + flash)
# ============================================================
def inject_css(theme: str):
    if theme == "dark":
        bg = "#0b1220"
        panel = "#0f172a"
        text = "#f8fafc"
        subtle = "#94a3b8"
        border = "rgba(255,255,255,0.10)"
        btn_bg = "#111c33"
    else:
        bg = "#f6f7fb"
        panel = "#ffffff"
        text = "#0f172a"
        subtle = "#64748b"
        border = "rgba(15,23,42,0.10)"
        btn_bg = "#f1f5f9"

    st.markdown(
        f"""
        <style>
        .stApp {{
          background: {bg};
          color: {text};
        }}

        .neo-wrap {{
          max-width: 1200px;
          margin: 0 auto;
        }}

        .neo-panel {{
          background: {panel};
          border: 1px solid {border};
          border-radius: 22px;
          padding: 18px 18px;
          box-shadow: 0 10px 24px rgba(0,0,0,0.10);
        }}

        .neo-subtle {{ color: {subtle}; }}

        .neo-kpi {{
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 10px;
        }}
        @media (max-width: 900px) {{
          .neo-kpi {{ grid-template-columns: repeat(3, 1fr); }}
        }}
        .neo-kpi-card {{
          background: {panel};
          border: 1px solid {border};
          border-radius: 16px;
          padding: 12px;
          text-align: center;
        }}
        .neo-kpi-title {{
          font-size: 12px;
          color: {subtle};
          margin-bottom: 6px;
        }}
        .neo-kpi-value {{
          font-size: 22px;
          font-weight: 900;
          letter-spacing: -0.02em;
        }}

        /* Boutons réponse XXL */
        .neo-answer-row {{
          display: grid;
          grid-template-columns: 1fr 1fr 1.2fr;
          gap: 12px;
          margin-top: 10px;
        }}
        .neo-answer-row-2 {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-top: 12px;
        }}
        @media (max-width: 900px) {{
          .neo-answer-row {{ grid-template-columns: 1fr 1fr; }}
          .neo-answer-row-2 {{ grid-template-columns: 1fr; }}
        }}

        /* Applique au button Streamlit */
        div.stButton > button {{
          width: 100%;
          height: 140px;
          font-size: 54px;
          font-weight: 900;
          border-radius: 26px;
          border: 2px solid {border};
          background: {btn_bg};
          color: {text};
          transition: transform 120ms ease, box-shadow 120ms ease;
        }}
        div.stButton > button:hover {{
          transform: translateY(-2px);
          box-shadow: 0 10px 20px rgba(0,0,0,0.18);
        }}

        /* Boutons petits (nav/reset) */
        .neo-small div.stButton > button {{
          height: 56px;
          font-size: 18px;
          font-weight: 800;
          border-radius: 14px;
        }}

        /* Flash vert "enregistré" */
        .neo-flash {{
          animation: neoFlash 420ms ease-out;
          border-radius: 14px;
          padding: 10px 12px;
          margin-top: 10px;
          border: 1px solid rgba(34,197,94,0.35);
          background: rgba(34,197,94,0.12);
        }}
        @keyframes neoFlash {{
          0% {{ transform: translateY(6px); opacity: 0.0; }}
          100% {{ transform: translateY(0px); opacity: 1.0; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APP
# ============================================================
init_db()
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")

# Session defaults
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "flash_ok" not in st.session_state:
    st.session_state.flash_ok = True
if "sound_ok" not in st.session_state:
    st.session_state.sound_ok = True
if "current_item" not in st.session_state:
    st.session_state.current_item = 1
if "just_saved" not in st.session_state:
    st.session_state.just_saved = False

inject_css(st.session_state.theme)

scoring_key = load_scoring_key(SCORING_KEY_CSV)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 👤 Patient")
    st.session_state.theme = st.radio("Thème", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1)
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
            labels = [f"{pid} — {name or 'Sans nom'}" for pid, name in patients]
            sel = st.selectbox("Sélection", labels, index=0)
            patient_id = sel.split(" — ")[0].strip()
            patient_name = next((n for p, n in patients if p == patient_id), "")
        else:
            st.info("Aucun patient. Passe en mode 'Créer'.")
    else:
        patient_id = st.text_input("ID patient (unique)", value="")
        patient_name = st.text_input("Nom (optionnel)", value="")
        if st.button("✅ Enregistrer", type="primary", disabled=(not patient_id.strip())):
            upsert_patient(patient_id.strip(), patient_name.strip())
            st.success("Patient enregistré.")
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses 'N' ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2,
    )

    st.markdown("---")
    st.markdown("### 🗑️ Supprimer patient")
    confirm_del = st.checkbox("Je confirme la suppression (backup auto)", value=False)
    if patient_id.strip():
        if st.button("🗑️ Supprimer", disabled=not confirm_del):
            backup_name = delete_patient(patient_id.strip())
            if backup_name:
                st.success(f"Supprimé. Backup créé: {backup_name}")
            else:
                st.success("Supprimé.")
            st.rerun()

# ---------------- Main ----------------
st.markdown('<div class="neo-wrap">', unsafe_allow_html=True)
st.title(APP_TITLE)
st.caption("Workflow clinique: 1 item → 5 boutons → item suivant • Calculs instantanés • Exports")

if not patient_id.strip():
    st.info("Sélectionne ou crée un patient dans la barre latérale.")
    st.stop()

responses = load_responses(patient_id)
answered = sum(1 for v in responses.values() if v != -1)
remaining = 240 - answered

final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(scoring_key, final_resp)

protocol_badge = "✅ VALIDE" if (answered == 240 and status["valid"]) else ("🔄 EN COURS" if answered < 240 else "❌ INVALIDE")

# Bande stats live (style cabinet)
st.markdown(
    f"""
    <div class="neo-kpi">
      <div class="neo-kpi-card"><div class="neo-kpi-title">Patient</div><div class="neo-kpi-value">{patient_id}</div></div>
      <div class="neo-kpi-card"><div class="neo-kpi-title">Saisis</div><div class="neo-kpi-value">{answered}</div></div>
      <div class="neo-kpi-card"><div class="neo-kpi-title">Restants</div><div class="neo-kpi-value">{remaining}</div></div>
      <div class="neo-kpi-card"><div class="neo-kpi-title">Items vides</div><div class="neo-kpi-value">{status["n_blank"]}</div></div>
      <div class="neo-kpi-card"><div class="neo-kpi-title">N observés</div><div class="neo-kpi-value">{status["n_count"]}</div></div>
      <div class="neo-kpi-card"><div class="neo-kpi-title">Statut</div><div class="neo-kpi-value">{protocol_badge}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.progress(answered / 240.0)

if not status["valid"]:
    st.error("Protocole INVALIDE")
    for r in status["reasons"]:
        st.write("•", r)

tabs = st.tabs(["🧮 Saisie", "📊 Scores", "📦 Exports"])

# ============================================================
# TAB 1 — SAISIE (ultra simple)
# ============================================================
with tabs[0]:
    st.markdown('<div class="neo-panel">', unsafe_allow_html=True)

    # Navigation / contexte
    colA, colB, colC, colD = st.columns([1.1, 1.2, 1.1, 1.0])
    with colA:
        cur_item = st.number_input("Item", 1, 240, int(st.session_state.current_item), step=1)
        st.session_state.current_item = int(cur_item)
    with colB:
        jump = st.number_input("Aller à", 1, 240, int(st.session_state.current_item), step=1)
        if jump != st.session_state.current_item:
            st.session_state.current_item = int(jump)
            st.rerun()
    with colC:
        st.markdown('<div class="neo-small">', unsafe_allow_html=True)
        if st.button("➡️ Prochain VIDE"):
            for i in range(st.session_state.current_item, 241):
                if responses[i] == -1:
                    st.session_state.current_item = i
                    st.rerun()
                    break
        st.markdown("</div>", unsafe_allow_html=True)
    with colD:
        st.markdown('<div class="neo-small">', unsafe_allow_html=True)
        if st.button("🧹 Réinitialiser"):
            reset_item(patient_id, st.session_state.current_item)
            st.session_state.just_saved = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    cur_item = int(st.session_state.current_item)
    cur_idx = responses[cur_item]
    cur_label = "VIDE" if cur_idx == -1 else IDX_TO_OPT[cur_idx]
    fac = item_to_facette.get(cur_item, "?")
    dom = facettes_to_domain.get(fac, "?")

    st.markdown(f"**Item {cur_item}/240** • Réponse: **{cur_label}** • Facette: **{fac}** • Domaine: **{dom}**")

    # Feedback visuel après sauvegarde
    if st.session_state.just_saved and st.session_state.flash_ok:
        st.markdown('<div class="neo-flash"><b>✓ Enregistré</b></div>', unsafe_allow_html=True)
        st.session_state.just_saved = False

    # Boutons XXL (3 + 2)
    st.markdown("### Choisir la réponse")
    clicked = None

    st.markdown('<div class="neo-answer-row">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        if st.button("FD"):
            clicked = 0
    with c2:
        if st.button("D"):
            clicked = 1
    with c3:
        if st.button("N"):
            clicked = 2
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="neo-answer-row-2">', unsafe_allow_html=True)
    c4, c5 = st.columns([1, 1])
    with c4:
        if st.button("A"):
            clicked = 3
    with c5:
        if st.button("FA"):
            clicked = 4
    st.markdown("</div>", unsafe_allow_html=True)

    # Bouton "VIDE" séparé (utile si tu veux marquer explicitement vide)
    st.markdown('<div class="neo-small">', unsafe_allow_html=True)
    if st.button("🕳️ Mettre VIDE"):
        clicked = -1
    st.markdown("</div>", unsafe_allow_html=True)

    if clicked is not None:
        save_response(patient_id, cur_item, int(clicked))
        if st.session_state.sound_ok:
            play_beep_once(volume=0.25)

        st.session_state.just_saved = True
        # Auto-avance systématique (sauf si item=240)
        if cur_item < 240:
            st.session_state.current_item = cur_item + 1
        st.rerun()

    # Nav rapide
    st.markdown('<div class="neo-small">', unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        if st.button("⬅️ -1"):
            st.session_state.current_item = max(1, cur_item - 1)
            st.rerun()
    with n2:
        if st.button("➡️ +1"):
            st.session_state.current_item = min(240, cur_item + 1)
            st.rerun()
    with n3:
        if st.button("⏭️ +10"):
            st.session_state.current_item = min(240, cur_item + 10)
            st.rerun()
    with n4:
        if st.button("⏮️ -10"):
            st.session_state.current_item = max(1, cur_item - 10)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — SCORES
# ============================================================
with tabs[1]:
    st.markdown('<div class="neo-panel">', unsafe_allow_html=True)

    st.subheader("Domaines (scores bruts)")
    st.dataframe(
        [{"Code": d, "Domaine": domain_labels[d], "Score brut": domain_scores[d]} for d in ["N", "E", "O", "A", "C"]],
        hide_index=True,
        use_container_width=True,
    )
    st.pyplot(plot_domains_radar(domain_scores))

    st.subheader("Facettes (scores bruts)")
    st.dataframe(
        [{"Code": f, "Facette": facette_labels[f], "Score brut": facette_scores[f]} for f in sorted(facette_labels.keys())],
        hide_index=True,
        use_container_width=True,
    )
    st.pyplot(plot_facets_line(facette_scores))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3 — EXPORTS
# ============================================================
with tabs[2]:
    st.markdown('<div class="neo-panel">', unsafe_allow_html=True)

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

    # PNG plots
    st.download_button("📥 Radar Domaines (PNG)", fig_to_bytes(plot_domains_radar(domain_scores), "png"), f"{patient_id}_domains.png", "image/png")
    st.download_button("📥 Courbe Facettes (PNG)", fig_to_bytes(plot_facets_line(facette_scores), "png"), f"{patient_id}_facettes.png", "image/png")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
