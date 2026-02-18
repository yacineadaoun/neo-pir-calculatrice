# app.py
# ============================================================
# NEO PI-R — Calculatrice Pro 2026 (Cabinet)
# Saisie manuelle ultra-rapide : 1 item → 5 boutons → item suivant
# Sauvegarde SQLite • Calcul instantané • Exports CSV/PDF • Graphiques
#
# ✅ UI mobile-first (boutons XXL 3+2)
# ✅ Flash vert quand réponse enregistrée (option)
# ✅ Son discret de validation (option, best-effort selon navigateur)
# ✅ Passage auto + transition douce
# ✅ Mode Clair / Sombre (CSS local)
# ✅ Réinitialiser réponse (VIDE) + Supprimer patient
# ✅ DB robuste (supporte anciens schémas response/response_idx)
# ============================================================

import io
import os
import csv
import sqlite3
import base64
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
import streamlit.components.v1 as components

# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "NEO PI-R — Calculatrice Pro 2026:ADAOUN YACINE"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # index 0..4
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}
IDX_TO_OPT = {i: k for k, i in OPT_TO_IDX.items()}

# ============================================================
# SCORING KEY (CSV)
# scoring_key.csv format:
# item,FD,D,N,A,FA
# 1,4,3,2,1,0
# ...
# ============================================================
@st.cache_resource
def load_scoring_key(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' introuvable. Ajoute scoring_key.csv à la racine du repo."
        )
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key: Dict[int, List[int]] = {}
        for row in reader:
            item = int(row["item"])
            key[item] = [
                int(row["FD"]),
                int(row["D"]),
                int(row["N"]),
                int(row["A"]),
                int(row["FA"]),
            ]
    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplet. Items manquants: {missing[:30]}")
    return key


# ============================================================
# 2) ITEM -> FACETTE -> DOMAINE
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

facette_labels = {
    "N1": "N1 - Anxiété", "N2": "N2 - Hostilité colérique", "N3": "N3 - Dépression",
    "N4": "N4 - Timidité", "N5": "N5 - Impulsivité", "N6": "N6 - Vulnérabilité",
    "E1": "E1 - Chaleur", "E2": "E2 - Grégarité", "E3": "E3 - Affirmation de soi",
    "E4": "E4 - Activité", "E5": "E5 - Recherche d'excitation", "E6": "E6 - Émotions positives",
    "O1": "O1 - Imagination", "O2": "O2 - Esthétique", "O3": "O3 - Sentiments",
    "O4": "O4 - Actions", "O5": "O5 - Idées", "O6": "O6 - Valeurs",
    "A1": "A1 - Confiance", "A2": "A2 - Franchise", "A3": "A3 - Altruisme",
    "A4": "A4 - Compliance", "A5": "A5 - Modestie", "A6": "A6 - Tendresse",
    "C1": "C1 - Compétence", "C2": "C2 - Ordre", "C3": "C3 - Sens du devoir",
    "C4": "C4 - Effort pour réussir", "C5": "C5 - Autodiscipline", "C6": "C6 - Délibération",
}

domain_labels = {
    "N": "Névrosisme",
    "E": "Extraversion",
    "O": "Ouverture",
    "A": "Agréabilité",
    "C": "Conscience",
}

# ============================================================
# 3) PROTOCOLE
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2  # N


# ============================================================
# 4) DB (SQLite) — robuste schéma
# ============================================================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _table_info(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]  # column names


def init_db():
    conn = db()
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
            response_idx INTEGER, -- -1 blank, 0..4
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (patient_id, item_id)
        )
    """)
    conn.commit()

    # Migration best-effort: some older versions used column name "response"
    cols = _table_info(conn, "responses")
    if "response" in cols and "response_idx" not in cols:
        conn.execute("ALTER TABLE responses ADD COLUMN response_idx INTEGER;")
        conn.execute("UPDATE responses SET response_idx = response WHERE response_idx IS NULL;")
        conn.commit()

    conn.close()


def upsert_patient(patient_id: str, name: str):
    conn = db()
    conn.execute(
        "INSERT INTO patients(patient_id, name) VALUES(?, ?) "
        "ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name",
        (patient_id, name),
    )
    conn.commit()
    conn.close()


def delete_patient(patient_id: str):
    conn = db()
    conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
    conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
    conn.commit()
    conn.close()


def list_patients(search: str = "") -> List[Tuple[str, str]]:
    conn = db()
    if search.strip():
        q = f"%{search.strip()}%"
        rows = conn.execute(
            "SELECT patient_id, COALESCE(name,'') FROM patients "
            "WHERE patient_id LIKE ? OR name LIKE ? "
            "ORDER BY created_at DESC",
            (q, q),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT patient_id, COALESCE(name,'') FROM patients ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def _responses_select_sql(conn: sqlite3.Connection) -> str:
    cols = _table_info(conn, "responses")
    if "response_idx" in cols:
        return "SELECT item_id, response_idx FROM responses WHERE patient_id=?"
    if "response" in cols:
        return "SELECT item_id, response FROM responses WHERE patient_id=?"
    # fallback
    return "SELECT item_id, response_idx FROM responses WHERE patient_id=?"


def load_responses(patient_id: str) -> Dict[int, int]:
    conn = db()
    sql = _responses_select_sql(conn)
    rows = conn.execute(sql, (patient_id,)).fetchall()
    conn.close()
    resp = {int(item): int(idx) for item, idx in rows}
    for i in range(1, 241):
        resp.setdefault(i, -1)
    return resp


def save_response(patient_id: str, item_id: int, response_idx: int):
    conn = db()
    cols = _table_info(conn, "responses")
    col = "response_idx" if "response_idx" in cols else ("response" if "response" in cols else "response_idx")
    conn.execute(
        f"INSERT INTO responses(patient_id, item_id, {col}) VALUES(?,?,?) "
        f"ON CONFLICT(patient_id, item_id) DO UPDATE SET {col}=excluded.{col}, updated_at=CURRENT_TIMESTAMP",
        (patient_id, item_id, response_idx),
    )
    conn.commit()
    conn.close()


def reset_response(patient_id: str, item_id: int):
    save_response(patient_id, item_id, -1)


# ============================================================
# 5) CALCUL
# ============================================================
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
        status["reasons"].append(f"Trop d'items vides : {n_blank} (>= {rules.max_blank_invalid})")

    if n_count >= rules.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop de réponses 'N' : {n_count} (>= {rules.max_N_invalid})")

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


# ============================================================
# 6) GRAPH
# ============================================================
def plot_domains_radar(domain_scores: Dict[str, int]):
    labels = ["N", "E", "O", "A", "C"]
    values = [domain_scores[k] for k in labels]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Domaines (scores bruts)")
    return fig


def plot_facets_line(facette_scores: Dict[str, int]):
    order = [
        "N1","N2","N3","N4","N5","N6",
        "E1","E2","E3","E4","E5","E6",
        "O1","O2","O3","O4","O5","O6",
        "A1","A2","A3","A4","A5","A6",
        "C1","C2","C3","C4","C5","C6",
    ]
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
    return buf.getvalue()


# ============================================================
# 7) PDF REPORT
# ============================================================
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
    y -= 25

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Patient ID: {patient_id}")
    y -= 16
    c.drawString(40, y, f"Nom: {patient_name}")
    y -= 20

    # Phase-friendly status:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"PROTOCOLE: {'VALIDE' if status['valid'] else 'INVALIDE'}")
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(
        40,
        y,
        f"Items vides: {status['n_blank']} | N observés: {status['n_count']} | Imputations: {status['imputed']}",
    )
    y -= 16

    if status["reasons"]:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Raisons:")
        y -= 14
        c.setFont("Helvetica", 10)
        for r in status["reasons"]:
            c.drawString(50, y, f"- {r}")
            y -= 12
        y -= 8

    # Domaines
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "DOMAINES (scores bruts)")
    y -= 16
    c.setFont("Helvetica", 10)
    for d in ["N", "E", "O", "A", "C"]:
        c.drawString(40, y, f"{domain_labels[d]} ({d}): {domain_scores[d]}")
        y -= 12
    y -= 10

    # Facettes
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "FACETTES (scores bruts)")
    y -= 16
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


# ============================================================
# 8) FEEDBACK (flash + sound)
# ============================================================
def play_beep_once():
    """
    Best-effort: tiny beep (base64 wav) played via JS.
    Note: some mobile browsers may block autoplay even after click; toggle is optional.
    """
    # A very short silent-ish beep WAV (safe placeholder).
    # If blocked, nothing breaks.
    wav_b64 = (
        "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="
    )
    html = f"""
    <script>
    try {{
      const audio = new Audio("data:audio/wav;base64,{wav_b64}");
      audio.volume = 0.25;
      audio.play();
    }} catch(e) {{}}
    </script>
    """
    components.html(html, height=0)


# ============================================================
# 9) UI STYLES (theme + XXL buttons + transitions)
# ============================================================
def inject_css(theme: str, flash: bool):
    # theme: "dark" or "light"
    if theme == "dark":
        bg = "#0b0f17"
        panel = "#0f1623"
        text = "#e9eef7"
        subtle = "#a8b3c7"
        border = "rgba(255,255,255,0.10)"
        btn_bg = "#121c2e"
        btn_bg2 = "#0f1828"
    else:
        bg = "#f6f8fc"
        panel = "#ffffff"
        text = "#0f172a"
        subtle = "#475569"
        border = "rgba(15,23,42,0.12)"
        btn_bg = "#ffffff"
        btn_bg2 = "#ffffff"

    flash_css = ""
    if flash:
        flash_css = """
        .neo-flash-ok {
          animation: neoFlashGreen 420ms ease-out 1;
        }
        @keyframes neoFlashGreen {
          0%   { box-shadow: 0 0 0 rgba(34,197,94,0.0); transform: translateY(0); }
          40%  { box-shadow: 0 0 0 6px rgba(34,197,94,0.18); transform: translateY(-1px); }
          100% { box-shadow: 0 0 0 rgba(34,197,94,0.0); transform: translateY(0); }
        }
        """

    st.markdown(
        f"""
        <style>
        /* App background */
        .stApp {{
          background: {bg};
          color: {text};
        }}
        /* Default text */
        html, body, [class*="css"] {{
          color: {text};
        }}
        /* Main blocks */
        .neo-panel {{
          background: {panel};
          border: 1px solid {border};
          border-radius: 18px;
          padding: 16px 18px;
        }}
        .neo-subtle {{
          color: {subtle};
        }}

        /* Smooth transition on rerun */
        .neo-fadein {{
          animation: neoFadeIn 260ms ease-out 1;
        }}
        @keyframes neoFadeIn {{
          from {{ opacity: 0.55; transform: translateY(4px); }}
          to   {{ opacity: 1; transform: translateY(0px); }}
        }}

        /* KPI band */
        .neo-kpi {{
          display: grid;
          grid-template-columns: repeat(4, minmax(0,1fr));
          gap: 10px;
        }}
        .neo-kpi-card {{
          background: {panel};
          border: 1px solid {border};
          border-radius: 14px;
          padding: 12px 12px;
        }}
        .neo-kpi-title {{
          font-size: 12px;
          color: {subtle};
          margin-bottom: 6px;
        }}
        .neo-kpi-value {{
          font-size: 24px;
          font-weight: 800;
          letter-spacing: -0.5px;
        }}

        /* XXL answer buttons: 3 + 2 layout */
        .neo-btn-grid {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
        }}
        .neo-btn-grid-2 {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
          margin-top: 14px;
        }}

        /* Streamlit buttons */
        div.stButton > button {{
          width: 100%;
          min-height: 130px; /* XXL */
          font-size: 54px;   /* XXL */
          font-weight: 900;
          border-radius: 22px;
          border: 2px solid {border};
          background: {btn_bg};
          color: {text};
          transition: transform 120ms ease, filter 120ms ease;
        }}
        div.stButton > button:hover {{
          transform: translateY(-2px);
          filter: brightness(1.03);
        }}
        div.stButton > button:active {{
          transform: translateY(1px);
          filter: brightness(0.98);
        }}

        /* Smaller navigation buttons */
        .neo-nav div.stButton > button {{
          min-height: 58px;
          font-size: 20px;
          font-weight: 800;
          border-radius: 14px;
          background: {btn_bg2};
        }}

        /* Reset button */
        .neo-reset div.stButton > button {{
          min-height: 56px;
          font-size: 18px;
          font-weight: 800;
          border-radius: 14px;
        }}

        {flash_css}

        /* Mobile tweaks */
        @media (max-width: 768px) {{
          .neo-kpi {{
            grid-template-columns: repeat(2, minmax(0,1fr));
          }}
          div.stButton > button {{
            min-height: 120px;
            font-size: 50px;
          }}
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

# Load scoring key once
scoring_key = load_scoring_key(SCORING_KEY_CSV)

# Session defaults
if "current_item" not in st.session_state:
    st.session_state.current_item = 1
if "last_saved" not in st.session_state:
    st.session_state.last_saved = False
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "flash_ok" not in st.session_state:
    st.session_state.flash_ok = True
if "sound_ok" not in st.session_state:
    st.session_state.sound_ok = False

# Sidebar — Patient & Settings
with st.sidebar:
    st.markdown("## 👤 Patient")
    st.session_state.theme = st.radio("Thème", ["dark", "light"], index=0 if st.session_state.theme == "dark" else 1)
    st.session_state.flash_ok = st.toggle("Flash vert", value=st.session_state.flash_ok)
    st.session_state.sound_ok = st.toggle("Son discret", value=st.session_state.sound_ok)
    st.markdown("---")

    mode = st.radio("Mode", ["Ouvrir", "Créer / Modifier"], index=0)

    search = st.text_input("Recherche (ID ou nom)", value="")
    existing = list_patients(search=search)

    patient_id = ""
    patient_name = ""

    if mode == "Ouvrir":
        if existing:
            labels = [f"{pid} — {name}" if name else pid for pid, name in existing]
            pick = st.selectbox("Sélection", labels, index=0)
            patient_id = pick.split(" — ")[0].strip()
            patient_name = dict(existing).get(patient_id, "")
        else:
            st.warning("Aucun patient trouvé.")
    else:
        patient_id = st.text_input("ID patient (unique)", value="")
        patient_name = st.text_input("Nom (optionnel)", value="")
        cA, cB = st.columns(2)
        with cA:
            if st.button("✅ Enregistrer", type="primary", disabled=(not patient_id.strip())):
                upsert_patient(patient_id.strip(), patient_name.strip())
                st.success("Enregistré.")
        with cB:
            st.write("")

    st.markdown("---")
    st.markdown("## 📋 Protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses 'N' ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2,
    )

    st.markdown("---")
    st.markdown("## 🗑️ Gestion patient")
    confirm_delete = st.checkbox("Confirmer suppression (irréversible)", value=False)
    if st.button("Supprimer patient", disabled=(not confirm_delete or not patient_id.strip())):
        delete_patient(patient_id.strip())
        st.success("Patient supprimé.")
        st.session_state.current_item = 1
        st.rerun()

# Apply CSS
inject_css(st.session_state.theme, st.session_state.flash_ok)

# Main header
st.markdown(f"<h1 style='margin-bottom:0'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<div class='neo-subtle'>Saisie manuelle assistée • Calcul instantané • Sauvegarde • Exports • Graphiques</div>", unsafe_allow_html=True)

if not patient_id.strip():
    st.info("Choisis ou crée un patient pour commencer.")
    st.stop()

# Load responses
responses = load_responses(patient_id)

# Progress
answered = sum(1 for i in range(1, 241) if responses[i] != -1)
remaining = 240 - answered

final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(scoring_key, final_resp)

# KPI band (compact, live)
proto_label = "EN COURS"
proto_badge = "🟦"
if answered == 240:
    proto_label = "VALIDE" if status["valid"] else "INVALIDE"
    proto_badge = "🟩" if status["valid"] else "🟥"
else:
    # do not scream invalid at start
    proto_label = "EN COURS"
    proto_badge = "🟦"

st.markdown(
    f"""
    <div class="neo-kpi neo-fadein">
      <div class="neo-kpi-card">
        <div class="neo-kpi-title">Patient</div>
        <div class="neo-kpi-value">{patient_id[:12]}{'…' if len(patient_id)>12 else ''}</div>
      </div>
      <div class="neo-kpi-card">
        <div class="neo-kpi-title">Saisis</div>
        <div class="neo-kpi-value">{answered}</div>
      </div>
      <div class="neo-kpi-card">
        <div class="neo-kpi-title">Restants</div>
        <div class="neo-kpi-value">{remaining}</div>
      </div>
      <div class="neo-kpi-card">
        <div class="neo-kpi-title">Statut</div>
        <div class="neo-kpi-value">{proto_badge} {proto_label}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.progress(answered / 240.0)

# Tabs
tabs = st.tabs(["🧮 Saisie", "📊 Résultats", "📦 Exports"])

# ============================================================
# TAB 1 — SAISIE (ultra simple)
# ============================================================
with tabs[0]:
    st.markdown("<div class='neo-panel neo-fadein'>", unsafe_allow_html=True)

    # Current item controls
    c1, c2, c3 = st.columns([1.1, 1.2, 1.0])
    with c1:
        item = st.number_input("Item", 1, 240, int(st.session_state.current_item), step=1)
        st.session_state.current_item = int(item)

    with c2:
        jump = st.number_input("Aller à", 1, 240, int(st.session_state.current_item), step=1, label_visibility="collapsed")
        st.caption("Aller à item")
        if int(jump) != int(st.session_state.current_item):
            st.session_state.current_item = int(jump)
            st.rerun()

    with c3:
        # find next blank
        if st.button("➡️ Prochain VIDE"):
            cur = int(st.session_state.current_item)
            found = None
            for i in list(range(cur, 241)) + list(range(1, cur)):
                if responses[i] == -1:
                    found = i
                    break
            if found is not None:
                st.session_state.current_item = found
                st.rerun()

    # Current response info
    cur_item = int(st.session_state.current_item)
    cur_idx = responses[cur_item]
    cur_opt = "VIDE" if cur_idx == -1 else IDX_TO_OPT[cur_idx]
    fac = item_to_facette.get(cur_item, "?")
    dom = facettes_to_domain.get(fac, "?")

    info_cols = st.columns(3)
    info_cols[0].markdown(f"**Item :** {cur_item} / 240")
    info_cols[1].markdown(f"**Réponse :** {cur_opt}")
    info_cols[2].markdown(f"**Facette :** {fac} • **Domaine :** {dom}")

    # Reset button (set blank)
    st.markdown("<div class='neo-reset'>", unsafe_allow_html=True)
    if st.button("🧹 Réinitialiser (VIDE)"):
        reset_response(patient_id, cur_item)
        st.session_state.last_saved = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # FLASH wrapper if last action saved
    flash_class = "neo-flash-ok" if st.session_state.last_saved and st.session_state.flash_ok else ""
    st.markdown(f"<div class='{flash_class}'>", unsafe_allow_html=True)

    # Answer buttons: 3 + 2 + (VIDE in nav row)
    # We keep VIDE as separate small action (already via reset), to reduce mistakes.
    st.markdown("### Choisir la réponse")
    st.markdown("<div class='neo-btn-grid'>", unsafe_allow_html=True)
    bA, bB, bC = st.columns(3)
    clicked = None
    with bA:
        if st.button("FD"):
            clicked = 0
    with bB:
        if st.button("D"):
            clicked = 1
    with bC:
        if st.button("N"):
            clicked = 2
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='neo-btn-grid-2'>", unsafe_allow_html=True)
    bD, bE = st.columns(2)
    with bD:
        if st.button("A"):
            clicked = 3
    with bE:
        if st.button("FA"):
            clicked = 4
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end flash wrapper

    # Save + auto-advance
    if clicked is not None:
        save_response(patient_id, cur_item, int(clicked))
        st.session_state.last_saved = True

        # optional sound
        if st.session_state.sound_ok:
            play_beep_once()

        # auto-advance
        if cur_item < 240:
            st.session_state.current_item = cur_item + 1
        st.rerun()

    # Navigation row (smaller)
    st.markdown("<div class='neo-nav'>", unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        if st.button("⬅️ -1"):
            st.session_state.current_item = max(1, cur_item - 1)
            st.session_state.last_saved = False
            st.rerun()
    with n2:
        if st.button("➡️ +1"):
            st.session_state.current_item = min(240, cur_item + 1)
            st.session_state.last_saved = False
            st.rerun()
    with n3:
        if st.button("⏭️ +10"):
            st.session_state.current_item = min(240, cur_item + 10)
            st.session_state.last_saved = False
            st.rerun()
    with n4:
        if st.button("⏮️ -10"):
            st.session_state.current_item = max(1, cur_item - 10)
            st.session_state.last_saved = False
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # panel

    # Clear last_saved flag after render so flash only once
    if st.session_state.last_saved:
        # Keep it True for this render, then reset next run
        st.session_state.last_saved = False

# ============================================================
# TAB 2 — RESULTATS
# ============================================================
with tabs[1]:
    st.markdown("<div class='neo-panel neo-fadein'>", unsafe_allow_html=True)

    # Phase-friendly status message
    if answered < 240:
        st.info("Saisie en cours — les résultats se mettent à jour en temps réel.")
    else:
        if status["valid"]:
            st.success("Protocole VALIDE.")
        else:
            st.error("Protocole INVALIDE.")
            for r in status["reasons"]:
                st.write("•", r)

    # Show compact protocol counters
    cA, cB, cC = st.columns(3)
    cA.metric("Items vides", status["n_blank"])
    cB.metric("N observés", status["n_count"])
    cC.metric("Imputations", status["imputed"])

    st.markdown("### Domaines")
    dom_table = [{"Code": d, "Domaine": domain_labels[d], "Score brut": domain_scores[d]} for d in ["N", "E", "O", "A", "C"]]
    st.dataframe(dom_table, hide_index=True, use_container_width=True)

    st.pyplot(plot_domains_radar(domain_scores))

    st.markdown("### Facettes")
    fac_rows = [{"Code": fac, "Facette": facette_labels[fac], "Score brut": facette_scores[fac]} for fac in sorted(facette_labels.keys())]
    st.dataframe(fac_rows, hide_index=True, use_container_width=True)

    st.pyplot(plot_facets_line(facette_scores))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3 — EXPORTS
# ============================================================
with tabs[2]:
    st.markdown("<div class='neo-panel neo-fadein'>", unsafe_allow_html=True)

    st.subheader("Exports")

    # CSV export
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["patient_id", patient_id])
    w.writerow(["name", patient_name])
    w.writerow(["answered", answered])
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

    # PDF export
    pdf_bytes = build_pdf_report_bytes(patient_id, patient_name, status, facette_scores, domain_scores)
    st.download_button("📥 Télécharger PDF", pdf_bytes, f"{patient_id}_neo_pir_report.pdf", "application/pdf")

    # PNG exports
    fig_radar = plot_domains_radar(domain_scores)
    st.download_button("📥 Domaines (PNG)", fig_to_bytes(fig_radar, "png"), f"{patient_id}_domaines.png", "image/png")
    fig_fac = plot_facets_line(facette_scores)
    st.download_button("📥 Facettes (PNG)", fig_to_bytes(fig_fac, "png"), f"{patient_id}_facettes.png", "image/png")

    st.markdown("</div>", unsafe_allow_html=True)
