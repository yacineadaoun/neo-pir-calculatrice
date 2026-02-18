import io
import os
import csv
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ============================================================
# 0) CONFIG
# ============================================================
APP_TITLE = "NEO PI-R — Calculatrice Pro 2026 (Cabinet)"
DB_PATH = "neo_pir.db"
SCORING_KEY_FILE = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # idx 0..4


# ============================================================
# 1) MAPPING PSYCHOMÉTRIQUE (Items -> Facettes -> Domaines)
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

facettes_to_domain = {**{f"N{i}": "N" for i in range(1, 7)},
                      **{f"E{i}": "E" for i in range(1, 7)},
                      **{f"O{i}": "O" for i in range(1, 7)},
                      **{f"A{i}": "A" for i in range(1, 7)},
                      **{f"C{i}": "C" for i in range(1, 7)}}

facette_labels = {
    'N1': 'N1 - Anxiété', 'N2': 'N2 - Hostilité colérique', 'N3': 'N3 - Dépression',
    'N4': 'N4 - Timidité', 'N5': 'N5 - Impulsivité', 'N6': 'N6 - Vulnérabilité',
    'E1': 'E1 - Chaleur', 'E2': 'E2 - Grégarité', 'E3': 'E3 - Affirmation de soi',
    'E4': 'E4 - Activité', 'E5': "E5 - Recherche d'excitation", 'E6': 'E6 - Émotions positives',
    'O1': 'O1 - Imagination', 'O2': 'O2 - Esthétique', 'O3': 'O3 - Sentiments',
    'O4': 'O4 - Actions', 'O5': 'O5 - Idées', 'O6': 'O6 - Valeurs',
    'A1': 'A1 - Confiance', 'A2': 'A2 - Franchise', 'A3': 'A3 - Altruisme',
    'A4': 'A4 - Conformité (Compliance)', 'A5': 'A5 - Modestie', 'A6': 'A6 - Tendresse',
    'C1': 'C1 - Compétence', 'C2': 'C2 - Ordre', 'C3': 'C3 - Sens du devoir',
    'C4': 'C4 - Effort pour réussir', 'C5': 'C5 - Autodiscipline', 'C6': 'C6 - Délibération'
}

domain_labels = {
    'N': 'Névrosisme',
    'E': 'Extraversion',
    'O': 'Ouverture',
    'A': 'Agréabilité',
    'C': 'Conscience'
}

FACET_ORDER = [
    "N1","N2","N3","N4","N5","N6",
    "E1","E2","E3","E4","E5","E6",
    "O1","O2","O3","O4","O5","O6",
    "A1","A2","A3","A4","A5","A6",
    "C1","C2","C3","C4","C5","C6",
]


# ============================================================
# 2) PROTOCOLE
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2  # "N"


# ============================================================
# 3) DB (robuste + migration)
# ============================================================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def ensure_schema():
    conn = db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients(
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses(
            patient_id TEXT,
            item_id INTEGER,
            response_idx INTEGER,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(patient_id, item_id)
        )
    """)
    conn.commit()

    # migration si colonne "response" existe
    cols = [r[1] for r in conn.execute("PRAGMA table_info(responses)").fetchall()]
    if "response_idx" not in cols and "response" in cols:
        conn.execute("ALTER TABLE responses ADD COLUMN response_idx INTEGER;")
        conn.execute("UPDATE responses SET response_idx = response WHERE response_idx IS NULL;")
        conn.commit()

    conn.close()


def get_resp_col(conn: sqlite3.Connection) -> str:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(responses)").fetchall()]
    if "response_idx" in cols:
        return "response_idx"
    if "response" in cols:
        return "response"
    return "response_idx"


def upsert_patient(patient_id: str, name: str):
    conn = db()
    conn.execute("""
        INSERT INTO patients(patient_id, name)
        VALUES(?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name
    """, (patient_id, name))
    conn.commit()
    conn.close()


def delete_patient(patient_id: str):
    conn = db()
    conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
    conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
    conn.commit()
    conn.close()


def list_patients(query: str = "") -> List[Tuple[str, str]]:
    conn = db()
    if query.strip():
        q = f"%{query.strip()}%"
        rows = conn.execute("""
            SELECT patient_id, COALESCE(name,'')
            FROM patients
            WHERE patient_id LIKE ? OR name LIKE ?
            ORDER BY created_at DESC
            LIMIT 200
        """, (q, q)).fetchall()
    else:
        rows = conn.execute("""
            SELECT patient_id, COALESCE(name,'')
            FROM patients
            ORDER BY created_at DESC
            LIMIT 200
        """).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def load_responses(patient_id: str) -> Dict[int, int]:
    conn = db()
    col = get_resp_col(conn)
    rows = conn.execute(
        f"SELECT item_id, {col} FROM responses WHERE patient_id=?",
        (patient_id,)
    ).fetchall()
    conn.close()

    resp = {i: -1 for i in range(1, 241)}
    for item_id, v in rows:
        resp[int(item_id)] = int(v) if v is not None else -1
    return resp


def save_response(patient_id: str, item_id: int, value: int):
    conn = db()
    col = get_resp_col(conn)
    conn.execute(
        f"""
        INSERT INTO responses(patient_id, item_id, {col})
        VALUES(?, ?, ?)
        ON CONFLICT(patient_id, item_id)
        DO UPDATE SET {col}=excluded.{col}, updated_at=CURRENT_TIMESTAMP
        """,
        (patient_id, item_id, value)
    )
    conn.commit()
    conn.close()


# ============================================================
# 4) SCORING KEY
# ============================================================
@st.cache_data
def load_scoring_key(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError("scoring_key.csv introuvable à la racine du repo.")
    key: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = int(row["item"])
            key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])]

    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplet (ex items manquants: {missing[:10]}).")
    return key


# ============================================================
# 5) CALCULS (protocole + scores)
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
        "blank_items": blanks
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
    facet_scores = {fac: 0 for fac in facette_labels.keys()}

    for item_id, idx in responses.items():
        if idx == -1:
            continue
        fac = item_to_facette.get(item_id)
        if fac is None:
            continue
        facet_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facet_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc

    return facet_scores, domain_scores


# ============================================================
# 6) GRAPHIQUES
# ============================================================
def plot_domains_radar(domain_scores: Dict[str, int]):
    labels = ["N", "E", "O", "A", "C"]
    values = [domain_scores[k] for k in labels]
    values += values[:1]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.12)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Profil Domaines — Scores bruts")
    return fig


def plot_facets_line(facet_scores: Dict[str, int]):
    y = [facet_scores[k] for k in FACET_ORDER]
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.plot(range(len(FACET_ORDER)), y, marker="o", linewidth=2)
    ax.set_xticks(range(len(FACET_ORDER)))
    ax.set_xticklabels(FACET_ORDER, rotation=60, ha="right")
    ax.set_title("Profil Facettes — Scores bruts")
    ax.set_ylabel("Score brut")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# 7) PDF (professionnel + graphiques)
# ============================================================
def build_pdf_report_bytes(
    patient_id: str,
    patient_name: str,
    status: dict,
    facet_scores: Dict[str, int],
    domain_scores: Dict[str, int],
    radar_png: bytes,
    facets_png: bytes,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 44
    c.setFont("Helvetica-Bold", 15)
    c.drawString(40, y, "RAPPORT NEO PI-R — Scores bruts (Cabinet Pro)")
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Patient ID : {patient_id}")
    y -= 14
    c.drawString(40, y, f"Nom : {patient_name if patient_name else '-'}")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"Statut protocole : {'VALIDE' if status['valid'] else 'INVALIDE'}")
    y -= 14

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Items vides : {status['n_blank']} | N observés : {status['n_count']} | Imputations : {status['imputed']}")
    y -= 16

    if status["reasons"]:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Raisons :")
        y -= 12
        c.setFont("Helvetica", 10)
        for r in status["reasons"]:
            c.drawString(52, y, f"- {r}")
            y -= 12
        y -= 6

    # Domaines + radar
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Domaines (scores bruts)")
    y -= 14

    c.setFont("Helvetica", 10)
    for d in ["N", "E", "O", "A", "C"]:
        c.drawString(40, y, f"{domain_labels[d]} ({d}) : {domain_scores[d]}")
        y -= 12

    y -= 6
    radar = ImageReader(io.BytesIO(radar_png))
    c.drawImage(radar, 320, y - 140, width=240, height=240, mask='auto')

    # Facettes + graph
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Facettes (scores bruts)")
    y -= 8

    facets_img = ImageReader(io.BytesIO(facets_png))
    c.drawImage(facets_img, 40, y - 190, width=520, height=180, mask='auto')

    c.showPage()

    # Page 2: liste facettes
    y = height - 44
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Détail des 30 facettes — Scores bruts")
    y -= 22

    c.setFont("Helvetica", 10)
    for fac in FACET_ORDER:
        c.drawString(40, y, f"{facette_labels[fac]} : {facet_scores[fac]}")
        y -= 12
        if y < 60:
            c.showPage()
            y = height - 44
            c.setFont("Helvetica", 10)

    c.save()
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# 8) FX UX (flash + beep) + Theme switch
# ============================================================
def play_beep_once(beep_id: int, enabled: bool):
    if not enabled:
        return
    # beep_id change -> re-render script -> play
    components.html(f"""
    <script>
    (function() {{
      const id = {beep_id};
      // simple beep
      try {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = 880;
        g.gain.value = 0.02;
        o.connect(g); g.connect(ctx.destination);
        o.start();
        setTimeout(() => {{ o.stop(); ctx.close(); }}, 90);
      }} catch(e) {{}}
    }})();
    </script>
    """, height=0)


def flash_success_once(enabled: bool):
    if not enabled:
        return
    components.html("""
    <style>
    .flash-ok {
      position: fixed; inset: 0; z-index: 9999;
      background: rgba(34,197,94,0.16);
      animation: flashFade 260ms ease-out forwards;
      pointer-events:none;
    }
    @keyframes flashFade { from {opacity:1;} to {opacity:0;} }
    </style>
    <div class="flash-ok"></div>
    """, height=0)


def inject_theme_css(theme: str):
    # theme in {"dark","light"}
    if theme == "dark":
        bg = "#0B1220"
        card = "rgba(255,255,255,0.04)"
        border = "rgba(255,255,255,0.10)"
        text = "#E5E7EB"
    else:
        bg = "#F6F7FB"
        card = "rgba(0,0,0,0.02)"
        border = "rgba(0,0,0,0.10)"
        text = "#111827"

    st.markdown(f"""
    <style>
    .stApp {{
      background: {bg};
      color: {text};
    }}
    .pro-card {{
      border-radius: 20px;
      padding: 18px;
      border: 1px solid {border};
      background: {card};
    }}
    .badge {{
      display:inline-flex; align-items:center; gap:8px;
      padding: 8px 12px; border-radius: 999px;
      font-weight: 800; font-size: 14px;
      border: 1px solid {border};
      background: {card};
    }}
    .answerpad div.stButton>button {{
      height: 150px !important;
      font-size: 52px !important;
      font-weight: 900 !important;
      border-radius: 26px !important;
      width: 100% !important;
      border: 2px solid {border} !important;
      box-shadow: 0 10px 24px rgba(0,0,0,0.10) !important;
      transition: transform 0.06s ease-in-out;
    }}
    .answerpad div.stButton>button:active {{ transform: scale(0.97); }}

    .btn-fd div.stButton>button {{ background:#B91C1C !important; color:white !important; }}
    .btn-d  div.stButton>button {{ background:#374151 !important; color:white !important; }}
    .btn-n  div.stButton>button {{ background:#E5E7EB !important; color:#111827 !important; }}
    .btn-a  div.stButton>button {{ background:#2563EB !important; color:white !important; }}
    .btn-fa div.stButton>button {{ background:#16A34A !important; color:white !important; }}

    .navrow div.stButton>button {{
      height: 70px !important;
      font-size: 22px !important;
      font-weight: 850 !important;
      border-radius: 18px !important;
      width: 100% !important;
    }}

    .btn-reset div.stButton>button {{
      height: 78px !important;
      font-size: 20px !important;
      font-weight: 900 !important;
      border-radius: 18px !important;
      width: 100% !important;
      background: transparent !important;
    }}

    @media (max-width: 768px) {{
      .answerpad div.stButton>button {{
        height: 170px !important;
        font-size: 56px !important;
      }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 9) APP
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")
ensure_schema()
scoring_key = load_scoring_key(SCORING_KEY_FILE)

# Session state init
if "current_item" not in st.session_state:
    st.session_state.current_item = 1
if "beep_counter" not in st.session_state:
    st.session_state.beep_counter = 0
if "flash" not in st.session_state:
    st.session_state.flash = False
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

st.title(APP_TITLE)
st.caption("Workflow clinique rapide : 1 item → 5 boutons → item suivant • Calculs instantanés • Exports")

# Sidebar
with st.sidebar:
    st.subheader("🧑‍⚕️ Patient")

    theme_choice = st.radio("Thème", ["Sombre", "Clair"], index=0 if st.session_state.theme == "dark" else 1)
    st.session_state.theme = "dark" if theme_choice == "Sombre" else "light"

    st.markdown("---")
    search = st.text_input("Recherche (ID ou nom)", value="").strip()
    patients = list_patients(search)

    if patients:
        labels = [f"{pid} — {name}" if name else pid for pid, name in patients]
        pick = st.selectbox("Ouvrir", labels, index=0)
        patient_id = pick.split(" — ")[0].strip()
        patient_name = dict(patients).get(patient_id, "")
    else:
        patient_id = ""
        patient_name = ""

    st.markdown("---")
    st.subheader("Créer / Modifier")
    new_id = st.text_input("ID patient", value="")
    new_name = st.text_input("Nom (optionnel)", value="")

    cA, cB = st.columns(2)
    with cA:
        if st.button("✅ Enregistrer", use_container_width=True, disabled=(not new_id.strip())):
            upsert_patient(new_id.strip(), new_name.strip())
            st.success("Patient enregistré.")
            st.rerun()

    with cB:
        st.caption("Suppression irréversible")
        confirm_del = st.checkbox("Confirmer", value=False)
        if st.button("🗑️ Supprimer", use_container_width=True, disabled=(not patient_id or not confirm_del)):
            delete_patient(patient_id)
            st.session_state.current_item = 1
            st.warning("Patient supprimé.")
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses 'N' ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2
    )

    st.markdown("---")
    st.subheader("🔔 Feedback")
    enable_flash = st.toggle("Flash vert", value=True)
    enable_beep = st.toggle("Son discret", value=False)

    st.markdown("---")
    debug = st.toggle("Debug", value=False)

# Apply theme CSS
inject_theme_css(st.session_state.theme)

# Need patient
if not patient_id:
    st.info("Crée ou sélectionne un patient dans la barre latérale.")
    st.stop()

# Load responses
responses = load_responses(patient_id)

# Progress & live stats
answered = sum(1 for v in responses.values() if v != -1)
blank_now = 240 - answered
st.progress(answered / 240.0)

band1, band2, band3, band4, band5 = st.columns(5)
band1.metric("Patient", patient_id)
band2.metric("Saisis", answered)
band3.metric("Restants", blank_now)
band4.metric("Item courant", int(st.session_state.current_item))
band5.metric("Nom", patient_name if patient_name else "-")

# Protocol + scores (live)
final_resp, status = apply_protocol_rules(responses, rules)
facet_scores, domain_scores = compute_scores(scoring_key, final_resp)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Items vides", status["n_blank"])
k2.metric("N observés", status["n_count"])
k3.metric("Imputations", status["imputed"])
k4.metric("Statut", "VALIDE" if status["valid"] else "INVALIDE")

if not status["valid"]:
    st.error("Protocole INVALIDE")
    for r in status["reasons"]:
        st.write("•", r)

# Tabs
tabs = st.tabs(["🧮 Saisie", "📊 Résultats", "📦 Exports", "🗂️ Global"])


# ============================================================
# TAB 1 — SAISIE ultra-rapide
# ============================================================
with tabs[0]:
    # Flash/beep
    if st.session_state.flash:
        flash_success_once(enable_flash)
        st.session_state.flash = False
    play_beep_once(st.session_state.beep_counter, enable_beep)

    st.markdown('<div class="pro-card">', unsafe_allow_html=True)

    item = int(max(1, min(240, st.session_state.current_item)))
    current_idx = responses[item]
    current_label = "VIDE" if current_idx == -1 else OPTIONS[current_idx]
    fac = item_to_facette.get(item, "—")
    dom = facettes_to_domain.get(fac, "—")

    st.markdown(f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; justify-content:space-between; margin-bottom:12px;">
      <div class="badge">Item <strong>{item}</strong> / 240</div>
      <div class="badge">Réponse : <strong>{current_label}</strong></div>
      <div class="badge">Facette : <strong>{fac}</strong> • Domaine : <strong>{dom}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    def commit(idx: int):
        save_response(patient_id, item, idx)
        st.session_state.flash = True
        st.session_state.beep_counter += 1
        if item < 240:
            st.session_state.current_item = item + 1
        st.rerun()

    # Reset
    st.markdown('<div class="btn-reset">', unsafe_allow_html=True)
    if st.button("🧹 Réinitialiser (VIDE)", use_container_width=True):
        save_response(patient_id, item, -1)
        st.session_state.flash = True
        st.session_state.beep_counter += 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Choisir la réponse")

    st.markdown('<div class="answerpad">', unsafe_allow_html=True)

    # Ligne 1 (3)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="btn-fd">', unsafe_allow_html=True)
        if st.button("FD", use_container_width=True):
            commit(0)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="btn-d">', unsafe_allow_html=True)
        if st.button("D", use_container_width=True):
            commit(1)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="btn-n">', unsafe_allow_html=True)
        if st.button("N", use_container_width=True):
            commit(2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ligne 2 (2)
    c4, c5 = st.columns(2)
    with c4:
        st.markdown('<div class="btn-a">', unsafe_allow_html=True)
        if st.button("A", use_container_width=True):
            commit(3)
        st.markdown('</div>', unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="btn-fa">', unsafe_allow_html=True)
        if st.button("FA", use_container_width=True):
            commit(4)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # answerpad

    st.markdown("---")
    st.markdown('<div class="navrow">', unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    if n1.button("⬅️ -1", use_container_width=True):
        st.session_state.current_item = max(1, item - 1); st.rerun()
    if n2.button("➡️ +1", use_container_width=True):
        st.session_state.current_item = min(240, item + 1); st.rerun()
    if n3.button("⏭️ +10", use_container_width=True):
        st.session_state.current_item = min(240, item + 10); st.rerun()
    if n4.button("⏮️ -10", use_container_width=True):
        st.session_state.current_item = max(1, item - 10); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if debug:
        st.write({"item": item, "current_idx": current_idx, "facette": fac, "domaine": dom})
        st.write("Score key item:", scoring_key.get(item))

    st.markdown('</div>', unsafe_allow_html=True)  # pro-card


# ============================================================
# TAB 2 — RESULTATS (scientifique + lisible)
# ============================================================
with tabs[1]:
    st.markdown("## Résultats — Scores bruts")
    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("### Domaines (Big Five)")
        dom_rows = [{"Code": d, "Domaine": domain_labels[d], "Score brut": domain_scores[d]} for d in ["N","E","O","A","C"]]
        st.dataframe(dom_rows, hide_index=True, use_container_width=True)

        fig_radar = plot_domains_radar(domain_scores)
        st.pyplot(fig_radar)

    with colR:
        st.markdown("### Facettes (30)")
        fac_rows = [{"Code": fac, "Facette": facette_labels[fac], "Score brut": facet_scores[fac]} for fac in FACET_ORDER]
        st.dataframe(fac_rows, hide_index=True, use_container_width=True)

        fig_facets = plot_facets_line(facet_scores)
        st.pyplot(fig_facets)


# ============================================================
# TAB 3 — EXPORTS (CSV + PDF + PNG)
# ============================================================
with tabs[2]:
    st.markdown("## Exports patient (Cabinet)")

    # CSV patient (structuré)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["patient_id", patient_id])
    w.writerow(["name", patient_name])
    w.writerow([])
    w.writerow(["STATUT_PROTOCOLE", "VALIDE" if status["valid"] else "INVALIDE"])
    w.writerow(["items_vides", status["n_blank"]])
    w.writerow(["n_observes", status["n_count"]])
    w.writerow(["imputations", status["imputed"]])
    w.writerow([])
    w.writerow(["DOMAINES"])
    w.writerow(["code", "label", "score_brut"])
    for d in ["N","E","O","A","C"]:
        w.writerow([d, domain_labels[d], domain_scores[d]])
    w.writerow([])
    w.writerow(["FACETTES"])
    w.writerow(["code", "label", "score_brut"])
    for fac in FACET_ORDER:
        w.writerow([fac, facette_labels[fac], facet_scores[fac]])

    st.download_button("📥 Télécharger CSV (patient)", out.getvalue(), f"{patient_id}_neo_pir.csv", "text/csv")

    # PNG graphs
    fig_radar = plot_domains_radar(domain_scores)
    radar_png = fig_to_png_bytes(fig_radar)
    st.download_button("📥 Profil Domaines (PNG)", radar_png, f"{patient_id}_domaines.png", "image/png")

    fig_facets = plot_facets_line(facet_scores)
    facets_png = fig_to_png_bytes(fig_facets)
    st.download_button("📥 Profil Facettes (PNG)", facets_png, f"{patient_id}_facettes.png", "image/png")

    # PDF report
    pdf_bytes = build_pdf_report_bytes(
        patient_id=patient_id,
        patient_name=patient_name,
        status=status,
        facet_scores=facet_scores,
        domain_scores=domain_scores,
        radar_png=radar_png,
        facets_png=facets_png,
    )
    st.download_button("📥 Télécharger PDF (rapport)", pdf_bytes, f"{patient_id}_neo_pir_report.pdf", "application/pdf")


# ============================================================
# TAB 4 — GLOBAL (export tous patients)
# ============================================================
with tabs[3]:
    st.markdown("## Export global (tous patients)")
    st.caption("Utile pour 200 copies : une seule extraction CSV de toute la base.")

    if st.button("📦 Générer CSV global", type="primary"):
        conn = db()
        col = get_resp_col(conn)
        rows = conn.execute(f"""
            SELECT p.patient_id, COALESCE(p.name,''), r.item_id, r.{col}
            FROM patients p
            LEFT JOIN responses r ON p.patient_id = r.patient_id
            ORDER BY p.patient_id, r.item_id
        """).fetchall()
        conn.close()

        out_all = io.StringIO()
        w2 = csv.writer(out_all)
        w2.writerow(["patient_id", "name", "item_id", "response_idx"])
        for pid, name, item_id, resp in rows:
            w2.writerow([pid, name, item_id, resp])

        st.download_button(
            "📥 Télécharger CSV global",
            out_all.getvalue(),
            "neo_pir_global.csv",
            "text/csv"
        )
