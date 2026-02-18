# app.py — NEO PI-R Calculatrice Clinique PRO (Cabinet 2026)
# ADAOUN YACINE
# ------------------------------------------------------------
# ✅ Saisie item→ 5 boutons XXL (3+2)
# ✅ Auto-avance + transition
# ✅ Flash vert + son discret (optionnel)
# ✅ Mode sombre/clair
# ✅ DB SQLite robuste + migration auto + suppression patient (backup)
# ✅ Scores facettes/domaines + exports CSV/PDF + images
# ------------------------------------------------------------

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
import streamlit.components.v1 as components

import numpy as np
import matplotlib
matplotlib.use("Agg")  # IMPORTANT Streamlit Cloud
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "🧮 NEO PI-R Pro 2026 — ADAOUN YACINE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "neo_pir.db")
SCORING_KEY_CSV = os.path.join(BASE_DIR, "scoring_key.csv")

OPTIONS = ["FD", "D", "N", "A", "FA"]          # index 0..4
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}
IDX_TO_OPT = {i: k for k, i in OPT_TO_IDX.items()}

# Facettes bases (mapping officiel de ton projet)
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
    "N1": "Anxiété", "N2": "Hostilité colérique", "N3": "Dépression", "N4": "Timidité",
    "N5": "Impulsivité", "N6": "Vulnérabilité",
    "E1": "Chaleur", "E2": "Grégarité", "E3": "Affirmation de soi", "E4": "Activité",
    "E5": "Recherche d'excitation", "E6": "Émotions positives",
    "O1": "Imagination", "O2": "Esthétique", "O3": "Sentiments", "O4": "Actions",
    "O5": "Idées", "O6": "Valeurs",
    "A1": "Confiance", "A2": "Franchise", "A3": "Altruisme", "A4": "Compliance",
    "A5": "Modestie", "A6": "Tendresse",
    "C1": "Compétence", "C2": "Ordre", "C3": "Sens du devoir", "C4": "Effort pour réussir",
    "C5": "Autodiscipline", "C6": "Délibération",
}

domain_labels = {"N": "Névrosisme", "E": "Extraversion", "O": "Ouverture", "A": "Agréabilité", "C": "Conscience"}


# ============================================================
# PROTOCOLE
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2  # N


# ============================================================
# SCORING KEY
# ============================================================
@st.cache_data
def load_scoring_key(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{os.path.basename(path)}' introuvable. Mets scoring_key.csv à la racine du repo.")
    key: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"item", "FD", "D", "N", "A", "FA"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"Colonnes CSV attendues: {sorted(required)} | Reçues: {reader.fieldnames}")
        for row in reader:
            item = int(row["item"])
            key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])]

    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplet. Items manquants (ex): {missing[:20]}")
    return key


# ============================================================
# DB (robuste + migration)
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

def _table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
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
        # schema cible
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                patient_id TEXT,
                item_id INTEGER,
                response_idx INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (patient_id, item_id)
            )
        """)

        # migration depuis anciennes versions
        cols = _table_cols(conn, "responses")
        if "response" in cols and "response_idx" not in cols:
            conn.execute("ALTER TABLE responses ADD COLUMN response_idx INTEGER;")
            conn.execute("UPDATE responses SET response_idx = response WHERE response_idx IS NULL;")

        # normaliser NULL -> -1
        cols = _table_cols(conn, "responses")
        if "response_idx" in cols:
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
        backup_path = os.path.join(BASE_DIR, f"backup_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        shutil.copy2(DB_PATH, backup_path)
    else:
        backup_path = ""
    with db_ctx() as conn:
        conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
        conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
    return backup_path

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
            "ON CONFLICT(patient_id, item_id) DO UPDATE SET response_idx=excluded.response_idx, updated_at=CURRENT_TIMESTAMP",
            (patient_id, item_id, response_idx),
        )

def reset_response(patient_id: str, item_id: int):
    save_response(patient_id, item_id, -1)


# ============================================================
# CALCULS
# ============================================================
def apply_protocol_rules(responses: Dict[int, int], rules: ProtocolRules):
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
        status["reasons"].append(f"Trop d'items vides : {n_blank} (≥ {rules.max_blank_invalid})")

    if n_count >= rules.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop de réponses N : {n_count} (≥ {rules.max_N_invalid})")

    new_resp = dict(responses)
    if status["valid"] and 0 < n_blank <= rules.impute_blank_if_leq:
        for it in blanks:
            new_resp[it] = rules.impute_option_index
            status["imputed"] += 1

    return new_resp, status

def compute_scores(scoring_key: Dict[int, List[int]], responses: Dict[int, int]):
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
# GRAPHIQUES + EXPORTS
# ============================================================
def plot_domains_radar(domain_scores: Dict[str, int]):
    labels = ["N", "E", "O", "A", "C"]
    values = [domain_scores[k] for k in labels]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    ax.plot(angles, values, linewidth=2.5)
    ax.fill(angles, values, alpha=0.12)
    ax.set_thetagrids(np.degrees(angles[:-1]), [domain_labels[k] for k in labels])
    ax.set_title("Profil Domaines — scores bruts", pad=20, fontweight="bold")
    ax.grid(True, alpha=0.3)
    return fig

def plot_facets_line(facette_scores: Dict[str, int]):
    order = [f"{d}{i}" for d in "NEOAC" for i in range(1, 7)]
    y = [facette_scores[k] for k in order]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(range(len(order)), y, marker="o", linewidth=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_title("30 Facettes — scores bruts", fontweight="bold")
    ax.set_ylabel("Score brut")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    return fig

def fig_to_bytes(fig, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def build_pdf_report_bytes(patient_id: str, patient_name: str, status: dict,
                           facette_scores: Dict[str, int], domain_scores: Dict[str, int]) -> bytes:
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 50

    c.setFont("Helvetica-Bold", 15)
    c.drawString(40, y, "RAPPORT NEO PI-R — Scores bruts")
    y -= 22

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Patient ID : {patient_id}")
    y -= 14
    c.drawString(40, y, f"Nom : {patient_name}")
    y -= 14
    c.drawString(40, y, f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"STATUT PROTOCOLE : {'VALIDE' if status['valid'] else 'INVALIDE'}")
    y -= 16

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Items vides : {status['n_blank']} | N : {status['n_count']} | Imputations : {status['imputed']}")
    y -= 16

    if status["reasons"]:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Raisons :")
        y -= 12
        c.setFont("Helvetica", 10)
        for r in status["reasons"]:
            c.drawString(50, y, f"- {r}")
            y -= 12
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "DOMAINES")
    y -= 16
    c.setFont("Helvetica", 10)
    for d in ["N", "E", "O", "A", "C"]:
        c.drawString(40, y, f"{d} — {domain_labels[d]} : {domain_scores[d]}")
        y -= 12
    y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "FACETTES")
    y -= 16
    c.setFont("Helvetica", 9)
    for fac in sorted(facette_labels.keys()):
        c.drawString(40, y, f"{fac} — {facette_labels[fac]} : {facette_scores[fac]}")
        y -= 11
        if y < 70:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)

    c.save()
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# FEEDBACK (son discret)
# ============================================================
def play_beep_once(volume: float = 0.25):
    # bip court base64 (wav minimal)
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


# ============================================================
# UI / CSS
# ============================================================
def inject_css(theme: str, flash_ok: bool):
    if theme == "dark":
        bg = "#0e1117"
        panel = "#111827"
        text = "#f9fafb"
        subtle = "#9ca3af"
        border = "rgba(255,255,255,0.10)"
        btn = "#1f2937"
    else:
        bg = "#f7fafc"
        panel = "#ffffff"
        text = "#0f172a"
        subtle = "#64748b"
        border = "rgba(0,0,0,0.10)"
        btn = "#f1f5f9"

    flash_css = ""
    if flash_ok:
        flash_css = """
        .neo-flash { animation: neoFlash 420ms ease-out; }
        @keyframes neoFlash {
          0% { box-shadow: 0 0 0 rgba(34,197,94,0); transform: translateY(0); }
          50% { box-shadow: 0 0 0 10px rgba(34,197,94,0.25); transform: translateY(-2px); }
          100% { box-shadow: 0 0 0 rgba(34,197,94,0); transform: translateY(0); }
        }
        """

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {text}; }}
        .neo-panel {{
          background: {panel};
          border: 1px solid {border};
          border-radius: 22px;
          padding: 18px 18px 16px 18px;
        }}
        .neo-subtle {{ color: {subtle}; }}
        .neo-band {{
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 12px;
          margin: 12px 0 6px 0;
        }}
        .neo-card {{
          background: {panel};
          border: 1px solid {border};
          border-radius: 18px;
          padding: 14px;
          text-align: center;
        }}
        .neo-card .t {{ font-size: 12px; color: {subtle}; margin-bottom: 6px; }}
        .neo-card .v {{ font-size: 26px; font-weight: 900; letter-spacing: -0.02em; }}

        /* Boutons XXL */
        div.stButton > button {{
          width: 100%;
          height: 130px;
          font-size: 52px;
          font-weight: 900;
          border-radius: 24px;
          border: 2px solid {border};
          background: {btn};
          color: {text};
          transition: transform 120ms ease, box-shadow 120ms ease;
        }}
        div.stButton > button:hover {{
          transform: translateY(-2px);
          box-shadow: 0 14px 26px rgba(0,0,0,0.18);
        }}
        div.stButton > button:active {{ transform: translateY(-1px); }}

        /* Bouton reset plus petit */
        .neo-reset div.stButton > button {{
          height: 60px;
          font-size: 18px;
          font-weight: 800;
          border-radius: 16px;
        }}

        /* Responsive */
        @media (max-width: 900px) {{
          .neo-band {{ grid-template-columns: repeat(2, 1fr); }}
          div.stButton > button {{ height: 110px; font-size: 44px; }}
        }}

        {flash_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APP
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")
init_db()
scoring_key = load_scoring_key(SCORING_KEY_CSV)

# Session state defaults
defaults = {
    "theme": "dark",
    "flash_ok": True,
    "sound_ok": True,
    "current_item": 1,
    "last_saved": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
with st.sidebar:
    st.markdown("## 👤 Patient")
    st.session_state["theme"] = st.radio("Apparence", ["dark", "light"], index=0 if st.session_state["theme"] == "dark" else 1)
    st.session_state["flash_ok"] = st.toggle("Flash vert (validation)", value=st.session_state["flash_ok"])
    st.session_state["sound_ok"] = st.toggle("Son discret (validation)", value=st.session_state["sound_ok"])

    st.markdown("---")
    mode = st.radio("Mode", ["Ouvrir", "Créer"], horizontal=True)
    search = st.text_input("Recherche", value="")

    patients = list_patients(search=search)
    patient_id = ""
    patient_name = ""

    if mode == "Ouvrir":
        if patients:
            labels = [f"{pid} — {name or 'Sans nom'}" for pid, name in patients]
            sel = st.selectbox("Patients", labels, index=0)
            patient_id = sel.split(" — ")[0].strip()
            patient_name = next((nm for pid, nm in patients if pid == patient_id), "")
        else:
            st.info("Aucun patient. Passe en mode Créer.")
    else:
        patient_id = st.text_input("Patient ID (unique)", value="")
        patient_name = st.text_input("Nom / Prénom", value="")
        if st.button("✅ Enregistrer", type="primary", disabled=(not patient_id.strip())):
            upsert_patient(patient_id.strip(), patient_name.strip())
            st.success("Enregistré.")
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
    st.markdown("## 🗑️ Gestion")
    if patient_id.strip():
        if st.button("🗑️ Supprimer patient (backup auto)"):
            backup = delete_patient(patient_id.strip())
            if backup:
                st.success("Patient supprimé. Backup créé dans le repo.")
                st.caption(os.path.basename(backup))
            else:
                st.success("Patient supprimé.")
            st.rerun()

# Must select patient
if not patient_id.strip():
    st.title(APP_TITLE)
    st.info("Sélectionne ou crée un patient pour commencer.")
    st.stop()

# Inject UI
inject_css(st.session_state["theme"], st.session_state["flash_ok"])

# Header
st.title(APP_TITLE)
st.caption("Saisie clinique ultra-rapide • Item par item • Calcul temps réel • Exports")

# Load & compute
responses = load_responses(patient_id)
answered = sum(1 for v in responses.values() if v != -1)
remaining = 240 - answered

final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(scoring_key, final_resp)

proto_label = "✅ VALIDE" if (answered == 240 and status["valid"]) else ("🟡 EN COURS" if answered < 240 else "❌ INVALIDE")

# Live band (stats)
st.markdown(
    f"""
    <div class="neo-band">
      <div class="neo-card"><div class="t">Patient</div><div class="v">{patient_id}</div></div>
      <div class="neo-card"><div class="t">Saisis</div><div class="v">{answered}</div></div>
      <div class="neo-card"><div class="t">Restants</div><div class="v">{remaining}</div></div>
      <div class="neo-card"><div class="t">Protocole</div><div class="v">{proto_label}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.progress(answered / 240.0)

if (answered >= 240) and (not status["valid"]):
    st.error("Protocole INVALIDE")
    for r in status["reasons"]:
        st.write("•", r)

tabs = st.tabs(["🧮 Saisie", "📊 Résultats", "📦 Exports"])


# ============================================================
# TAB 1 — SAISIE (item par item)
# ============================================================
with tabs[0]:
    st.markdown('<div class="neo-panel">', unsafe_allow_html=True)

    # Navigation top
    colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
    with colA:
        cur = st.number_input("Item", 1, 240, int(st.session_state["current_item"]), step=1)
        st.session_state["current_item"] = int(cur)
    with colB:
        if st.button("⬅️ -1"):
            st.session_state["current_item"] = max(1, st.session_state["current_item"] - 1)
            st.rerun()
    with colC:
        if st.button("➡️ +1"):
            st.session_state["current_item"] = min(240, st.session_state["current_item"] + 1)
            st.rerun()
    with colD:
        if st.button("⏭️ Prochain vide"):
            start = st.session_state["current_item"]
            found = None
            for i in range(start, 241):
                if responses.get(i, -1) == -1:
                    found = i
                    break
            if found is None:
                for i in range(1, start):
                    if responses.get(i, -1) == -1:
                        found = i
                        break
            if found is not None:
                st.session_state["current_item"] = found
                st.rerun()

    cur_item = int(st.session_state["current_item"])
    current_value = responses.get(cur_item, -1)
    current_label = "VIDE" if current_value == -1 else IDX_TO_OPT[current_value]
    fac = item_to_facette.get(cur_item, "?")
    dom = facettes_to_domain.get(fac, "?") if fac != "?" else "?"
    st.markdown(f"**Item {cur_item}/240** • **Facette {fac}** • **Domaine {dom}**  \nRéponse actuelle : **{current_label}**")

    # reset item
    st.markdown('<div class="neo-reset">', unsafe_allow_html=True)
    if st.button("🧹 Réinitialiser cet item (VIDE)"):
        reset_response(patient_id, cur_item)
        st.session_state["last_saved"] = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Buttons (3 + 2) very large
    flash_class = "neo-flash" if st.session_state["last_saved"] else ""
    st.markdown(f'<div class="{flash_class}">', unsafe_allow_html=True)

    row1 = st.columns(3)
    clicked = None
    if row1[0].button("FD"):
        clicked = 0
    if row1[1].button("D"):
        clicked = 1
    if row1[2].button("N"):
        clicked = 2

    row2 = st.columns(2)
    if row2[0].button("A"):
        clicked = 3
    if row2[1].button("FA"):
        clicked = 4

    st.markdown("</div>", unsafe_allow_html=True)

    # Save + feedback + auto-advance + transition
    if clicked is not None:
        save_response(patient_id, cur_item, int(clicked))
        st.session_state["last_saved"] = True

        if st.session_state["sound_ok"]:
            play_beep_once(volume=0.25)

        # auto next
        if cur_item < 240:
            st.session_state["current_item"] = cur_item + 1

        st.rerun()

    # Clear flash flag after render
    if st.session_state["last_saved"]:
        # une fois affiché, on remet à False pour éviter flash permanent
        st.session_state["last_saved"] = False

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 2 — RESULTATS
# ============================================================
with tabs[1]:
    st.markdown('<div class="neo-panel">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items vides", status["n_blank"])
    c2.metric("Réponses N", status["n_count"])
    c3.metric("Imputations", status["imputed"])
    c4.metric("Statut", "VALIDE" if (answered == 240 and status["valid"]) else "EN COURS" if answered < 240 else "INVALIDE")

    st.markdown("### Domaines")
    st.dataframe(
        [{"Code": d, "Domaine": domain_labels[d], "Score brut": domain_scores[d]} for d in ["N", "E", "O", "A", "C"]],
        hide_index=True,
        use_container_width=True,
    )
    st.pyplot(plot_domains_radar(domain_scores))

    st.markdown("### Facettes")
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

    # CSV complet
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["patient_id", patient_id])
    w.writerow(["name", patient_name])
    w.writerow(["date", datetime.now().isoformat(timespec="minutes")])
    w.writerow([])
    w.writerow(["protocole_valid", int(status["valid"])])
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

    # Images
    radar_png = fig_to_bytes(plot_domains_radar(domain_scores), "png")
    facets_png = fig_to_bytes(plot_facets_line(facette_scores), "png")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Radar (PNG)", radar_png, f"{patient_id}_radar.png", "image/png")
    with col2:
        st.download_button("📥 Facettes (PNG)", facets_png, f"{patient_id}_facettes.png", "image/png")

    st.markdown("</div>", unsafe_allow_html=True)
