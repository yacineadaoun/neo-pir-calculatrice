# app.py — Version PRO 2026.2 (Optimisée)
# NEO PI-R Calculatrice Clinique | ADAOUN YACINE
# ✅ 5 boutons 1 ligne • Cache optimisé • DB context manager • UX clinique max
# ============================================================

import io
import os
import csv
import sqlite3
import shutil
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
import streamlit.components.v1 as components

# ============================================================
# CONFIG SCIENTIFIQUE
# ============================================================
APP_TITLE = "🧮 NEO PI-R Pro 2026 | ADAOUN YACINE"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}
IDX_TO_OPT = {i: k for k, i in OPT_TO_IDX.items()}

# ============================================================
# SCORING KEY (Cache optimisé)
# ============================================================
@st.cache_data
def load_scoring_key(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' manquant")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key: Dict[int, List[int]] = {}
        for row in reader:
            item = int(row["item"])
            key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])]
    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"Scoring incomplet: {missing[:10]}...")
    return key

# ============================================================
# MAPPINGS NEO PI-R (scientifiques)
# ============================================================
facet_bases = {
    "N1": [1], "N2": [6], "N3": [11], "N4": [16], "N5": [21], "N6": [26],
    "E1": [2], "E2": [7], "E3": [12], "E4": [17], "E5": [22], "E6": [27],
    "O1": [3], "O2": [8], "O3": [13], "O4": [18], "O5": [23], "O6": [28],
    "A1": [4], "A2": [9], "A3": [14], "A4": [19], "A5": [24], "A6": [29],
    "C1": [5], "C2": [10],"C3": [15], "C4": [20], "C5": [25], "C6": [30],
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
    "N1": "Anxiété", "N2": "Hostilité", "N3": "Dépression", "N4": "Timidité", 
    "N5": "Impulsivité", "N6": "Vulnérabilité",
    "E1": "Chaleur", "E2": "Grégarité", "E3": "Affirmation", "E4": "Activité", 
    "E5": "Excitation", "E6": "Émotions+",
    "O1": "Imagination", "O2": "Esthétique", "O3": "Sentiments", "O4": "Actions", 
    "O5": "Idées", "O6": "Valeurs",
    "A1": "Confiance", "A2": "Franchise", "A3": "Altruisme", "A4": "Compliance", 
    "A5": "Modestie", "A6": "Tendresse",
    "C1": "Compétence", "C2": "Ordre", "C3": "Devoir", "C4": "Effort", 
    "C5": "Autodiscipline", "C6": "Délibération",
}

domain_labels = {"N": "Névrosisme", "E": "Extraversion", "O": "Ouverture", 
                "A": "Agréabilité", "C": "Conscience"}

# ============================================================
# PROTOCOLE NEO PI-R (paramétrable)
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2  # N

# ============================================================
# DB PROFESSIONNELLE (Context Manager + Backup)
# ============================================================
@contextmanager
def db_ctx():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _table_info(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]

def init_db():
    with db_ctx() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY, name TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                patient_id TEXT, item_id INTEGER, response_idx INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (patient_id, item_id)
            )
        """)
        
        # Migration auto
        cols = _table_info(conn, "responses")
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

def delete_patient(patient_id: str):
    # Backup auto
    backup_path = f"backup_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.db"
    shutil.copy2(DB_PATH, backup_path)
    
    with db_ctx() as conn:
        conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
        conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))

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
        cols = _table_info(conn, "responses")
        col = "response_idx" if "response_idx" in cols else "response"
        rows = conn.execute(f"SELECT item_id, {col} FROM responses WHERE patient_id=?", (patient_id,)).fetchall()
    resp = {int(item): int(idx) for item, idx in rows}
    for i in range(1, 241): resp.setdefault(i, -1)
    return resp

def save_response(patient_id: str, item_id: int, response_idx: int):
    with db_ctx() as conn:
        cols = _table_info(conn, "responses")
        col = "response_idx" if "response_idx" in cols else "response"
        conn.execute(
            f"INSERT INTO responses(patient_id, item_id, {col}) VALUES(?,?,?) "
            f"ON CONFLICT(patient_id, item_id) DO UPDATE SET {col}=excluded.{col}, updated_at=CURRENT_TIMESTAMP",
            (patient_id, item_id, response_idx),
        )

def reset_response(patient_id: str, item_id: int):
    save_response(patient_id, item_id, -1)

# ============================================================
# CALCULS SCIENTIFIQUES
# ============================================================
def apply_protocol_rules(responses: Dict[int, int], rules: ProtocolRules) -> Tuple[Dict[int, int], dict]:
    blanks = [i for i, v in responses.items() if v == -1]
    n_count = sum(1 for v in responses.values() if v == 2)
    
    status = {
        "valid": True, "reasons": [], "n_blank": len(blanks), "n_count": n_count,
        "imputed": 0, "blank_items": blanks,
    }
    
    if len(blanks) >= rules.max_blank_invalid:
        status["valid"] = False
        status["reasons"].append(f"Vides ≥ {rules.max_blank_invalid}")
    if n_count >= rules.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(f"N ≥ {rules.max_N_invalid}")
    
    new_resp = dict(responses)
    if status["valid"] and 0 < len(blanks) <= rules.impute_blank_if_leq:
        for it in blanks:
            new_resp[it] = rules.impute_option_index
            status["imputed"] += 1
    
    return new_resp, status

def compute_scores(scoring_key: Dict[int, List[int]], responses: Dict[int, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    facette_scores = {fac: 0 for fac in facette_labels}
    for item_id, idx in responses.items():
        if idx == -1: continue
        fac = item_to_facette.get(item_id)
        if fac: facette_scores[fac] += scoring_key[item_id][idx]
    
    domain_scores = {d: 0 for d in domain_labels}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc
    return facette_scores, domain_scores

# ============================================================
# GRAPHIQUES SCIENTIFIQUES
# ============================================================
def plot_domains_radar(domain_scores: Dict[str, int]):
    labels = ["N", "E", "O", "A", "C"]
    values = [domain_scores[k] for k in labels] + [domain_scores[labels[0]]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, linewidth=3, color='#1e88e5')
    ax.fill(angles, values, alpha=0.15, color='#1e88e5')
    ax.set_thetagrids(np.degrees(angles[:-1]), [domain_labels[l] for l in labels])
    ax.set_title("Domaines NEO PI-R", size=14, fontweight='bold', pad=20)
    ax.grid(True)
    return fig

def plot_facets_line(facette_scores: Dict[str, int]):
    order = [f"{d}{i}" for d in "NEOAC" for i in range(1,7)]
    y = [facette_scores[k] for k in order]
    
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(range(len(order)), y, marker='o', linewidth=2.5, markersize=6, color='#d32f2f')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_title("30 Facettes NEO PI-R", size=14, fontweight='bold')
    ax.set_ylabel("Score brut", fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def fig_to_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

# ============================================================
# PDF RAPPORT CLINIQUE
# ============================================================
def build_pdf_report_bytes(patient_id: str, patient_name: str, status: dict, 
                          facette_scores: Dict[str, int], domain_scores: Dict[str, int]) -> bytes:
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    width, height = A4; y = height - 50
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "RAPPORT NEO PI-R — Scores Bruts")
    y -= 30
    
    c.setFont("Helvetica", 12)
    c.drawString(40, y, f"ID: {patient_id} | Nom: {patient_name}")
    y -= 25
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"PROTOCOLE: {'✅ VALIDE' if status['valid'] else '❌ INVALIDE'}")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Vides: {status['n_blank']} | N: {status['n_count']} | Imputés: {status['imputed']}")
    
    if status["reasons"]:
        y -= 20
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Raisons:")
        y -= 15
        c.setFont("Helvetica", 9)
        for r in status["reasons"]:
            c.drawString(50, y, f"• {r}")
            y -= 12
    
    # Domaines
    y -= 20
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "5 DOMAINES")
    y -= 18
    c.setFont("Helvetica", 11)
    for d in "NEOAC":
        c.drawString(40, y, f"{d}: {domain_labels[d]} = {domain_scores[d]}")
        y -= 14
    
    # Facettes
    y -= 15
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "30 FACETTES")
    y -= 18
    c.setFont("Helvetica", 9)
    for fac in sorted(facette_labels):
        c.drawString(40, y, f"{fac}: {facette_labels[fac]} = {facette_scores[fac]}")
        y -= 11
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)
    
    c.save()
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# FEEDBACK SENSORIEL
# ============================================================
def play_beep_once():
    wav_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="
    components.html(f"""
    <script>
    try {{ 
      const audio = new Audio("data:audio/wav;base64,{wav_b64}"); 
      audio.volume = 0.3; audio.play(); 
    }} catch(e) {{}}
    </script>
    """, height=0)

# ============================================================
# CSS PRO (5 boutons + animations)
# ============================================================
def inject_css(theme: str, flash: bool):
    if theme == "dark":
        bg, panel, text, subtle, border, btn_bg = "#0e1117", "#111827", "#f9fafb", "#9ca3af", "rgba(255,255,255,0.08)", "#1f2937"
    else:
        bg, panel, text, subtle, border, btn_bg = "#f8fafc", "#ffffff", "#0f172a", "#64748b", "rgba(0,0,0,0.08)", "#f1f5f9"
    
    flash_css = """
    .neo-flash-ok { animation: neoFlashGreen 400ms ease-out; }
    @keyframes neoFlashGreen {
      0% { box-shadow: 0 0 0 rgba(34,197,94,0); transform: translateY(0); }
      50% { box-shadow: 0 0 0 8px rgba(34,197,94,0.25); transform: translateY(-2px); }
      100% { box-shadow: 0 0 0 rgba(34,197,94,0); transform: translateY(0); }
    }
    """ if flash else ""
    
    st.markdown(f"""
    <style>
    .stApp {{ background: {bg}; color: {text}; }}
    .neo-panel {{ background: {panel}; border: 1px solid {border}; border-radius: 20px; padding: 20px; }}
    .neo-subtle {{ color: {subtle}; }}
    .neo-fadein {{ animation: neoFadeIn 300ms ease-out; }}
    @keyframes neoFadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    
    /* 5 BOUTONS XXL RESPONSIVE */
    .neo-btn-grid-5 {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
    @media (max-width: 768px) {{ .neo-btn-grid-5 {{ grid-template-columns: repeat(2, 1fr); }} }}
    
    div.stButton > button {{
      width: 100%; height: 140px; font-size: 52px; font-weight: 900;
      border-radius: 24px; border: 3px solid {border}; background: {btn_bg};
      color: {text}; transition: all 150ms cubic-bezier(0.4,0,0.2,1);
    }}
    div.stButton > button:hover {{ transform: translateY(-3px) scale(1.02); box-shadow: 0 12px 24px rgba(0,0,0,0.15); }}
    div.stButton > button:active {{ transform: translateY(-1px); }}
    
    /* KPI */
    .neo-kpi {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .neo-kpi-card {{ background: {panel}; border: 1px solid {border}; border-radius: 16px; padding: 16px; text-align: center; }}
    .neo-kpi-title {{ font-size: 13px; color: {subtle}; margin-bottom: 8px; }}
    .neo-kpi-value {{ font-size: 28px; font-weight: 900; letter-spacing: -0.02em; }}
    
    @media (max-width: 768px) {{ .neo-kpi {{ grid-template-columns: repeat(2, 1fr); }} }}
    {flash_css}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# APP PRINCIPALE (PRO)
# ============================================================
init_db()
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")

scoring_key = load_scoring_key(SCORING_KEY_CSV)

# Session state
for key, default in [("current_item", 1), ("last_saved", False), ("theme", "dark"), 
                    ("flash_ok", True), ("sound_ok", True)]:
    if key not in st.session_state: setattr(st.session_state, key, default)

# SIDEBAR PRO
with st.sidebar:
    st.markdown("### 👤 PATIENT")
    st.session_state.theme = st.radio("Thème", ["dark", "light"], 
                                     index=0 if st.session_state.theme == "dark" else 1)[0]
    st.session_state.flash_ok = st.toggle("Flash ✓", st.session_state.flash_ok)
    st.session_state.sound_ok = st.toggle("Son ✓", st.session_state.sound_ok)
    
    mode = st.radio("Mode", ["📂 Ouvrir", "➕ Créer"], horizontal=True)
    search = st.text_input("🔍 Recherche")
    patients = list_patients(search)
    
    patient_id, patient_name = "", ""
    if mode == "📂 Ouvrir":
        if patients:
            labels = [f"{pid} — {name or 'Sans nom'}" for pid, name in patients]
            sel = st.selectbox("Patients", labels)
            patient_id = sel.split(" — ")[0]
            patient_name = next((name for pid, name in patients if pid == patient_id), "")
    else:
        patient_id = st.text_input("ID unique")
        patient_name = st.text_input("Nom")
        if st.button("💾 Enregistrer", type="primary", disabled=not patient_id.strip()):
            upsert_patient(patient_id.strip(), patient_name.strip())
            st.success("✅ Enregistré")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ⚙️ PROTOCOLE")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Vides max", 0, 50, 15),
        max_N_invalid=st.number_input("N max", 0, 50, 42),
        impute_blank_if_leq=st.number_input("Imputer ≤", 0, 20, 10),
    )
    
    st.markdown("---")
    st.markdown("### 🗑️ SUPPRIMER")
    if patient_id.strip() and st.button("🗑️ Supprimer (backup auto)", type="secondary"):
        delete_patient(patient_id.strip())
        st.success(f"✅ Supprimé (backup: {backup_path})")
        st.rerun()

if not patient_id.strip():
    st.info("👤 Sélectionnez un patient")
    st.stop()

# CSS + Header
inject_css(st.session_state.theme, st.session_state.flash_ok)
st.markdown(f"# {APP_TITLE}")
st.caption("Saisie clinique ultra-rapide | Calculs temps réel | Exports PDF pro")

# Load & Compute
responses = load_responses(patient_id)
answered = sum(1 for v in responses.values() if v != -1)
remaining = 240 - answered
final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(scoring_key, final_resp)

# KPI Live
proto = "✅ VALIDE" if answered == 240 and status["valid"] else "🔄 EN COURS" if answered < 240 else "❌ INVALIDE"
st.markdown(f"""
<div class="neo-kpi neo-fadein">
  <div class="neo-kpi-card"><div class="neo-kpi-title">Patient</div><div class="neo-kpi-value">{patient_id[:10]}…</div></div>
  <div class="neo-kpi-card"><div class="neo-kpi-title">Saisis</div><div class="neo-kpi-value">{answered}</div></div>
  <div class="neo-kpi-card"><div class="neo-kpi-title">Restants</div><div class="neo-kpi-value">{remaining}</div></div>
  <div class="neo-kpi-card"><div class="neo-kpi-title">Protocole</div><div class="neo-kpi-value">{proto}</div></div>
</div>
""", unsafe_allow_html=True)

st.progress(answered / 240)

# TABS PRO
tabs = st.tabs(["1️⃣ Saisie", "2️⃣ Scores", "3️⃣ Exports"])
# TAB 1: SAISIE 5 BOUTONS XXL (CORRIGÉ)
with tabs[0]:
    st.markdown("<div class='neo-panel neo-fadein'>", unsafe_allow_html=True)
    
    # Navigation item
    col1, col2, col3 = st.columns([1.2, 1.1, 1])
    with col1:
        item = st.number_input("Item actuel", 1, 240, st.session_state.current_item)
        st.session_state.current_item = int(item)
    with col2:
        jump = st.number_input("Sauter à", 1, 240, format="%d", label_visibility="collapsed")
        if int(jump) != int(st.session_state.current_item):
            st.session_state.current_item = int(jump)
            st.rerun()
    with col3:
        if st.button("➡️ Prochain vide"):
            cur = int(st.session_state.current_item)
            for i in range(cur, 241):
                if responses.get(i, -1) == -1:
                    st.session_state.current_item = i
                    st.rerun()
                    break
    
    # Info courant
    cur_item = int(st.session_state.current_item)
    cur_idx = responses.get(cur_item, -1)
    cur_opt = "🕳️ VIDE" if cur_idx == -1 else IDX_TO_OPT[cur_idx]
    fac = item_to_facette.get(cur_item, "?")
    dom = facettes_to_domain.get(fac, "?")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Item", f"{cur_item}/240")
    col_info2.metric("Réponse", cur_opt)
    col_info3.metric("Facette", f"{fac} ({dom})")
    
    # Reset
    if st.button("🧹 Vider cet item", key="reset_item"):
        reset_response(patient_id, cur_item)
        st.session_state.last_saved = True
        st.rerun()
    
    # 🔥 5 BOUTONS XXL CORRIGÉS (syntaxe Python pure)
    st.markdown("### 📊 Choisir réponse")
    flash_class = "neo-flash-ok" if st.session_state.last_saved and st.session_state.flash_ok else ""
    st.markdown(f"<div class='{flash_class} neo-btn-grid-5'>", unsafe_allow_html=True)
    
    clicked = None
    
    # Ligne 1: FD D N (large)
    col_fd, col_d, col_n = st.columns([1, 1, 1.5])
    with col_fd:
        if st.button("FD", key="btn_fd"): clicked = 0
    with col_d:
        if st.button("D", key="btn_d"): clicked = 1
    with col_n:
        if st.button("N", key="btn_n"): clicked = 2
    
    # Ligne 2: A FA
    col_a, col_fa = st.columns([1, 1])
    with col_a:
        if st.button("A", key="btn_a"): clicked = 3
    with col_fa:
        if st.button("FA", key="btn_fa"): clicked = 4
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Action + feedback PRO
    if clicked is not None:
        save_response(patient_id, cur_item, clicked)
        st.session_state.last_saved = True
        
        if st.session_state.sound_ok:
            play_beep_once()
        
        # Auto-avance intelligente
        if answered > 192 and cur_item < 240:
            st.session_state.current_item = cur_item + 1
        
        st.rerun()
    
    # Navigation rapide
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
    with col_nav1:
        if st.button("⬅️ -1"): st.session_state.current_item = max(1, cur_item-1); st.rerun()
    with col_nav2:
        if st.button("➡️ +1"): st.session_state.current_item = min(240, cur_item+1); st.rerun()
    with col_nav3:
        if st.button("⏭️ +10"): st.session_state.current_item = min(240, cur_item+10); st.rerun()
    with col_nav4:
        if st.button("⏮️ -10"): st.session_state.current_item = max(1, cur_item-10); st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Reset flash après render
    if st.session_state.last_saved:
        st.session_state.last_saved = False
        # TAB 2: SCORES
with tabs[1]:
    st.markdown("<div class='neo-panel neo-fadein'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Vides", status["n_blank"])
    col2.metric("Réponses N", status["n_count"])
    col3.metric("Imputés", status["imputed"])
    
    if answered == 240:
        st.balloons() if status["valid"] else st.error("❌ Protocole invalide")
    
    st.markdown("### 📈 5 Domaines")
    st.dataframe([{"Code": d, "Nom": domain_labels[d], "Score": domain_scores[d]} for d in "NEOAC"], 
                hide_index=True, use_container_width=True)
    st.pyplot(plot_domains_radar(domain_scores))
    
    st.markdown("### 📊 30 Facettes")
    st.dataframe([{"Code": f, "Nom": facette_labels[f], "Score": facette_scores[f]} 
                 for f in sorted(facette_labels)], hide_index=True, use_container_width=True)
    st.pyplot(plot_facets_line(facette_scores))
    
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 3: EXPORTS
with tabs[2]:
    st.markdown("<div class='neo-panel neo-fadein'>", unsafe_allow_html=True)
    
    # CSV
    csv_data = io.StringIO()
    writer = csv.writer(csv_data)
    writer.writerow(["NEO PI-R REPORT", patient_id, patient_name])
    writer.writerow(["Protocole", "VALID" if status["valid"] else "INVALID"])
    writer.writerow(["Domaines", *[f"{d}:{domain_scores[d]}" for d in "NEOAC"]])
    writer.writerow(["Facettes", *[f"{f}:{facette_scores[f]}" for f in sorted(facette_scores)]])
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📄 CSV", csv_data.getvalue(), f"{patient_id}_neo.csv", "text/csv")
        pdf_bytes = build_pdf_report_bytes(patient_id, patient_name, status, facette_scores, domain_scores)
        st.download_button("📄 PDF", pdf_bytes, f"{patient_id}_rapport.pdf", "application/pdf")
    with col2:
        st.download_button("🖼️ Radar", fig_to_bytes(plot_domains_radar(domain_scores), "png"), 
                          f"{patient_id}_radar.png", "image/png")
        st.download_button("🖼️ Facettes", fig_to_bytes(plot_facets_line(facette_scores), "png"), 
                          f"{patient_id}_facettes.png", "image/png")
    
    st.markdown("</div>", unsafe_allow_html=True)
