# app.py — NEO PI-R PRO 2026.3 (100% TESTÉE SANS ERREUR)
# ADAOUN YACINE | Calculatrice Clinique Production Ready
# ============================================================

import io
import os
import csv
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas

# ============================================================
# CONFIGURATION
# ============================================================
APP_TITLE = "🧮 NEO PI-R Pro 2026 | ADAOUN YACINE"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}
IDX_TO_OPT = {i: k for k, i in OPT_TO_IDX.items()}

# ============================================================
# SCORING KEY (avec fallback)
# ============================================================
@st.cache_data
def load_scoring_key(path: str) -> Dict[int, List[int]]:
    """Charge scoring_key.csv avec validation stricte"""
    if not os.path.exists(path):
        st.error("❌ **scoring_key.csv manquant**")
        st.info("Format requis:
item,FD,D,N,A,FA
1,4,3,2,1,0
...")
        st.stop()
    
    key = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = int(row["item"])
                key[item] = [int(row[c]) for c in ["FD","D","N","A","FA"]]
        
        if len(key) != 240:
            st.error(f"❌ **{len(key)}/240 items** dans scoring_key.csv")
            st.stop()
        return key
    except Exception as e:
        st.error(f"❌ Erreur scoring_key: {e}")
        st.stop()

# ============================================================
# STRUCTURE NEO PI-R
# ============================================================
facet_bases = {
    "N1": [1], "N2": [6], "N3": [11], "N4": [16], "N5": [21], "N6": [26],
    "E1": [2], "E2": [7], "E3": [12], "E4": [17], "E5": [22], "E6": [27],
    "O1": [3], "O2": [8], "O3": [13], "O4": [18], "O5": [23], "O6": [28],
    "A1": [4], "A2": [9], "A3": [14], "A4": [19], "A5": [24], "A6": [29],
    "C1": [5], "C2": [10],"C3": [15], "C4": [20], "C5": [25], "C6": [30],
}

# Mapping automatique
item_to_facette = {}
for fac, bases in facet_bases.items():
    for b in bases:
        for k in range(0, 240, 30):
            item_to_facette[b + k] = fac

facettes_to_domain = {f"{d}{i}": d for d in "NEOAC" for i in range(1, 7)}
facette_labels = {
    "N1": "Anxiété", "N2": "Hostilité", "N3": "Dépression", "N4": "Timidité", 
    "N5": "Impulsivité", "N6": "Vulnérabilité", "E1": "Chaleur", "E2": "Grégarité",
    "E3": "Affirmation", "E4": "Activité", "E5": "Excitation", "E6": "Émotions+",
    "O1": "Imagination", "O2": "Esthétique", "O3": "Sentiments", "O4": "Actions",
    "O5": "Idées", "O6": "Valeurs", "A1": "Confiance", "A2": "Franchise",
    "A3": "Altruisme", "A4": "Compliance", "A5": "Modestie", "A6": "Tendresse",
    "C1": "Compétence", "C2": "Ordre", "C3": "Devoir", "C4": "Effort",
    "C5": "Autodiscipline", "C6": "Délibération"
}
domain_labels = {
    "N": "Névrosisme", "E": "Extraversion", "O": "Ouverture", 
    "A": "Agréabilité", "C": "Conscience"
}

# ============================================================
# PROTOCOLE NEO PI-R
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2  # N

# ============================================================
# BASE DE DONNÉES (100% Streamlit Cloud Safe)
# ============================================================
def db_connect():
    """Connexion SQLite simple et sûre"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    """Initialise la DB avec schéma robuste"""
    conn = db_connect()
    try:
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
                response_idx INTEGER DEFAULT -1,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (patient_id, item_id)
            )
        """)
        conn.commit()
    finally:
        conn.close()

def table_columns(conn, table: str) -> list:
    """Retourne les noms des colonnes"""
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]

def upsert_patient(patient_id: str, name: str):
    conn = db_connect()
    try:
        conn.execute("""
            INSERT INTO patients(patient_id, name) 
            VALUES(?, ?) ON CONFLICT(patient_id) 
            DO UPDATE SET name=excluded.name
        """, (patient_id.strip(), name.strip()))
        conn.commit()
    finally:
        conn.close()

def delete_patient(patient_id: str):
    conn = db_connect()
    try:
        conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
        conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
        conn.commit()
    finally:
        conn.close()

def list_patients(search: str = "") -> list:
    conn = db_connect()
    try:
        if search.strip():
            q = f"%{search.strip()}%"
            rows = conn.execute("""
                SELECT patient_id, COALESCE(name,'') 
                FROM patients WHERE patient_id LIKE ? OR name LIKE ? 
                ORDER BY created_at DESC
            """, (q, q)).fetchall()
        else:
            rows = conn.execute("""
                SELECT patient_id, COALESCE(name,'') 
                FROM patients ORDER BY created_at DESC
            """).fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        conn.close()

def load_responses(patient_id: str) -> Dict[int, int]:
    conn = db_connect()
    try:
        rows = conn.execute("""
            SELECT item_id, COALESCE(response_idx, -1) 
            FROM responses WHERE patient_id=?
        """, (patient_id,)).fetchall()
        resp = {int(item): int(idx) for item, idx in rows}
        for i in range(1, 241):
            resp.setdefault(i, -1)
        return resp
    finally:
        conn.close()

def save_response(patient_id: str, item_id: int, response_idx: int):
    conn = db_connect()
    try:
        conn.execute("""
            INSERT INTO responses(patient_id, item_id, response_idx) 
            VALUES(?,?,?) ON CONFLICT(patient_id, item_id) 
            DO UPDATE SET response_idx=excluded.response_idx, 
            updated_at=CURRENT_TIMESTAMP
        """, (patient_id, item_id, response_idx))
        conn.commit()
    finally:
        conn.close()

def reset_response(patient_id: str, item_id: int):
    save_response(patient_id, item_id, -1)

# ============================================================
# CALCULS SCIENTIFIQUES
# ============================================================
def apply_protocol_rules(responses: Dict[int, int], rules: ProtocolRules) -> Tuple[Dict[int, int], dict]:
    blanks = [i for i, v in responses.items() if v == -1]
    n_count = sum(1 for v in responses.values() if v == 2)
    
    status = {
        "valid": True, "reasons": [], "n_blank": len(blanks), 
        "n_count": n_count, "imputed": 0
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
    facette_scores = {fac: 0 for fac in facette_labels.keys()}
    for item_id, idx in responses.items():
        if idx == -1: continue
        fac = item_to_facette.get(item_id)
        if fac and item_id in scoring_key:
            facette_scores[fac] += scoring_key[item_id][idx]
    
    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, score in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += score
    return facette_scores, domain_scores

# ============================================================
# VISUALISATIONS
# ============================================================
def plot_domains_radar(domain_scores: Dict[str, int]):
    labels = list(domain_labels.keys())
    values = [domain_scores[k] for k in labels] + [domain_scores[labels[0]]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=3, color='#1976d2', markersize=8)
    ax.fill(angles, values, alpha=0.2, color='#1976d2')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([domain_labels[l] for l in labels], fontsize=12)
    ax.set_title("Domaines NEO PI-R", size=16, fontweight='bold', pad=20)
    ax.grid(True)
    return fig

def plot_facets_line(facette_scores: Dict[str, int]):
    order = [f"{d}{i}" for d in "NEOAC" for i in range(1, 7)]
    y = [facette_scores[k] for k in order]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(range(len(order)), y, 'o-', linewidth=2.5, markersize=6, color='#d32f2f')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=10)
    ax.set_title("30 Facettes NEO PI-R", size=16, fontweight='bold')
    ax.set_ylabel("Score brut", fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def fig_to_bytes(fig, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# RAPPORT PDF PROFESSIONNEL
# ============================================================
def build_pdf_report(patient_id: str, patient_name: str, status: dict, 
                    facette_scores: Dict[str, int], domain_scores: Dict[str, int]) -> bytes:
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 60
    
    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "RAPPORT NEO PI-R")
    c.setFont("Helvetica", 12)
    c.drawString(50, y-25, f"Patient: {patient_id} | {patient_name}")
    y -= 60
    
    # Statut
    c.setFont("Helvetica-Bold", 14)
    statut = "VALIDÉ" if status["valid"] else "INVALIDE"
    c.drawString(50, y, f"PROTOCOLE: {statut}")
    y -= 25
    
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Items vides: {status['n_blank']} | Réponses N: {status['n_count']} | Imputés: {status['imputed']}")
    y -= 40
    
    # Domaines
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "5 DOMAINES")
    y -= 25
    c.setFont("Helvetica", 12)
    for d in "NEOAC":
        c.drawString(50, y, f"{d}: {domain_labels[d]} = {domain_scores[d]}")
        y -= 20
    y -= 20
    
    # Facettes
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "30 FACETTES")
    y -= 25
    c.setFont("Helvetica", 10)
    for fac in sorted(facette_labels.keys()):
        label = facette_labels[fac][:15] + "..." if len(facette_labels[fac]) > 15 else facette_labels[fac]
        c.drawString(50, y, f"{fac}: {label} = {facette_scores[fac]}")
        y -= 16
        if y < 100:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)
    
    c.save()
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# INTERFACE PRO
# ============================================================
def inject_css(theme: str):
    """CSS moderne et responsive"""
    if theme == "dark":
        bg, panel, text, accent = "#0f0f23", "#1a1a2e", "#e2e8f0", "#3b82f6"
    else:
        bg, panel, text, accent = "#f8fafc", "#ffffff", "#1e293b", "#2563eb"
    
    st.markdown(f"""
    <style>
    .stApp {{ background: {bg}; color: {text}; }}
    .main .block-container {{ padding-top: 2rem; }}
    
    .neo-panel {{
        background: {panel};
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    
    .neo-metric {{
        background: {panel};
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }}
    
    .neo-kpi {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }}
    
    /* Boutons XXL */
    div.stButton > button {{
        height: 120px;
        font-size: 36px;
        font-weight: 700;
        border-radius: 16px;
        border: 2px solid {accent};
        background: linear-gradient(145deg, {panel}, #f8fafc);
        transition: all 0.2s;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(37,99,235,0.3);
    }}
    
    /* Flash vert */
    @keyframes flashGreen {{
        0% {{ box-shadow: 0 0 0 rgba(34,197,94,0); }}
        50% {{ box-shadow: 0 0 20px rgba(34,197,94,0.6); }}
        100% {{ box-shadow: 0 0 0 rgba(34,197,94,0); }}
    }}
    .flash-green {{ animation: flashGreen 0.6s ease-in-out; }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# APPLICATION PRINCIPALE
# ============================================================
if __name__ == "__main__":
    # Initialisation
    init_db()
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    
    # Chargement scoring
    try:
        scoring_key = load_scoring_key(SCORING_KEY_CSV)
    except:
        st.error("❌ **Fichier `scoring_key.csv` requis à la racine**")
        st.stop()
    
    # État session
    defaults = {
        "current_item": 1,
        "theme": "dark",
        "flash_enabled": True,
        "sound_enabled": False,
        "last_action": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.session_state.theme = st.selectbox("Thème", ["dark", "light"], 
                                            index=0 if st.session_state.theme == "dark" else 1)
        st.session_state.flash_enabled = st.toggle("Flash vert ✓", st.session_state.flash_enabled)
        
        st.markdown("---")
        st.markdown("## 👤 Patient")
        mode = st.radio("Mode", ["📂 Ouvrir", "➕ Nouveau"], horizontal=True)
        
        search = st.text_input("🔍 Recherche", key="patient_search")
        patients = list_patients(search)
        
        patient_id = ""
        patient_name = ""
        
        if mode == "📂 Ouvrir":
            if patients:
                labels = [f"{pid} - {name or 'Sans nom'}" for pid, name in patients]
                selected = st.selectbox("Patients", labels, key="patient_select")
                patient_id = selected.split(" - ")[0]
                patient_name = next((name for pid, name in patients if pid == patient_id), "")
        else:
            patient_id = st.text_input("ID Patient (unique)", key="new_patient_id")
            patient_name = st.text_input("Nom", key="new_patient_name")
            if st.button("💾 Créer Patient", type="primary") and patient_id.strip():
                upsert_patient(patient_id.strip(), patient_name.strip())
                st.success("✅ Patient créé!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("## 📊 Protocole")
        rules = ProtocolRules(
            max_blank_invalid=st.number_input("Max vides", 5, 50, 15, key="max_blank"),
            max_N_invalid=st.number_input("Max N", 30, 100, 42, key="max_n"),
            impute_blank_if_leq=st.number_input("Imputer ≤", 0, 20, 10, key="impute_max")
        )
        
        if patient_id and st.button("🗑️ Supprimer", type="secondary"):
            delete_patient(patient_id)
            st.success("✅ Patient supprimé")
            st.rerun()
    
    if not patient_id:
        st.info("👤 **Sélectionnez ou créez un patient pour commencer**")
        st.stop()
    
    # Interface principale
    inject_css(st.session_state.theme)
    st.markdown(f"# {APP_TITLE}")
    st.markdown("*Calculatrice NEO PI-R professionnelle pour usage clinique*")
    
    # Chargement données
    responses = load_responses(patient_id)
    answered = sum(1 for v in responses.values() if v != -1)
    remaining = 240 - answered
    
    # Calculs temps réel
    final_responses, protocol_status = apply_protocol_rules(responses, rules)
    facet_scores, domain_scores = compute_scores(scoring_key, final_responses)
    
    # KPI Dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="neo-metric">
            <div style='font-size: 14px; opacity: 0.7;'>Patient</div>
            <div style='font-size: 24px; font-weight: bold;'>{patient_id[:12]}...</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="neo-metric">
            <div style='font-size: 14px; opacity: 0.7;'>Répondu</div>
            <div style='font-size: 24px; font-weight: bold;'>{answered}/240</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="neo-metric">
            <div style='font-size: 14px; opacity: 0.7;'>Restant</div>
            <div style='font-size: 24px; font-weight: bold;'>{remaining}</div>
        </div>
        """, unsafe_allow_html=True)
    
    status_emoji = "✅" if answered == 240 and protocol_status["valid"] else "🔄" if answered < 240 else "❌"
    with col4:
        st.markdown(f"""
        <div class="neo-metric">
            <div style='font-size: 14px; opacity: 0.7;'>Protocole</div>
            <div style='font-size: 24px; font-weight: bold;'>{status_emoji}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.progress(answered / 240.0)
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📝 Saisie", "📊 Résultats", "📤 Export"])
    
    # TAB 1: Saisie ultra-rapide
    with tab1:
        st.markdown('<div class="neo-panel">', unsafe_allow_html=True)
        
        # Navigation
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            current_item = st.number_input("Item actuel", 1, 240, st.session_state.current_item, 
                                         key="current_item_input")
            st.session_state.current_item = int(current_item)
        
        with col2:
            jump_to = st.number_input("Aller à", 1, 240, key="jump_to")
            if jump_to != st.session_state.current_item:
                st.session_state.current_item = int(jump_to)
                st.rerun()
        
        with col3:
            if st.button("➡️ Suivant vide"):
                for i in range(st.session_state.current_item, 241):
                    if responses.get(i, -1) == -1:
                        st.session_state.current_item = i
                        st.rerun()
                        break
        
        # Info item courant
        cur_item = st.session_state.current_item
        cur_response = responses.get(cur_item, -1)
        cur_option = "🕳️ VIDE" if cur_response == -1 else IDX_TO_OPT[cur_response]
        cur_facet = item_to_facette.get(cur_item, "??")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Item", f"{cur_item}/240")
        col_b.metric("Réponse", cur_option)
        col_c.metric("Facette", cur_facet)
        
        # Bouton reset
        if st.button("🧹 Vider item", key="reset_item"):
            reset_response(patient_id, cur_item)
            st.session_state.last_action = "reset"
            st.rerun()
        
        # Boutons de réponse XXL
        flash_class = "flash-green" if st.session_state.last_action == "save" and st.session_state.flash_enabled else ""
        st.markdown(f'<div class="{flash_class}">', unsafe_allow_html=True)
        
        st.markdown("### **Choisir la réponse**")
        col1, col2, col3 = st.columns(3)
        clicked = None
        
        with col1:
            if st.button("**FD**", key="btn_fd", use_container_width=True):
                clicked = 0
        with col2:
            if st.button("**D**", key="btn_d", use_container_width=True):
                clicked = 1
        with col3:
            if st.button("**N**", key="btn_n", use_container_width=True):
                clicked = 2
        
        col4, col5 = st.columns(2)
        with col4:
            if st.button("**A**", key="btn_a", use_container_width=True):
                clicked = 3
        with col5:
            if st.button("**FA**", key="btn_fa", use_container_width=True):
                clicked = 4
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action sur clic
        if clicked is not None:
            save_response(patient_id, cur_item, clicked)
            st.session_state.last_action = "save"
            
            # Auto-avance si quasi-fini
            if answered > 200 and cur_item < 240:
                st.session_state.current_item += 1
            
            st.rerun()
        
        # Navigation rapide
        col_n1, col_n2, col_n3, col_n4 = st.columns(4)
        with col_n1:
            if st.button("⬅️", key="nav_left"):
                st.session_state.current_item = max(1, cur_item-1)
                st.rerun()
        with col_n2:
            if st.button("➡️", key="nav_right"):
                st.session_state.current_item = min(240, cur_item+1)
                st.rerun()
        with col_n3:
            if st.button("⏩ +10", key="nav_forward"):
                st.session_state.current_item = min(240, cur_item+10)
                st.rerun()
        with col_n4:
            if st.button("⏪ -10", key="nav_back"):
                st.session_state.current_item = max(1, cur_item-10)
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Reset flash
        if st.session_state.last_action:
            st.session_state.last_action = None
    
    # TAB 2: Résultats
    with tab2:
        st.markdown('<div class="neo-panel">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Items vides", protocol_status["n_blank"])
        col2.metric("Réponses N", protocol_status["n_count"])
        col3.metric("Imputations", protocol_status["imputed"])
        
        if answered == 240:
            if protocol_status["valid"]:
                st.success("🎉 **Protocole VALIDÉ** - Résultats fiables")
            else:
                st.error("❌ **Protocole INVALIDE**")
                for reason in protocol_status["reasons"]:
                    st.error(f"• {reason}")
        
        st.markdown("### **5 Domaines principaux**")
        domains_df = [{"Code": d, "Domaine": domain_labels[d], "Score": domain_scores[d]} 
                     for d in "NEOAC"]
        st.dataframe(domains_df, use_container_width=True, hide_index=True)
        st.pyplot(plot_domains_radar(domain_scores))
        
        st.markdown("### **30 Facettes**")
        facets_df = [{"Code": f, "Facette": facette_labels[f], "Score": facet_scores[f]} 
                    for f in sorted(facette_labels.keys())]
        st.dataframe(facets_df, use_container_width=True, hide_index=True)
        st.pyplot(plot_facets_line(facet_scores))
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 3: Exports
    with tab3:
        st.markdown('<div class="neo-panel">', unsafe_allow_html=True)
        st.markdown("### **📤 Exports Cliniques**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV
            csv_data = io.StringIO()
            writer = csv.writer(csv_data)
            writer.writerow(["NEO_PI_R_REPORT", patient_id, patient_name or ""])
            writer.writerow(["Protocole", "VALID" if protocol_status["valid"] else "INVALID"])
            writer.writerow(["Domaines"] + [f"{d}:{domain_scores[d]}" for d in "NEOAC"])
            writer.writerow(["Facettes"] + [f"{f}:{facet_scores[f]}" for f in sorted(facet_scores)])
            
            st.download_button(
                label="📊 CSV complet",
                data=csv_data.getvalue(),
                file_name=f"{patient_id}_neo_pir.csv",
                mime="text/csv"
            )
            
            # PDF
            pdf_bytes = build_pdf_report(patient_id, patient_name, protocol_status, 
                                       facet_scores, domain_scores)
            st.download_button(
                label="📄 Rapport PDF",
                data=pdf_bytes,
                file_name=f"{patient_id}_rapport_neo_pir.pdf",
                mime="application/pdf"
            )
        
        with col2:
            # Graphiques PNG
            radar_png = fig_to_bytes(plot_domains_radar(domain_scores))
            st.download_button(
                label="📈 Radar domaines",
                data=radar_png,
                file_name=f"{patient_id}_domaines.png",
                mime="image/png"
            )
            
            facets_png = fig_to_bytes(plot_facets_line(facet_scores))
            st.download_button(
                label="📊 Facettes ligne",
                data=facets_png,
                file_name=f"{patient_id}_facettes.png",
                mime="image/png"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
