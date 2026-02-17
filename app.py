import io
import os
import csv
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "NEO PI-R — Calculatrice Pro (ADAOUN YACINE)"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}
IDX_TO_OPT = {i: k for i, k in enumerate(OPTIONS)}

TEXT_ALIASES = {
    "FD": "FD", "F": "FD", "0": "FD",
    "D": "D", "1": "D",
    "N": "N", "NEUTRE": "N", "NEUTRAL": "N", "2": "N",
    "A": "A", "3": "A",
    "FA": "FA", "4": "FA",
}

DOMAIN_ORDER = ["N", "E", "O", "A", "C"]
FACET_ORDER = [
    "N1","N2","N3","N4","N5","N6",
    "E1","E2","E3","E4","E5","E6",
    "O1","O2","O3","O4","O5","O6",
    "A1","A2","A3","A4","A5","A6",
    "C1","C2","C3","C4","C5","C6",
]


# ============================================================
# 1) SCORING KEY
# ============================================================
@st.cache_data(show_spinner=False)
def load_scoring_key_csv(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' introuvable. Mets scoring_key.csv à la racine du repo."
        )
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key: Dict[int, List[int]] = {}
        for row in reader:
            item = int(row["item"])
            key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])]

    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplet. Items manquants: {missing[:30]}")
    return key


# ============================================================
# 2) ITEM -> FACETTE
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
    'A4': 'A4 - Compliance', 'A5': 'A5 - Modestie', 'A6': 'A6 - Tendresse',
    'C1': 'C1 - Compétence', 'C2': 'C2 - Ordre', 'C3': 'C3 - Sens du devoir',
    'C4': 'C4 - Effort pour réussir', 'C5': 'C5 - Autodiscipline', 'C6': 'C6 - Délibération'
}
domain_labels = {'N': 'Névrosisme', 'E': 'Extraversion', 'O': 'Ouverture', 'A': 'Agréabilité', 'C': 'Conscience'}


# ============================================================
# 3) PROTOCOLE
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2


# ============================================================
# 4) DB
# ============================================================
@st.cache_resource(show_spinner=False)
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        patient_id TEXT,
        item_id INTEGER,
        response_idx INTEGER,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (patient_id, item_id)
    )""")
    conn.commit()


def upsert_patient(patient_id: str, name: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO patients(patient_id, name) VALUES(?, ?) "
        "ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name",
        (patient_id, name)
    )
    conn.commit()


def list_patients() -> List[Tuple[str, str]]:
    conn = get_conn()
    rows = conn.execute("SELECT patient_id, COALESCE(name,'') AS name FROM patients ORDER BY created_at DESC").fetchall()
    return [(r["patient_id"], r["name"]) for r in rows]


def load_responses(patient_id: str) -> Dict[int, int]:
    conn = get_conn()
    rows = conn.execute("SELECT item_id, response_idx FROM responses WHERE patient_id=?", (patient_id,)).fetchall()
    resp = {int(r["item_id"]): int(r["response_idx"]) for r in rows}
    for i in range(1, 241):
        resp.setdefault(i, -1)
    return resp


def save_response(patient_id: str, item_id: int, response_idx: int):
    conn = get_conn()
    conn.execute(
        "INSERT INTO responses(patient_id, item_id, response_idx) VALUES(?,?,?) "
        "ON CONFLICT(patient_id, item_id) DO UPDATE SET response_idx=excluded.response_idx, updated_at=CURRENT_TIMESTAMP",
        (patient_id, item_id, response_idx)
    )
    conn.commit()


# ============================================================
# 5) CALCUL
# ============================================================
def apply_protocol_rules(responses: Dict[int, int], rules: ProtocolRules) -> Tuple[Dict[int, int], dict]:
    blanks = [i for i, v in responses.items() if v == -1]
    n_blank = len(blanks)
    n_count = sum(1 for v in responses.values() if v == 2)

    status = {"valid": True, "reasons": [], "n_blank": n_blank, "n_count": n_count, "imputed": 0, "blank_items": blanks}
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


def compute_scores(responses: Dict[int, int], scoring_key: Dict[int, List[int]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    facette_scores = {fac: 0 for fac in facette_labels.keys()}
    for item_id, idx in responses.items():
        if idx == -1:
            continue
        fac = item_to_facette.get(item_id)
        if fac:
            facette_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc
    return facette_scores, domain_scores


# ============================================================
# UI HELPERS
# ============================================================
def item_from_rc(r: int, c: int) -> int:
    return r + 30 * c  # r in 1..30, c in 0..7


def color_for_idx(idx: int) -> str:
    if idx == -1:
        return "#2a2a2a"
    # palette distincte
    return ["#7c3aed", "#2563eb", "#16a34a", "#f59e0b", "#ef4444"][idx]


# ============================================================
# APP
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")
st.title(APP_TITLE)
st.caption("Interface grille 30×8 + gros boutons • Correction rapide (200 copies)")

st.markdown("""
<style>
.cellbtn button{height:34px !important; font-size:12px !important; font-weight:800 !important; border-radius:10px !important; width:100% !important;}
.bigbtn button{height:84px !important; font-size:28px !important; font-weight:900 !important; border-radius:22px !important; width:100% !important;}
.panel{border:1px solid rgba(255,255,255,.12); border-radius:18px; padding:12px 14px; background:rgba(255,255,255,.03);}
</style>
""", unsafe_allow_html=True)

init_db()
scoring_key = load_scoring_key_csv(SCORING_KEY_CSV)

with st.sidebar:
    st.subheader("Patient")
    mode = st.radio("Mode", ["Ouvrir", "Créer"], index=0)
    existing = list_patients()

    if mode == "Ouvrir":
        if existing:
            labels = [f"{pid} — {name}" if name else pid for pid, name in existing]
            pick = st.selectbox("Sélection", labels, index=0)
            patient_id = pick.split(" — ")[0].strip()
            patient_name = dict(existing).get(patient_id, "")
        else:
            st.warning("Aucun patient. Crée un patient.")
            st.stop()
    else:
        patient_id = st.text_input("Patient ID", value="")
        patient_name = st.text_input("Nom", value="")
        if st.button("✅ Créer", type="primary", disabled=(not patient_id.strip())):
            upsert_patient(patient_id.strip(), patient_name.strip())
            st.success("Créé ✅ (repasse en Ouvrir)")
            st.stop()

    st.markdown("---")
    st.subheader("Protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses N ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2
    )

responses = load_responses(patient_id)

if "sel_r" not in st.session_state:
    st.session_state.sel_r = 1
if "sel_c" not in st.session_state:
    st.session_state.sel_c = 0

# header progress
answered = sum(1 for i in range(1, 241) if responses[i] != -1)
st.progress(answered / 240.0)
st.write(f"Progression : **{answered}/240**")

final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(final_resp, scoring_key)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Vides", status["n_blank"])
k2.metric("N", status["n_count"])
k3.metric("Imputés", status["imputed"])
k4.metric("Protocole", "VALIDE" if status["valid"] else "INVALIDE")

tabs = st.tabs(["🧾 Grille", "📊 Résultats", "📦 Exports"])

# ============================================================
# TAB GRILLE
# ============================================================
with tabs[0]:
    left, right = st.columns([2.2, 1], gap="large")

    # ----------- GRID 30x8
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Grille 30×8 (clique pour éditer)")

        # colon headers
        header = st.columns(8)
        for j in range(8):
            header[j].markdown(f"**Col {j+1}**")

        # grid rows
        for r in range(1, 31):
            cols = st.columns(8, gap="small")
            for c in range(8):
                it = item_from_rc(r, c)
                idx = responses[it]
                label = "—" if idx == -1 else IDX_TO_OPT[idx]
                bg = color_for_idx(idx)

                # button style per cell via markdown hack
                cols[c].markdown(
                    f"""
                    <div class="cellbtn">
                        <style>
                        div[data-testid="stVerticalBlock"] > div:has(> div.cellbtn-{it}) button {{
                            background: {bg} !important;
                            color: white !important;
                        }}
                        </style>
                        <div class="cellbtn-{it}"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if cols[c].button(f"{it}:{label}", key=f"cell_{it}", use_container_width=True):
                    st.session_state.sel_r = r
                    st.session_state.sel_c = c

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------- RIGHT PANEL (selected item)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        sel_item = item_from_rc(int(st.session_state.sel_r), int(st.session_state.sel_c))
        cur = responses[sel_item]
        cur_label = "VIDE" if cur == -1 else IDX_TO_OPT[cur]

        st.subheader("Éditeur rapide")
        st.write(f"Item sélectionné: **{sel_item}**")
        st.write(f"Réponse actuelle: **{cur_label}**")
        st.write(f"Facette: **{item_to_facette.get(sel_item,'?')}**")

        # Big buttons in 2 rows
        b1, b2 = st.columns(2)
        with b1:
            st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
            if st.button("FD", use_container_width=True):
                save_response(patient_id, sel_item, 0)
                st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with b2:
            st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
            if st.button("D", use_container_width=True):
                save_response(patient_id, sel_item, 1)
                st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        b3, b4 = st.columns(2)
        with b3:
            st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
            if st.button("N", use_container_width=True):
                save_response(patient_id, sel_item, 2)
                st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with b4:
            st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
            if st.button("A", use_container_width=True):
                save_response(patient_id, sel_item, 3)
                st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
        if st.button("FA", use_container_width=True):
            save_response(patient_id, sel_item, 4)
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🧽 VIDE", use_container_width=True):
            save_response(patient_id, sel_item, -1)
            st.experimental_rerun()

        st.markdown("---")
        if st.button("➡️ Aller au prochain VIDE", use_container_width=True):
            blanks = [i for i, v in responses.items() if v == -1]
            if blanks:
                nxt = blanks[0]
                # convertir item->(r,c)
                r = ((nxt - 1) % 30) + 1
                c = (nxt - 1) // 30
                st.session_state.sel_r = r
                st.session_state.sel_c = c
                st.experimental_rerun()
            else:
                st.success("Aucun vide ✅")

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB RESULTATS
# ============================================================
with tabs[1]:
    st.subheader("Résultats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Domaines")
        st.dataframe(
            [{"Code": d, "Domaine": domain_labels[d], "Score brut": domain_scores[d]} for d in DOMAIN_ORDER],
            hide_index=True, use_container_width=True
        )
    with col2:
        st.markdown("### Facettes")
        st.dataframe(
            [{"Code": f, "Facette": facette_labels[f], "Score brut": facette_scores[f]} for f in FACET_ORDER],
            hide_index=True, use_container_width=True
        )


# ============================================================
# TAB EXPORTS (minimal)
# ============================================================
with tabs[2]:
    st.subheader("Exports")
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["patient_id", patient_id])
    w.writerow(["name", patient_name])
    w.writerow(["valid", status["valid"]])
    w.writerow(["blanks", status["n_blank"]])
    w.writerow(["n_count", status["n_count"]])
    w.writerow(["imputed", status["imputed"]])
    w.writerow([])
    w.writerow(["DOMAINES"])
    for d in DOMAIN_ORDER:
        w.writerow([d, domain_labels[d], domain_scores[d]])
    w.writerow([])
    w.writerow(["FACETTES"])
    for f in FACET_ORDER:
        w.writerow([f, facette_labels[f], facette_scores[f]])

    st.download_button("📥 Télécharger CSV", out.getvalue(), f"{patient_id}_neo_pir.csv", "text/csv")
