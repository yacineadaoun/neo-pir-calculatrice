import io
import os
import csv
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "NEO PI-R — by ADAOUN YACINE"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # index 0..4
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}
IDX_TO_OPT = {i: k for i, k in enumerate(OPTIONS)}

TEXT_ALIASES = {
    "FD": "FD", "F": "FD", "0": "FD",
    "D": "D", "1": "D",
    "N": "N", "NEUTRE": "N", "2": "N",
    "A": "A", "3": "A",
    "FA": "FA", "4": "FA",
}

BUTTON_META = [
    {"label": "FD", "idx": 0, "emoji": "⬅️", "help": "Fortement en désaccord"},
    {"label": "D",  "idx": 1, "emoji": "◀️", "help": "Désaccord"},
    {"label": "N",  "idx": 2, "emoji": "⏺️", "help": "Neutre"},
    {"label": "A",  "idx": 3, "emoji": "▶️", "help": "Accord"},
    {"label": "FA", "idx": 4, "emoji": "➡️", "help": "Fortement d'accord"},
]
EMPTY_META = {"label": "VIDE", "idx": -1, "emoji": "🧽", "help": "Marquer vide"}

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
@st.cache_resource
def load_scoring_key(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' introuvable. Ajoute scoring_key.csv à la racine du repo.")
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


scoring_key = load_scoring_key(SCORING_KEY_CSV)


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
    impute_option_index: int = 2  # N


# ============================================================
# 4) DB (SQLite) + SETTINGS
# ============================================================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


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
        response_idx INTEGER,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (patient_id, item_id)
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        patient_id TEXT PRIMARY KEY,
        current_item INTEGER DEFAULT 1,
        current_row INTEGER DEFAULT 1,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


def upsert_patient(patient_id: str, name: str):
    conn = db()
    conn.execute(
        "INSERT INTO patients(patient_id, name) VALUES(?, ?) "
        "ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name",
        (patient_id, name)
    )
    conn.commit()
    conn.close()


def list_patients() -> List[Tuple[str, str]]:
    conn = db()
    rows = conn.execute("SELECT patient_id, COALESCE(name,'') FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def load_responses(patient_id: str) -> Dict[int, int]:
    conn = db()
    rows = conn.execute("SELECT item_id, response_idx FROM responses WHERE patient_id=?", (patient_id,)).fetchall()
    conn.close()
    resp = {int(item): int(idx) for item, idx in rows}
    for i in range(1, 241):
        resp.setdefault(i, -1)
    return resp


def save_response(patient_id: str, item_id: int, response_idx: int):
    conn = db()
    conn.execute(
        "INSERT INTO responses(patient_id, item_id, response_idx) VALUES(?,?,?) "
        "ON CONFLICT(patient_id, item_id) DO UPDATE SET response_idx=excluded.response_idx, updated_at=CURRENT_TIMESTAMP",
        (patient_id, item_id, response_idx)
    )
    conn.commit()
    conn.close()


def save_many(patient_id: str, items: List[int], idxs: List[int]):
    conn = db()
    conn.executemany(
        "INSERT INTO responses(patient_id, item_id, response_idx) VALUES(?,?,?) "
        "ON CONFLICT(patient_id, item_id) DO UPDATE SET response_idx=excluded.response_idx, updated_at=CURRENT_TIMESTAMP",
        [(patient_id, it, ix) for it, ix in zip(items, idxs)]
    )
    conn.commit()
    conn.close()


def load_settings(patient_id: str) -> Tuple[int, int]:
    conn = db()
    row = conn.execute("SELECT current_item, current_row FROM settings WHERE patient_id=?", (patient_id,)).fetchone()
    conn.close()
    if row:
        return int(row[0]), int(row[1])
    return 1, 1


def save_settings(patient_id: str, current_item: int, current_row: int):
    conn = db()
    conn.execute(
        "INSERT INTO settings(patient_id, current_item, current_row) VALUES(?,?,?) "
        "ON CONFLICT(patient_id) DO UPDATE SET current_item=excluded.current_item, current_row=excluded.current_row, updated_at=CURRENT_TIMESTAMP",
        (patient_id, current_item, current_row)
    )
    conn.commit()
    conn.close()


def clear_patient(patient_id: str):
    conn = db()
    conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
    conn.execute("DELETE FROM settings WHERE patient_id=?", (patient_id,))
    conn.commit()
    conn.close()


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


def compute_scores(responses: Dict[int, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
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
# 6) SAISIE (parsing ligne)
# ============================================================
def normalize_token(tok: str) -> Optional[str]:
    t = tok.strip().upper()
    if not t:
        return None
    if t == "FA":
        return "FA"
    return TEXT_ALIASES.get(t)


def parse_line_8(text: str) -> Tuple[Optional[List[int]], str]:
    if text is None:
        return None, "vide"
    raw = text.replace(",", " ").replace(";", " ").replace("/", " ").replace("|", " ")
    toks = [t for t in raw.split() if t.strip()]
    if len(toks) != 8:
        return None, f"Il faut 8 réponses (tu as {len(toks)})."
    out = []
    for t in toks:
        nt = normalize_token(t)
        if nt is None:
            return None, f"Token invalide: '{t}'. Autorisés: FD D N A FA (ou 0..4)."
        out.append(OPT_TO_IDX[nt])
    return out, "ok"


# ============================================================
# 7) GRAPHIQUES
# ============================================================
def plot_domains_radar(domain_scores: Dict[str, int]):
    labels = DOMAIN_ORDER
    values = [domain_scores[k] for k in labels]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Domaines (scores bruts)")
    return fig


def plot_facets_line(facette_scores: Dict[str, int]):
    order = FACET_ORDER
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
# 8) PDF REPORT
# ============================================================
def build_pdf_report_bytes(
    patient_id: str,
    patient_name: str,
    status: dict,
    facette_scores: Dict[str, int],
    domain_scores: Dict[str, int]
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "RAPPORT NEO PI-R — Scores bruts")
    y -= 25

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Patient ID: {patient_id}")
    y -= 16
    c.drawString(40, y, f"Nom: {patient_name}")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"STATUT PROTOCOLE: {'VALIDE' if status['valid'] else 'INVALIDE'}")
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Items vides: {status['n_blank']} | N observés: {status['n_count']} | Imputations: {status['imputed']}")
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

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "DOMAINES (scores bruts)")
    y -= 16
    c.setFont("Helvetica", 10)
    for d in DOMAIN_ORDER:
        c.drawString(40, y, f"{domain_labels[d]} ({d}): {domain_scores[d]}")
        y -= 12
    y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "FACETTES (scores bruts)")
    y -= 16
    c.setFont("Helvetica", 9)
    for fac in FACET_ORDER:
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
# 9) UI (PRO)
# ============================================================
init_db()
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")

# --- CSS PRO: très gros boutons + sélection + layout ---
st.markdown("""
<style>
/* titres */
h1 {letter-spacing: .2px;}
/* gros boutons */
.big-btn div.stButton > button{
    height: 86px !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    border-radius: 22px !important;
    width: 100% !important;
}
/* bouton vide */
.big-btn-empty div.stButton > button{
    height: 86px !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    border-radius: 22px !important;
    width: 100% !important;
    opacity: 0.9;
}
/* pastille réponse actuelle */
.pill{
    display:inline-block;
    padding:8px 12px;
    border-radius:999px;
    font-weight:800;
    font-size:16px;
    border:1px solid rgba(255,255,255,.2);
    background: rgba(255,255,255,.06);
}
/* carte */
.card{
    border:1px solid rgba(255,255,255,.08);
    border-radius:18px;
    padding:14px 16px;
    background: rgba(255,255,255,.03);
}
</style>
""", unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption("✅ Mode manuel ultra-rapide (200 copies) • ✅ Reprise automatique • ✅ Vérification • ✅ Exports • ✅ Graphiques")


# ---------- Sidebar : Patient + règles ----------
with st.sidebar:
    st.subheader("👤 Patient")
    mode = st.radio("Mode", ["Ouvrir patient", "Créer patient"], index=0)

    existing = list_patients()
    if mode == "Ouvrir patient":
        if existing:
            labels = [f"{pid} — {name}" if name else pid for pid, name in existing]
            pick = st.selectbox("Sélection", labels, index=0)
            patient_id = pick.split(" — ")[0].strip()
            patient_name = dict(existing).get(patient_id, "")
        else:
            st.warning("Aucun patient. Crée un patient.")
            patient_id = ""
            patient_name = ""
    else:
        patient_id = st.text_input("Patient ID (unique)", value="")
        patient_name = st.text_input("Nom / Prénom", value="")

    st.markdown("---")
    st.subheader("🧾 Règles protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses N ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2
    )

    st.markdown("---")
    st.subheader("⚙️ Outils")
    debug = st.toggle("Debug", value=False)


if mode == "Créer patient":
    if st.button("✅ Créer / enregistrer", type="primary", disabled=(not patient_id.strip())):
        upsert_patient(patient_id.strip(), patient_name.strip())
        st.success("Patient enregistré. Passe en mode 'Ouvrir patient'.")
        st.stop()

if not patient_id.strip():
    st.info("Choisis ou crée un patient pour commencer.")
    st.stop()


# ---------- Charger données ----------
responses = load_responses(patient_id)
saved_item, saved_row = load_settings(patient_id)

if "current_item" not in st.session_state:
    st.session_state.current_item = int(saved_item)

if "current_row" not in st.session_state:
    st.session_state.current_row = int(saved_row)

# ---------- KPI + progression ----------
answered = sum(1 for i in range(1, 241) if responses[i] != -1)
st.progress(answered / 240.0)
k1, k2, k3 = st.columns([1, 1, 1])
k1.metric("Réponses saisies", f"{answered}/240")
k2.metric("Restant", 240 - answered)
k3.metric("Patient", patient_id)

final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(final_resp)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Items vides", status["n_blank"])
c2.metric("N observés", status["n_count"])
c3.metric("Imputations", status["imputed"])
c4.metric("Protocole", "VALIDE" if status["valid"] else "INVALIDE")

if not status["valid"]:
    st.error("Protocole INVALIDE")
    for r in status["reasons"]:
        st.write("•", r)

# ---------- Tabs ----------
tabs = st.tabs(["🧮 Saisie rapide", "✅ Vérification", "📊 Résultats", "📦 Exports"])


# ============================================================
# TAB 1 — SAISIE RAPIDE
# ============================================================
with tabs[0]:
    left, right = st.columns([1.15, 0.85], gap="large")

    # ------------------ Mode Item ------------------
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1) Mode Item (gros boutons)")
        topA, topB = st.columns([1, 1])
        with topA:
            item = st.number_input("Item", 1, 240, int(st.session_state.current_item), step=1)
            st.session_state.current_item = int(item)
        with topB:
            goto = st.text_input("Aller à…", value="", placeholder="ex: 120")
            if goto.strip().isdigit():
                gi = int(goto.strip())
                if 1 <= gi <= 240:
                    st.session_state.current_item = gi
                    save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
                    st.rerun()

        current_idx = responses[int(st.session_state.current_item)]
        current_label = "VIDE" if current_idx == -1 else IDX_TO_OPT[current_idx]
        st.markdown(f"Réponse actuelle : <span class='pill'> {current_label} </span>", unsafe_allow_html=True)

        # boutons (2 lignes)
        row1 = st.columns(3)
        row2 = st.columns(3)

        clicked = None
        btn_cols = [row1[0], row1[1], row1[2], row2[0], row2[1]]
        for col, meta in zip(btn_cols, BUTTON_META):
            with col:
                st.markdown("<div class='big-btn'>", unsafe_allow_html=True)
                if st.button(f"{meta['emoji']} {meta['label']}", help=meta["help"], use_container_width=True):
                    clicked = meta["idx"]
                st.markdown("</div>", unsafe_allow_html=True)

        with row2[2]:
            st.markdown("<div class='big-btn-empty'>", unsafe_allow_html=True)
            if st.button(f"{EMPTY_META['emoji']} {EMPTY_META['label']}", help=EMPTY_META["help"], use_container_width=True):
                clicked = -1
            st.markdown("</div>", unsafe_allow_html=True)

        # appliquer clic + autopass
        if clicked is not None:
            save_response(patient_id, int(st.session_state.current_item), int(clicked))
            nxt = min(240, int(st.session_state.current_item) + 1)
            st.session_state.current_item = nxt
            save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
            st.rerun()

        nav = st.columns(4)
        if nav[0].button("⬅️ Précédent", use_container_width=True):
            st.session_state.current_item = max(1, st.session_state.current_item - 1)
            save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
            st.rerun()
        if nav[1].button("➡️ Suivant", use_container_width=True):
            st.session_state.current_item = min(240, st.session_state.current_item + 1)
            save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
            st.rerun()
        if nav[2].button("⏭️ +10", use_container_width=True):
            st.session_state.current_item = min(240, st.session_state.current_item + 10)
            save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
            st.rerun()
        if nav[3].button("🧹 Reset patient", use_container_width=True):
            clear_patient(patient_id)
            st.success("Réponses effacées.")
            st.session_state.current_item = 1
            st.session_state.current_row = 1
            save_settings(patient_id, 1, 1)
            st.rerun()

        st.caption("Raccourcis (optionnel) : 1=FD 2=D 3=N 4=A 5=FA 0=VIDE (si ton navigateur/Streamlit les accepte).")
        st.markdown("</div>", unsafe_allow_html=True)

        if debug:
            st.markdown("**Debug item**")
            st.write("item_to_facette:", item_to_facette.get(int(st.session_state.current_item)))
            st.write("scoring_key[item]:", scoring_key.get(int(st.session_state.current_item)))


    # ------------------ Mode Ligne ------------------
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2) Mode Ligne (8 réponses d’un coup)")

        row = st.number_input("Ligne (1..30)", 1, 30, int(st.session_state.current_row), step=1)
        st.session_state.current_row = int(row)

        st.write(f"Items de cette ligne : **{row}, {row+30}, {row+60}, {row+90}, {row+120}, {row+150}, {row+180}, {row+210}**")
        st.caption("Format : 8 tokens séparés par espaces. Ex: `N A D FA N N A FD` (ou 0..4).")

        line_text = st.text_input("Saisie ligne", value="", placeholder="N A D FA N N A FD")
        colv1, colv2 = st.columns([1, 1])
        if colv1.button("✅ Valider", type="primary", use_container_width=True):
            idxs, msg = parse_line_8(line_text)
            if idxs is None:
                st.error(msg)
            else:
                items = [int(row) + 30 * c for c in range(8)]
                save_many(patient_id, items, idxs)

                # auto: ligne suivante
                st.session_state.current_row = min(30, int(row) + 1)
                save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
                st.success("Ligne enregistrée ✅")
                st.rerun()

        if colv2.button("➡️ Ligne suivante", use_container_width=True):
            st.session_state.current_row = min(30, int(row) + 1)
            save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
            st.rerun()

        st.markdown("---")
        st.markdown("### Contrôle rapide (ligne)")
        show_row = int(row)
        row_items = [show_row + 30*c for c in range(8)]
        row_vals = ["—" if responses[it] == -1 else IDX_TO_OPT[responses[it]] for it in row_items]
        st.write("Réponses :", row_vals)

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 2 — VERIFICATION
# ============================================================
with tabs[1]:
    st.subheader("Vérification rapide (items vides / navigation 1 clic)")

    blanks = status["blank_items"]
    st.write(f"Items vides : **{len(blanks)}**")

    if blanks:
        # afficher en grille (20 par ligne visuel)
        cols = st.columns(10)
        for i, it in enumerate(blanks[:200]):  # garde-fou
            with cols[i % 10]:
                if st.button(f"{it}", use_container_width=True):
                    st.session_state.current_item = int(it)
                    save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
                    st.success(f"Navigation vers item {it}")
                    st.rerun()
        if len(blanks) > 200:
            st.info("Liste très longue — affichage limité à 200.")
    else:
        st.success("Aucun item vide ✅")

    st.markdown("---")
    st.subheader("Checklist")
    st.write("✅ Vérifie les items marqués VIDE (si c'est intentionnel).")
    st.write("✅ Vérifie si tu as trop de réponses N (>= seuil).")
    st.write("✅ Export quand tout est OK.")


# ============================================================
# TAB 3 — RESULTATS
# ============================================================
with tabs[2]:
    st.subheader("Résultats (temps réel)")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Domaines")
        dom_table = [{"Domaine": domain_labels[d], "Code": d, "Score brut": domain_scores[d]} for d in DOMAIN_ORDER]
        st.dataframe(dom_table, hide_index=True, use_container_width=True)

        fig1 = plot_domains_radar(domain_scores)
        st.pyplot(fig1)

    with col2:
        st.markdown("### Facettes")
        fac_rows = [{"Facette": facette_labels[fac], "Code": fac, "Score brut": facette_scores[fac]} for fac in FACET_ORDER]
        st.dataframe(fac_rows, hide_index=True, use_container_width=True)

        fig2 = plot_facets_line(facette_scores)
        st.pyplot(fig2)

    if debug:
        st.markdown("### Debug")
        st.write("status:", status)


# ============================================================
# TAB 4 — EXPORTS
# ============================================================
with tabs[3]:
    st.subheader("Exports (CSV + PDF + Images)")

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
    for d in DOMAIN_ORDER:
        w.writerow([d, domain_labels[d], domain_scores[d]])
    w.writerow([])
    w.writerow(["FACETTES"])
    w.writerow(["code", "label", "score_brut"])
    for fac in FACET_ORDER:
        w.writerow([fac, facette_labels[fac], facette_scores[fac]])

    st.download_button("📥 Télécharger CSV", out.getvalue(), f"{patient_id}_neo_pir.csv", "text/csv")

    pdf_bytes = build_pdf_report_bytes(patient_id, patient_name, status, facette_scores, domain_scores)
    st.download_button("📥 Télécharger PDF", pdf_bytes, f"{patient_id}_neo_pir_report.pdf", "application/pdf")

    fig_radar = plot_domains_radar(domain_scores)
    st.download_button("📥 Profil Domaines (PNG)", fig_to_bytes(fig_radar, "png"), f"{patient_id}_domains.png", "image/png")

    fig_fac = plot_facets_line(facette_scores)
    st.download_button("📥 Profil Facettes (PNG)", fig_to_bytes(fig_fac, "png"), f"{patient_id}_facettes.png", "image/png")
