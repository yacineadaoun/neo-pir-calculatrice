import io
import os
import csv
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import numpy as np

import matplotlib
matplotlib.use("Agg")  # IMPORTANT pour Streamlit Cloud
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "NEO PI-R — Calculatrice (by ADAOUN YACINE)"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # index 0..4
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

BUTTONS = [
    ("FD", 0, "⬅️", "Fortement en désaccord"),
    ("D",  1, "◀️", "Désaccord"),
    ("N",  2, "⏺️", "Neutre"),
    ("A",  3, "▶️", "Accord"),
    ("FA", 4, "➡️", "Fortement d'accord"),
]
EMPTY_BTN = ("VIDE", -1, "🧽", "Marquer vide")


# ============================================================
# 1) SCORING KEY
# ============================================================
@st.cache_data(show_spinner=False)
def load_scoring_key_csv(path: str) -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ '{path}' introuvable.\n\n"
            f"➡️ Mets le fichier 'scoring_key.csv' à la racine du repo GitHub."
        )

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"item", "FD", "D", "N", "A", "FA"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "❌ scoring_key.csv n’a pas les bonnes colonnes.\n"
                "Colonnes attendues: item, FD, D, N, A, FA"
            )

        key: Dict[int, List[int]] = {}
        for row in reader:
            item = int(row["item"])
            key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])]

    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"❌ scoring_key.csv incomplet. Items manquants (ex): {missing[:30]}")
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
    impute_option_index: int = 2  # N


# ============================================================
# 4) DB (SQLite) ROBUSTE
# ============================================================
@st.cache_resource(show_spinner=False)
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_db():
    conn = get_conn()
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


def save_many(patient_id: str, items: List[int], idxs: List[int]):
    conn = get_conn()
    conn.executemany(
        "INSERT INTO responses(patient_id, item_id, response_idx) VALUES(?,?,?) "
        "ON CONFLICT(patient_id, item_id) DO UPDATE SET response_idx=excluded.response_idx, updated_at=CURRENT_TIMESTAMP",
        [(patient_id, it, ix) for it, ix in zip(items, idxs)]
    )
    conn.commit()


def load_settings(patient_id: str) -> Tuple[int, int]:
    conn = get_conn()
    row = conn.execute("SELECT current_item, current_row FROM settings WHERE patient_id=?", (patient_id,)).fetchone()
    if row:
        return int(row["current_item"]), int(row["current_row"])
    return 1, 1


def save_settings(patient_id: str, current_item: int, current_row: int):
    conn = get_conn()
    conn.execute(
        "INSERT INTO settings(patient_id, current_item, current_row) VALUES(?,?,?) "
        "ON CONFLICT(patient_id) DO UPDATE SET current_item=excluded.current_item, current_row=excluded.current_row, updated_at=CURRENT_TIMESTAMP",
        (patient_id, current_item, current_row)
    )
    conn.commit()


def clear_patient(patient_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
    conn.execute("DELETE FROM settings WHERE patient_id=?", (patient_id,))
    conn.commit()


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


def compute_scores(responses: Dict[int, int], scoring_key: Dict[int, List[int]]) -> Tuple[Dict[str, int], Dict[str, int]]:
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
# 6) PARSING LIGNE
# ============================================================
def normalize_token(tok: str) -> Optional[str]:
    t = tok.strip().upper()
    if not t:
        return None
    if t == "FA":
        return "FA"
    return TEXT_ALIASES.get(t)


def parse_line_8(text: str) -> Tuple[Optional[List[int]], str]:
    raw = (text or "").replace(",", " ").replace(";", " ").replace("/", " ").replace("|", " ")
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
    values = [domain_scores[k] for k in labels] + [domain_scores[labels[0]]]
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
    y = [facette_scores[k] for k in FACET_ORDER]
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.plot(range(len(FACET_ORDER)), y, marker="o", linewidth=2)
    ax.set_xticks(range(len(FACET_ORDER)))
    ax.set_xticklabels(FACET_ORDER, rotation=60, ha="right")
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
# 8) PDF
# ============================================================
def build_pdf_report_bytes(patient_id: str, patient_name: str, status: dict,
                           facette_scores: Dict[str, int], domain_scores: Dict[str, int]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "RAPPORT NEO PI-R — Scores bruts")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Patient ID: {patient_id}")
    y -= 14
    c.drawString(40, y, f"Nom: {patient_name}")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"STATUT PROTOCOLE: {'VALIDE' if status['valid'] else 'INVALIDE'}")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Items vides: {status['n_blank']} | N observés: {status['n_count']} | Imputations: {status['imputed']}")
    y -= 16

    if status["reasons"]:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Raisons:")
        y -= 12
        c.setFont("Helvetica", 10)
        for r in status["reasons"]:
            c.drawString(50, y, f"- {r}")
            y -= 11
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "DOMAINES")
    y -= 14
    c.setFont("Helvetica", 10)
    for d in DOMAIN_ORDER:
        c.drawString(40, y, f"{domain_labels[d]} ({d}): {domain_scores[d]}")
        y -= 12
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "FACETTES")
    y -= 14
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
# UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")
st.title(APP_TITLE)
st.caption("Saisie manuelle rapide • Sauvegarde SQLite • Calcul immédiat • Exports CSV/PDF • Graphiques")

# CSS (gros boutons + lisible)
st.markdown("""
<style>
.big button{height:82px !important; font-size:28px !important; font-weight:900 !important; border-radius:22px !important; width:100% !important;}
.big2 button{height:62px !important; font-size:20px !important; font-weight:800 !important; border-radius:18px !important; width:100% !important;}
.pill{display:inline-block; padding:8px 12px; border-radius:999px; font-weight:900; background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);}
.card{border:1px solid rgba(255,255,255,.10); border-radius:18px; padding:14px 16px; background: rgba(255,255,255,.03);}
</style>
""", unsafe_allow_html=True)

# init + scoring_key safe
init_db()
try:
    scoring_key = load_scoring_key_csv(SCORING_KEY_CSV)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.subheader("👤 Patient")
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
            patient_id, patient_name = "", ""
    else:
        patient_id = st.text_input("Patient ID (unique)", value="")
        patient_name = st.text_input("Nom / Prénom", value="")

    st.markdown("---")
    st.subheader("🧾 Protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses N ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2
    )

    st.markdown("---")
    debug = st.toggle("Debug", value=False)

    if mode == "Créer":
        if st.button("✅ Créer / enregistrer", type="primary", disabled=(not patient_id.strip())):
            upsert_patient(patient_id.strip(), patient_name.strip())
            st.success("Patient enregistré ✅ (repasse en mode Ouvrir)")
            st.stop()

if not patient_id.strip():
    st.info("Choisis ou crée un patient pour commencer.")
    st.stop()

# data
responses = load_responses(patient_id)
saved_item, saved_row = load_settings(patient_id)

if "current_item" not in st.session_state:
    st.session_state.current_item = saved_item
if "current_row" not in st.session_state:
    st.session_state.current_row = saved_row

answered = sum(1 for i in range(1, 241) if responses[i] != -1)
st.progress(answered / 240.0)
st.write(f"Progression: **{answered}/240**")

final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(final_resp, scoring_key)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Items vides", status["n_blank"])
k2.metric("N observés", status["n_count"])
k3.metric("Imputations", status["imputed"])
k4.metric("Protocole", "VALIDE" if status["valid"] else "INVALIDE")

if not status["valid"]:
    st.error("Protocole INVALIDE")
    for r in status["reasons"]:
        st.write("•", r)

tabs = st.tabs(["🧮 Saisie", "✅ Vérification", "📊 Résultats", "📦 Exports"])

# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------
with tabs[0]:
    cA, cB = st.columns([1.2, 1], gap="large")

    with cA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Mode Item (gros boutons)")

        top1, top2 = st.columns([1, 1])
        with top1:
            item = st.number_input("Item", 1, 240, int(st.session_state.current_item), step=1)
        with top2:
            goto = st.text_input("Aller à", value="", placeholder="ex: 120")
            if goto.strip().isdigit():
                gi = int(goto.strip())
                if 1 <= gi <= 240:
                    st.session_state.current_item = gi
                    save_settings(patient_id, gi, st.session_state.current_row)
                    st.experimental_rerun()

        item = int(item)
        st.session_state.current_item = item
        current_idx = responses[item]
        current_label = "VIDE" if current_idx == -1 else IDX_TO_OPT[current_idx]
        st.markdown(f"Réponse actuelle: <span class='pill'>{current_label}</span>", unsafe_allow_html=True)

        # FORM = évite les bugs de rerun
        with st.form("form_item", clear_on_submit=False):
            row1 = st.columns(3)
            row2 = st.columns(3)
            clicked = None

            cols = [row1[0], row1[1], row1[2], row2[0], row2[1]]
            for col, (lab, idx, emo, help_) in zip(cols, BUTTONS):
                with col:
                    st.markdown("<div class='big'>", unsafe_allow_html=True)
                    if st.form_submit_button(f"{emo} {lab}", help=help_, use_container_width=True):
                        clicked = idx
                    st.markdown("</div>", unsafe_allow_html=True)

            with row2[2]:
                st.markdown("<div class='big'>", unsafe_allow_html=True)
                if st.form_submit_button(f"{EMPTY_BTN[2]} {EMPTY_BTN[0]}", help=EMPTY_BTN[3], use_container_width=True):
                    clicked = -1
                st.markdown("</div>", unsafe_allow_html=True)

            # appliquer après soumission
            if clicked is not None:
                save_response(patient_id, item, int(clicked))
                nxt = min(240, item + 1)
                st.session_state.current_item = nxt
                save_settings(patient_id, nxt, st.session_state.current_row)
                st.success("Enregistré ✅")
                st.experimental_rerun()

        nav = st.columns(4)
        if nav[0].button("⬅️ Précédent", use_container_width=True):
            st.session_state.current_item = max(1, item - 1)
            save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
            st.experimental_rerun()
        if nav[1].button("➡️ Suivant", use_container_width=True):
            st.session_state.current_item = min(240, item + 1)
            save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
            st.experimental_rerun()
        if nav[2].button("⏭️ +10", use_container_width=True):
            st.session_state.current_item = min(240, item + 10)
            save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
            st.experimental_rerun()
        if nav[3].button("🧹 Reset patient", use_container_width=True):
            clear_patient(patient_id)
            st.session_state.current_item = 1
            st.session_state.current_row = 1
            save_settings(patient_id, 1, 1)
            st.success("Patient réinitialisé ✅")
            st.experimental_rerun()

        if debug:
            st.write("Facette:", item_to_facette.get(item))
            st.write("Score map:", scoring_key.get(item))

        st.markdown("</div>", unsafe_allow_html=True)

    with cB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Mode Ligne (8 réponses d’un coup)")

        row = st.number_input("Ligne (1..30)", 1, 30, int(st.session_state.current_row), step=1)
        row = int(row)
        st.session_state.current_row = row

        items = [row + 30*c for c in range(8)]
        st.write("Items:", items)

        with st.form("form_line", clear_on_submit=True):
            line_text = st.text_input("Saisie 8 réponses", value="", placeholder="N A D FA N N A FD")
            ok = st.form_submit_button("✅ Valider la ligne", use_container_width=True)
            if ok:
                idxs, msg = parse_line_8(line_text)
                if idxs is None:
                    st.error(msg)
                else:
                    save_many(patient_id, items, idxs)
                    st.success("Ligne enregistrée ✅")
                    st.session_state.current_row = min(30, row + 1)
                    save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
                    st.experimental_rerun()

        # contrôle
        row_vals = ["—" if responses[it] == -1 else IDX_TO_OPT[responses[it]] for it in items]
        st.write("Contrôle:", row_vals)

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Vérification")
    blanks = status["blank_items"]
    st.write(f"Items vides: **{len(blanks)}**")

    if blanks:
        grid = st.columns(10)
        for i, it in enumerate(blanks[:200]):
            with grid[i % 10]:
                if st.button(str(it), use_container_width=True):
                    st.session_state.current_item = int(it)
                    save_settings(patient_id, st.session_state.current_item, st.session_state.current_row)
                    st.experimental_rerun()
        if len(blanks) > 200:
            st.info("Affichage limité à 200 items.")
    else:
        st.success("Aucun item vide ✅")

# ------------------------------------------------------------
# TAB 3
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Résultats")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("### Domaines")
        dom_table = [{"Domaine": domain_labels[d], "Code": d, "Score brut": domain_scores[d]} for d in DOMAIN_ORDER]
        st.dataframe(dom_table, hide_index=True, use_container_width=True)
        st.pyplot(plot_domains_radar(domain_scores))

    with c2:
        st.markdown("### Facettes")
        fac_table = [{"Facette": facette_labels[f], "Code": f, "Score brut": facette_scores[f]} for f in FACET_ORDER]
        st.dataframe(fac_table, hide_index=True, use_container_width=True)
        st.pyplot(plot_facets_line(facette_scores))

    if debug:
        st.write("status:", status)

# ------------------------------------------------------------
# TAB 4
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Exports")

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
    for f in FACET_ORDER:
        w.writerow([f, facette_labels[f], facette_scores[f]])

    st.download_button("📥 CSV", out.getvalue(), f"{patient_id}_neo_pir.csv", "text/csv")

    pdf_bytes = build_pdf_report_bytes(patient_id, patient_name, status, facette_scores, domain_scores)
    st.download_button("📥 PDF", pdf_bytes, f"{patient_id}_neo_pir_report.pdf", "application/pdf")

    fig_r = plot_domains_radar(domain_scores)
    st.download_button("📥 Domaines PNG", fig_to_bytes(fig_r, "png"), f"{patient_id}_domains.png", "image/png")

    fig_f = plot_facets_line(facette_scores)
    st.download_button("📥 Facettes PNG", fig_to_bytes(fig_f, "png"), f"{patient_id}_facettes.png", "image/png")
