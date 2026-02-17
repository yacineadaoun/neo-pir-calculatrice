import io
import os
import csv
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# COMPAT RERUN (NEW/OLD STREAMLIT)
# ============================================================
def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "NEO PI-R — Calculatrice Pro (ADAOUN YACINE)"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # index 0..4
IDX_TO_OPT = {i: k for i, k in enumerate(OPTIONS)}

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
            f"❌ '{path}' introuvable.\n➡️ Mets 'scoring_key.csv' à la racine du repo."
        )
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"item", "FD", "D", "N", "A", "FA"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("❌ scoring_key.csv doit contenir: item, FD, D, N, A, FA")

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
# 4) DB (robuste Streamlit Cloud)
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
    CREATE TABLE IF NOT EXISTS patients(
        patient_id TEXT PRIMARY KEY,
        name TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS responses(
        patient_id TEXT,
        item_id INTEGER,
        response_idx INTEGER,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(patient_id, item_id)
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS settings(
        patient_id TEXT PRIMARY KEY,
        current_item INTEGER DEFAULT 1,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()


def upsert_patient(patient_id: str, name: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO patients(patient_id,name) VALUES(?,?) "
        "ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name",
        (patient_id, name)
    )
    conn.commit()


def delete_patient(patient_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
    conn.execute("DELETE FROM settings WHERE patient_id=?", (patient_id,))
    conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
    conn.commit()


def list_patients() -> List[Tuple[str, str]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT patient_id, COALESCE(name,'') AS name FROM patients ORDER BY created_at DESC"
    ).fetchall()
    return [(r["patient_id"], r["name"]) for r in rows]


def load_responses(patient_id: str) -> Dict[int, int]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT item_id, response_idx FROM responses WHERE patient_id=?",
        (patient_id,)
    ).fetchall()
    resp = {int(r["item_id"]): int(r["response_idx"]) for r in rows}
    for i in range(1, 241):
        resp.setdefault(i, -1)
    return resp


def save_response(patient_id: str, item_id: int, response_idx: int):
    conn = get_conn()
    conn.execute(
        "INSERT INTO responses(patient_id,item_id,response_idx) VALUES(?,?,?) "
        "ON CONFLICT(patient_id,item_id) DO UPDATE SET response_idx=excluded.response_idx, updated_at=CURRENT_TIMESTAMP",
        (patient_id, item_id, response_idx)
    )
    conn.commit()


def load_current_item(patient_id: str) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT current_item FROM settings WHERE patient_id=?",
        (patient_id,)
    ).fetchone()
    return int(row["current_item"]) if row else 1


def save_current_item(patient_id: str, current_item: int):
    conn = get_conn()
    conn.execute(
        "INSERT INTO settings(patient_id,current_item) VALUES(?,?) "
        "ON CONFLICT(patient_id) DO UPDATE SET current_item=excluded.current_item, updated_at=CURRENT_TIMESTAMP",
        (patient_id, current_item)
    )
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
        if fac:
            facette_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc
    return facette_scores, domain_scores


# ============================================================
# 6) GRAPHIQUES
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
# 7) PDF
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
st.caption("Saisie manuelle assistée • Calcul instantané • Sauvegarde • Exports")

st.markdown("""
<style>
/* Tous les boutons Streamlit (desktop + mobile) */
div.stButton > button, div.stDownloadButton > button {
    height: 92px !important;
    font-size: 30px !important;
    font-weight: 900 !important;
    border-radius: 24px !important;
    width: 100% !important;
}

/* Boutons “navigation” un peu moins hauts */
button:has(span:contains("Précédent")),
button:has(span:contains("Suivant")),
button:has(span:contains("+10")),
button:has(span:contains("-10")) {
    height: 60px !important;
    font-size: 20px !important;
    border-radius: 18px !important;
}

/* Sur petits écrans (téléphone) : encore plus grand */
@media (max-width: 600px) {
  div.stButton > button, div.stDownloadButton > button {
      height: 105px !important;
      font-size: 34px !important;
  }
}
</style>
""", unsafe_allow_html=True)

init_db()

try:
    scoring_key = load_scoring_key_csv(SCORING_KEY_CSV)
except Exception as e:
    st.error(str(e))
    st.stop()

# state for delete confirmation
if "confirm_delete_open" not in st.session_state:
    st.session_state.confirm_delete_open = False

with st.sidebar:
    st.subheader("👤 Patient")
    mode = st.radio("Mode", ["Ouvrir", "Créer"], index=0)

    existing = list_patients()

    if mode == "Ouvrir":
        if not existing:
            st.warning("Aucun patient. Crée un patient.")
            st.stop()

        labels = [f"{pid} — {name}" if name else pid for pid, name in existing]
        pick = st.selectbox("Sélection", labels, index=0)
        patient_id = pick.split(" — ")[0].strip()
        patient_name = dict(existing).get(patient_id, "")
    else:
        patient_id = st.text_input("Patient ID (unique)", value="")
        patient_name = st.text_input("Nom / Prénom", value="")
        if st.button("✅ Créer", type="primary", disabled=(not patient_id.strip())):
            upsert_patient(patient_id.strip(), patient_name.strip())
            st.success("Patient créé ✅ (repasse en mode Ouvrir)")
            st.stop()

    st.markdown("---")
    st.subheader("🧾 Protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses N ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2
    )

    # Delete patient (secure)
    st.markdown("---")
    st.subheader("⚠️ Gestion patient")

    if mode == "Ouvrir":
        if st.button("🗑 Supprimer patient", use_container_width=True):
            st.session_state.confirm_delete_open = True

        if st.session_state.confirm_delete_open:
            st.warning("Suppression définitive. Cette action est irréversible.")
            confirm = st.text_input("Tape le Patient ID pour confirmer", value="", placeholder=patient_id)
            colx, coly = st.columns(2)
            with colx:
                if st.button("Annuler", use_container_width=True):
                    st.session_state.confirm_delete_open = False
                    rerun()
            with coly:
                if st.button("Confirmer suppression", use_container_width=True, type="primary", disabled=(confirm.strip() != patient_id)):
                    delete_patient(patient_id)
                    st.session_state.confirm_delete_open = False
                    st.success("Patient supprimé ✅")
                    rerun()

# data
responses = load_responses(patient_id)
saved_item = load_current_item(patient_id)

if "current_item" not in st.session_state:
    st.session_state.current_item = int(saved_item)

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

tabs = st.tabs(["🧮 Saisie", "📊 Résultats", "📦 Exports"])

# ============================================================
# TAB 1 — SAISIE
# ============================================================
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    topL, topM, topR = st.columns([1, 1, 1.2])

    with topL:
        item = st.number_input("Item", 1, 240, int(st.session_state.current_item), step=1)
        item = int(item)
        st.session_state.current_item = item
        save_current_item(patient_id, item)

    with topM:
        go = st.text_input("Aller à item", value="", placeholder="ex: 120")
        if go.strip().isdigit():
            gi = int(go.strip())
            if 1 <= gi <= 240 and gi != st.session_state.current_item:
                st.session_state.current_item = gi
                save_current_item(patient_id, gi)
                rerun()

    with topR:
        if st.button("➡️ Prochain VIDE", use_container_width=True):
            blanks = [i for i, v in responses.items() if v == -1]
            if blanks:
                st.session_state.current_item = blanks[0]
                save_current_item(patient_id, st.session_state.current_item)
                rerun()
            else:
                st.success("Aucun item vide ✅")

    current_idx = responses[item]
    current_label = "VIDE" if current_idx == -1 else IDX_TO_OPT[current_idx]
    st.markdown(f"Réponse actuelle: <span class='pill'>{current_label}</span>", unsafe_allow_html=True)
    st.write(f"Facette: **{item_to_facette.get(item, '?')}**")

    # Reset answer
    rc1, rc2 = st.columns([1.2, 3])
    with rc1:
        if st.button("🧹 Réinitialiser réponse", use_container_width=True):
            save_response(patient_id, item, -1)
            save_current_item(patient_id, item)
            rerun()
    with rc2:
        st.caption("Efface la réponse de l’item courant (met VIDE).")

    st.markdown("---")

    # 1 clic = save + next (gros boutons)
    with st.form("answer_form", clear_on_submit=False):
        clicked = None
        r1 = st.columns(3)
        r2 = st.columns(3)

        with r1[0]:
            st.markdown("<div class='big'>", unsafe_allow_html=True)
            if st.form_submit_button("FD", use_container_width=True):
                clicked = 0
            st.markdown("</div>", unsafe_allow_html=True)
        with r1[1]:
            st.markdown("<div class='big'>", unsafe_allow_html=True)
            if st.form_submit_button("D", use_container_width=True):
                clicked = 1
            st.markdown("</div>", unsafe_allow_html=True)
        with r1[2]:
            st.markdown("<div class='big'>", unsafe_allow_html=True)
            if st.form_submit_button("N", use_container_width=True):
                clicked = 2
            st.markdown("</div>", unsafe_allow_html=True)

        with r2[0]:
            st.markdown("<div class='big'>", unsafe_allow_html=True)
            if st.form_submit_button("A", use_container_width=True):
                clicked = 3
            st.markdown("</div>", unsafe_allow_html=True)
        with r2[1]:
            st.markdown("<div class='big'>", unsafe_allow_html=True)
            if st.form_submit_button("FA", use_container_width=True):
                clicked = 4
            st.markdown("</div>", unsafe_allow_html=True)
        with r2[2]:
            st.markdown("<div class='big'>", unsafe_allow_html=True)
            if st.form_submit_button("VIDE", use_container_width=True):
                clicked = -1
            st.markdown("</div>", unsafe_allow_html=True)

        if clicked is not None:
            save_response(patient_id, item, int(clicked))
            nxt = min(240, item + 1)
            st.session_state.current_item = nxt
            save_current_item(patient_id, nxt)
            rerun()

    nav = st.columns(4)
    for i, (lab, delta) in enumerate([("⬅️ Précédent", -1), ("➡️ Suivant", +1), ("⏭️ +10", +10), ("⏮️ -10", -10)]):
        with nav[i]:
            st.markdown("<div class='nav'>", unsafe_allow_html=True)
            if st.button(lab, use_container_width=True, key=f"nav_{lab}"):
                new_it = int(min(240, max(1, item + delta)))
                st.session_state.current_item = new_it
                save_current_item(patient_id, new_it)
                rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — RESULTATS
# ============================================================
with tabs[1]:
    st.subheader("Résultats (temps réel)")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("### Domaines")
        st.dataframe(
            [{"Code": d, "Domaine": domain_labels[d], "Score brut": domain_scores[d]} for d in DOMAIN_ORDER],
            hide_index=True,
            use_container_width=True
        )
        st.pyplot(plot_domains_radar(domain_scores), clear_figure=True)

    with c2:
        st.markdown("### Facettes")
        st.dataframe(
            [{"Code": f, "Facette": facette_labels[f], "Score brut": facette_scores[f]} for f in FACET_ORDER],
            hide_index=True,
            use_container_width=True
        )
        st.pyplot(plot_facets_line(facette_scores), clear_figure=True)

# ============================================================
# TAB 3 — EXPORTS
# ============================================================
with tabs[2]:
    st.subheader("Exports (CSV + PDF + PNG)")

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

    st.download_button("📥 Télécharger CSV", out.getvalue(), f"{patient_id}_neo_pir.csv", "text/csv")

    pdf_bytes = build_pdf_report_bytes(patient_id, patient_name, status, facette_scores, domain_scores)
    st.download_button("📥 Télécharger PDF", pdf_bytes, f"{patient_id}_neo_pir_report.pdf", "application/pdf")

    fig_r = plot_domains_radar(domain_scores)
    st.download_button("📥 Domaines PNG", fig_to_bytes(fig_r, "png"), f"{patient_id}_domains.png", "image/png")

    fig_f = plot_facets_line(facette_scores)
    st.download_button("📥 Facettes PNG", fig_to_bytes(fig_f, "png"), f"{patient_id}_facettes.png", "image/png")
