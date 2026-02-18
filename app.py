import io
import os
import csv
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "NEO PI-R — Calculatrice Pro (Cabinet 2026)"
DB_PATH = "neo_pir.db"
SCORING_KEY_CSV = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # index 0..4
OPT_TO_IDX = {k: i for i, k in enumerate(OPTIONS)}


# ============================================================
# 1) SCORING KEY
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
        raise ValueError(f"scoring_key.csv incomplet. Items manquants: {missing[:30]} ...")
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
# 4) DB (SQLite)
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
        response_idx INTEGER, -- -1 blank, 0..4
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (patient_id, item_id)
    )
    """)
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


def list_patients() -> List[Tuple[str, str]]:
    conn = db()
    rows = conn.execute(
        "SELECT patient_id, COALESCE(name,'') FROM patients ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def load_responses(patient_id: str) -> Dict[int, int]:
    conn = db()
    rows = conn.execute(
        "SELECT item_id, response_idx FROM responses WHERE patient_id=?",
        (patient_id,),
    ).fetchall()
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
        (patient_id, item_id, response_idx),
    )
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
        if not fac:
            continue
        facette_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc

    return facette_scores, domain_scores


# ============================================================
# 6) GRAPHIQUES
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
    c = canvas.Canvas(buf, pagesize=A4)
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
    for d in ["N", "E", "O", "A", "C"]:
        c.drawString(40, y, f"{domain_labels[d]} ({d}): {domain_scores[d]}")
        y -= 12
    y -= 10

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
# 8) UX: Beep + Flash
# ============================================================
def play_beep_once(trigger_id: int):
    # WebAudio beep (léger, discret). Déclenché à chaque validation.
    components.html(
        f"""
        <script>
        (function() {{
          try {{
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const o = ctx.createOscillator();
            const g = ctx.createGain();
            o.type = 'sine';
            o.frequency.value = 880; // beep doux
            g.gain.value = 0.02;     // volume discret
            o.connect(g);
            g.connect(ctx.destination);
            o.start();
            setTimeout(() => {{ o.stop(); ctx.close(); }}, 80);
          }} catch(e) {{}}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


# ============================================================
# 9) APP
# ============================================================
init_db()
st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")

# Load scoring key once
scoring_key = load_scoring_key(SCORING_KEY_CSV)

# Session defaults
if "current_item" not in st.session_state:
    st.session_state.current_item = 1
if "beep_counter" not in st.session_state:
    st.session_state.beep_counter = 0
if "flash" not in st.session_state:
    st.session_state.flash = False


# Sidebar: Mode + Theme + Rules + Patient management
with st.sidebar:
    st.subheader("Patient")
    mode = st.radio("Mode", ["Ouvrir", "Créer"], index=0)

    existing = list_patients()

    patient_id = ""
    patient_name = ""

    if mode == "Ouvrir":
        if existing:
            labels = [f"{pid} — {name}" if name else pid for pid, name in existing]
            pick = st.selectbox("Sélection", labels, index=0)
            patient_id = pick.split(" — ")[0].strip()
            patient_name = dict(existing).get(patient_id, "")
        else:
            st.warning("Aucun patient. Crée un patient.")
    else:
        patient_id = st.text_input("Patient ID (unique)", value="").strip()
        patient_name = st.text_input("Nom / Prénom", value="").strip()
        if st.button("✅ Enregistrer patient", type="primary", disabled=(not patient_id)):
            upsert_patient(patient_id, patient_name)
            st.success("Patient enregistré. Passe en mode Ouvrir.")
            st.rerun()

    st.markdown("---")
    st.subheader("Affichage")
    theme = st.radio("Thème", ["Sombre", "Clair"], index=0, horizontal=True)

    st.markdown("---")
    st.subheader("Règles protocole")
    rules = ProtocolRules(
        max_blank_invalid=st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15),
        max_N_invalid=st.number_input("Réponses N ⇒ invalide si ≥", 0, 240, 42),
        impute_blank_if_leq=st.number_input("Imputation si blancs ≤", 0, 240, 10),
        impute_option_index=2,
    )

    st.markdown("---")
    st.subheader("Gestion patient")
    confirm_delete = st.checkbox("Confirmer suppression (irréversible)", value=False)
    if st.button("🗑️ Supprimer patient", disabled=(not patient_id or not confirm_delete)):
        delete_patient(patient_id)
        st.success("Patient supprimé.")
        st.session_state.current_item = 1
        st.rerun()


# Stop if no patient
if not patient_id:
    st.title(APP_TITLE)
    st.info("Choisis ou crée un patient pour commencer.")
    st.stop()


# Theme CSS + UI CSS (big buttons, header band, fade-in, flash green)
is_dark = (theme == "Sombre")
bg = "#0b0f17" if is_dark else "#f6f7fb"
panel = "#111827" if is_dark else "#ffffff"
text = "#e5e7eb" if is_dark else "#0f172a"
muted = "#9ca3af" if is_dark else "#475569"
border = "#1f2937" if is_dark else "#e5e7eb"

st.markdown(
    f"""
<style>
/* Background */
.stApp {{
  background: {bg};
  color: {text};
}}

/* Main container */
.block-container {{
  padding-top: 1.2rem;
  animation: fadein 0.18s ease-in;
}}

@keyframes fadein {{
  from {{ opacity: 0.0; transform: translateY(6px); }}
  to   {{ opacity: 1.0; transform: translateY(0px); }}
}}

/* Cards / panels */
.pro-card {{
  background: {panel};
  border: 1px solid {border};
  border-radius: 10px;
  padding: 16px 10px;
}}

/* Live band */
.liveband {{
  display:flex;
  gap:14px;
  align-items:center;
  justify-content:space-between;
  background:{panel};
  border:1px solid {border};
  border-radius:14px;
  padding:14px 14px;
  margin-bottom: 14px;
}}
.liveband .kpi {{
  display:flex;
  gap:14px;
  align-items:center;
}}
.k {{
  display:flex; flex-direction:column;
}}
.k .lab {{ font-size: 12px; color: {muted}; }}
.k .val {{ font-size: 14px; font-weight: 800; color: {text}; }}

/* Flash green bar */
.flash-ok {{
  border-left: 10px solid #22c55e;
  animation: flashGlow 0.45s ease-out;
}}
@keyframes flashGlow {{
  0% {{ box-shadow: 0 0 0 rgba(34,197,94,0.0); }}
  45% {{ box-shadow: 0 0 22px rgba(34,197,94,0.28); }}
  100% {{ box-shadow: 0 0 0 rgba(34,197,94,0.0); }}
}}

/* SUPER BIG buttons (responses) */
div.stButton > button {{
  height: 14px !important;
  font-size: 14px !important;
  font-weight: 900 !important;
  border-radius: 14px !important;
  width: 100% !important;
  border: 2px solid {border} !important;
}}

/* Mobile extra big */
@media (max-width: 768px) {{
  div.stButton > button {{
    height: 190px !important;
    font-size: 58px !important;
  }}
}}

/* Navigation buttons */
.navrow div.stButton > button {{
  height: 68px !important;
  font-size: 22px !important;
  font-weight: 800 !important;
  border-radius: 18px !important;
}}

/* Make metrics look clean */
[data-testid="stMetricValue"] {{
  font-weight: 900 !important;
}}

</style>
""",
    unsafe_allow_html=True,
)

# Load responses
responses = load_responses(patient_id)

# Compute stats live
answered = sum(1 for i in range(1, 241) if responses[i] != -1)
progress = answered / 240.0

final_resp, status = apply_protocol_rules(responses, rules)
facette_scores, domain_scores = compute_scores(scoring_key, final_resp)

# Current item clamp
st.session_state.current_item = int(max(1, min(240, st.session_state.current_item)))
item = st.session_state.current_item
current_idx = responses[item]
current_label = "VIDE" if current_idx == -1 else OPTIONS[current_idx]
fac = item_to_facette.get(item, "—")
dom = facettes_to_domain.get(fac, "—")


# Header
st.title(APP_TITLE)
st.caption("Saisie manuelle ultra-rapide • Calcul instantané • SQLite • Exports CSV/PDF • Mode cabinet")

# Liveband
band_class = "liveband flash-ok" if st.session_state.flash else "liveband"
st.markdown(
    f"""
<div class="{band_class}">
  <div style="min-width:240px;">
    <div style="font-size:13px;color:{muted};font-weight:700;">Progression</div>
    <div style="font-size:20px;font-weight:900;">{answered}/240</div>
  </div>
  <div class="kpi">
    <div class="k"><div class="lab">Items vides</div><div class="val">{status["n_blank"]}</div></div>
    <div class="k"><div class="lab">N observés</div><div class="val">{status["n_count"]}</div></div>
    <div class="k"><div class="lab">Imputations</div><div class="val">{status["imputed"]}</div></div>
    <div class="k"><div class="lab">Protocole</div><div class="val">{"VALIDE" if status["valid"] else "INVALIDE"}</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.progress(progress)

if not status["valid"]:
    with st.container():
        st.error("Protocole INVALIDE")
        for r in status["reasons"]:
            st.write("•", r)

# Tabs
tab1, tab2, tab3 = st.tabs(["🧮 Saisie", "📊 Résultats", "📦 Exports"])


# ============================================================
# TAB 1 — SAISIE ULTRA RAPIDE (item -> 5 gros boutons -> suivant)
# ============================================================
with tab1:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)

    topA, topB, topC = st.columns([1, 1, 1.2])
    with topA:
        st.markdown("### Item courant")
        st.write(f"**{item} / 240**")

    with topB:
        jump = st.number_input("Aller à l’item", 1, 240, item, step=1)
        if int(jump) != item:
            st.session_state.current_item = int(jump)
            st.rerun()

    with topC:
        if st.button("➡️ Prochain VIDE", type="secondary"):
            nxt = None
            for i in range(item + 1, 241):
                if responses[i] == -1:
                    nxt = i
                    break
            if nxt is None:
                for i in range(1, item + 1):
                    if responses[i] == -1:
                        nxt = i
                        break
            if nxt is not None:
                st.session_state.current_item = int(nxt)
                st.rerun()
            else:
                st.toast("Aucun item vide ✅", icon="✅")

    st.markdown(f"**Réponse actuelle :** `{current_label}`  •  **Facette :** `{fac}`  •  **Domaine :** `{dom}`")

    # Actions
    act1, act2 = st.columns([1, 2])
    with act1:
        if st.button("🧹 Réinitialiser (VIDE)", type="secondary"):
            save_response(patient_id, item, -1)
            st.session_state.flash = True
            st.session_state.beep_counter += 1
            st.toast("Réponse effacée (VIDE).", icon="🧹")
            st.rerun()

    with act2:
        st.caption("Astuce: clique une réponse → l’app passe automatiquement à l’item suivant.")

    st.markdown("---")
    st.markdown("## Choisir la réponse")

    # ========= 3 + 2 buttons, massive =========
    # LINE 1
    r1c1, r1c2, r1c3 = st.columns(3)

    def commit_answer(idx: int):
        save_response(patient_id, item, idx)
        st.session_state.flash = True
        st.session_state.beep_counter += 1
        st.toast("Enregistré ✅", icon="✅")
        if item < 240:
            st.session_state.current_item = item + 1
        st.rerun()

    if r1c1.button("FD"):
        commit_answer(0)
    if r1c2.button("D"):
        commit_answer(1)
    if r1c3.button("N"):
        commit_answer(2)

    st.markdown("<br>", unsafe_allow_html=True)

    # LINE 2
    r2c1, r2c2 = st.columns(2)
    if r2c1.button("A"):
        commit_answer(3)
    if r2c2.button("FA"):
        commit_answer(4)

    st.markdown("---")

    # Navigation row
    st.markdown('<div class="navrow">', unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    if n1.button("⬅️ Précédent"):
        st.session_state.current_item = max(1, item - 1)
        st.rerun()
    if n2.button("➡️ Suivant"):
        st.session_state.current_item = min(240, item + 1)
        st.rerun()
    if n3.button("⏭️ +10"):
        st.session_state.current_item = min(240, item + 10)
        st.rerun()
    if n4.button("⏮️ -10"):
        st.session_state.current_item = max(1, item - 10)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end card


# Beep trigger (only when counter changes)
if st.session_state.beep_counter > 0:
    play_beep_once(st.session_state.beep_counter)

# Reset flash after rendering once
if st.session_state.flash:
    st.session_state.flash = False


# ============================================================
# TAB 2 — RESULTATS
# ============================================================
with tab2:
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### Domaines")
        dom_table = [{"Domaine": domain_labels[d], "Code": d, "Score brut": domain_scores[d]} for d in ["N","E","O","A","C"]]
        st.dataframe(dom_table, hide_index=True, use_container_width=True)

        fig1 = plot_domains_radar(domain_scores)
        st.pyplot(fig1)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### Facettes")
        fac_rows = [{"Facette": facette_labels[fac], "Code": fac, "Score brut": facette_scores[fac]}
                    for fac in sorted(facette_labels.keys())]
        st.dataframe(fac_rows, hide_index=True, use_container_width=True)

        fig2 = plot_facets_line(facette_scores)
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 3 — EXPORTS
# ============================================================
with tab3:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.subheader("Exports (CSV + PDF + PNG)")

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

    # PNG
    fig_radar = plot_domains_radar(domain_scores)
    st.download_button("📥 Profil Domaines (PNG)", fig_to_bytes(fig_radar, "png"), f"{patient_id}_domains.png", "image/png")

    fig_fac = plot_facets_line(facette_scores)
    st.download_button("📥 Profil Facettes (PNG)", fig_to_bytes(fig_fac, "png"), f"{patient_id}_facettes.png", "image/png")

    st.markdown("</div>", unsafe_allow_html=True)
