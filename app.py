import streamlit as st
import sqlite3
import os
import csv
from typing import Dict

# =========================
# CONFIG
# =========================
APP_TITLE = "NEO PI-R — Calculatrice Pro"
DB_PATH = "neo_pir.db"
SCORING_KEY_FILE = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]  # 0..4

st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")


# =========================
# DB (robuste)
# =========================
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

    # --- Compat: si une ancienne table existe avec colonne "response" au lieu de "response_idx"
    cols = [r[1] for r in conn.execute("PRAGMA table_info(responses)").fetchall()]
    if "response_idx" not in cols and "response" in cols:
        # migration simple: ajouter response_idx puis copier
        conn.execute("ALTER TABLE responses ADD COLUMN response_idx INTEGER;")
        conn.execute("UPDATE responses SET response_idx = response WHERE response_idx IS NULL;")
        conn.commit()

    conn.close()


def get_response_column(conn: sqlite3.Connection) -> str:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(responses)").fetchall()]
    if "response_idx" in cols:
        return "response_idx"
    if "response" in cols:
        return "response"
    # fallback : on force la création au bon format
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


def load_responses(patient_id: str) -> Dict[int, int]:
    conn = db()
    col = get_response_column(conn)

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
    col = get_response_column(conn)

    # si colonne "response" -> écrire dedans ; sinon response_idx
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


# =========================
# SCORING KEY
# =========================
@st.cache_data
def load_scoring_key() -> Dict[int, list]:
    if not os.path.exists(SCORING_KEY_FILE):
        raise FileNotFoundError("scoring_key.csv manquant à la racine du repo.")

    key = {}
    with open(SCORING_KEY_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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
        raise ValueError(f"scoring_key.csv incomplet (items manquants, ex: {missing[:10]}).")
    return key


# =========================
# UI STYLE (boutons massifs + lisibles)
# =========================
st.markdown("""
<style>
.pro-card{
  border-radius:18px; padding:18px;
  border:1px solid rgba(120,120,120,0.25);
  background: rgba(255,255,255,0.03);
}
.answerpad div.stButton>button{
  height: 150px !important;
  font-size: 52px !important;
  font-weight: 900 !important;
  border-radius: 26px !important;
  width:100% !important;
}
.navrow div.stButton>button{
  height: 70px !important;
  font-size: 22px !important;
  font-weight: 850 !important;
  border-radius: 18px !important;
}
@media (max-width:768px){
  .answerpad div.stButton>button{ height: 170px !important; font-size: 56px !important; }
}
</style>
""", unsafe_allow_html=True)


# =========================
# APP
# =========================
ensure_schema()
scoring_key = load_scoring_key()

st.title(APP_TITLE)
st.caption("Correction ultra-rapide : 1 item → 5 boutons → item suivant")

# Sidebar patient
with st.sidebar:
    st.subheader("Patient")

    patient_id = st.text_input("ID patient", value="").strip()
    name = st.text_input("Nom (optionnel)", value="").strip()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Enregistrer", use_container_width=True, disabled=(not patient_id)):
            upsert_patient(patient_id, name)
            st.success("Patient enregistré.")
            st.rerun()

    with c2:
        confirm = st.checkbox("Confirmer suppression", value=False)
        if st.button("🗑️ Supprimer", use_container_width=True, disabled=(not patient_id or not confirm)):
            delete_patient(patient_id)
            st.warning("Patient supprimé.")
            # reset item
            st.session_state.current_item = 1
            st.rerun()

if not patient_id:
    st.info("Entre un ID patient pour commencer.")
    st.stop()

responses = load_responses(patient_id)

if "current_item" not in st.session_state:
    st.session_state.current_item = 1

item = int(max(1, min(240, st.session_state.current_item)))

answered = sum(1 for v in responses.values() if v != -1)
st.progress(answered / 240.0)
st.write(f"Progression : **{answered}/240**")

current_idx = responses[item]
current_label = "VIDE" if current_idx == -1 else OPTIONS[current_idx]

st.markdown('<div class="pro-card">', unsafe_allow_html=True)
st.markdown(f"### Item **{item} / 240** — Réponse actuelle : **{current_label}**")

def go_next():
    st.session_state.current_item = min(240, item + 1)

def commit(idx: int):
    save_response(patient_id, item, idx)
    if item < 240:
        go_next()
    st.rerun()

# reset
if st.button("🧹 Réinitialiser (VIDE)", use_container_width=True):
    save_response(patient_id, item, -1)
    st.rerun()

st.markdown("## Choisir la réponse")

st.markdown('<div class="answerpad">', unsafe_allow_html=True)

r1 = st.columns(3)
if r1[0].button("FD"): commit(0)
if r1[1].button("D"):  commit(1)
if r1[2].button("N"):  commit(2)

st.markdown("<br>", unsafe_allow_html=True)

r2 = st.columns(2)
if r2[0].button("A"):  commit(3)
if r2[1].button("FA"): commit(4)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="navrow">', unsafe_allow_html=True)
n1, n2, n3, n4 = st.columns(4)
if n1.button("⬅️"): st.session_state.current_item = max(1, item-1); st.rerun()
if n2.button("➡️"): st.session_state.current_item = min(240, item+1); st.rerun()
if n3.button("+10"): st.session_state.current_item = min(240, item+10); st.rerun()
if n4.button("-10"): st.session_state.current_item = max(1, item-10); st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Résultats simples (score brut total) - stable
with st.expander("📊 Résultats (score total brut)", expanded=False):
    total = 0
    for it, val in responses.items():
        if val != -1:
            total += scoring_key[it][val]
    st.metric("Score total brut", total)
