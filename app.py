import streamlit as st
import sqlite3
import os
import csv
from typing import Dict, List
from dataclasses import dataclass

# ======================================================
# CONFIG
# ======================================================

st.set_page_config(page_title="NEO PI-R Calculator Pro", layout="wide")

DB_PATH = "neo_pir.db"
SCORING_KEY_FILE = "scoring_key.csv"

OPTIONS = ["FD", "D", "N", "A", "FA"]

# ======================================================
# DATABASE
# ======================================================

def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        patient_id TEXT PRIMARY KEY,
        name TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS responses(
        patient_id TEXT,
        item_id INTEGER,
        response INTEGER,
        PRIMARY KEY(patient_id,item_id)
    )
    """)
    conn.commit()
    conn.close()

def save_response(patient_id, item_id, value):
    conn = db()
    conn.execute("""
    INSERT INTO responses(patient_id,item_id,response)
    VALUES(?,?,?)
    ON CONFLICT(patient_id,item_id)
    DO UPDATE SET response=excluded.response
    """, (patient_id,item_id,value))
    conn.commit()
    conn.close()

def load_responses(patient_id):
    conn = db()
    rows = conn.execute(
        "SELECT item_id,response FROM responses WHERE patient_id=?",
        (patient_id,)
    ).fetchall()
    conn.close()
    resp = {i:-1 for i in range(1,241)}
    for r in rows:
        resp[r[0]] = r[1]
    return resp

def delete_patient(patient_id):
    conn = db()
    conn.execute("DELETE FROM responses WHERE patient_id=?", (patient_id,))
    conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
    conn.commit()
    conn.close()

# ======================================================
# SCORING
# ======================================================

@st.cache_data
def load_scoring_key():
    if not os.path.exists(SCORING_KEY_FILE):
        st.error("scoring_key.csv manquant.")
        st.stop()
    key = {}
    with open(SCORING_KEY_FILE,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key[int(row["item"])] = [
                int(row["FD"]),
                int(row["D"]),
                int(row["N"]),
                int(row["A"]),
                int(row["FA"])
            ]
    return key

scoring_key = load_scoring_key()

# ======================================================
# UI STYLE
# ======================================================

st.markdown("""
<style>
div.stButton > button {
    height: 140px !important;
    font-size: 42px !important;
    font-weight: 900 !important;
    border-radius: 22px !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# APP
# ======================================================

init_db()

st.title("NEO PI-R — Calculatrice Professionnelle")

# SIDEBAR
with st.sidebar:
    st.subheader("Patient")

    patient_id = st.text_input("ID patient")
    name = st.text_input("Nom")

    if st.button("Créer / Mettre à jour patient"):
        conn = db()
        conn.execute("""
        INSERT INTO patients(patient_id,name)
        VALUES(?,?)
        ON CONFLICT(patient_id)
        DO UPDATE SET name=excluded.name
        """,(patient_id,name))
        conn.commit()
        conn.close()
        st.success("Patient enregistré.")

    if st.button("Supprimer patient"):
        delete_patient(patient_id)
        st.warning("Patient supprimé.")
        st.rerun()

if not patient_id:
    st.stop()

responses = load_responses(patient_id)

if "current_item" not in st.session_state:
    st.session_state.current_item = 1

item = st.session_state.current_item

st.progress(sum(1 for v in responses.values() if v!=-1)/240)
st.write(f"Item **{item}** / 240")

current_value = responses[item]
current_label = "VIDE" if current_value==-1 else OPTIONS[current_value]
st.write(f"Réponse actuelle : **{current_label}**")

# ======================================================
# BUTTONS 3 + 2 MASSIFS
# ======================================================

def answer(idx):
    save_response(patient_id,item,idx)
    if item<240:
        st.session_state.current_item +=1
    st.rerun()

row1 = st.columns(3)
if row1[0].button("FD"): answer(0)
if row1[1].button("D"): answer(1)
if row1[2].button("N"): answer(2)

st.markdown("<br>", unsafe_allow_html=True)

row2 = st.columns(2)
if row2[0].button("A"): answer(3)
if row2[1].button("FA"): answer(4)

# RESET
if st.button("Réinitialiser"):
    save_response(patient_id,item,-1)
    st.rerun()

# NAVIGATION
nav = st.columns(4)
if nav[0].button("⬅️"):
    st.session_state.current_item = max(1,item-1)
    st.rerun()
if nav[1].button("➡️"):
    st.session_state.current_item = min(240,item+1)
    st.rerun()
if nav[2].button("+10"):
    st.session_state.current_item = min(240,item+10)
    st.rerun()
if nav[3].button("-10"):
    st.session_state.current_item = max(1,item-10)
    st.rerun()

# ======================================================
# RESULTATS
# ======================================================

if st.checkbox("Afficher résultats"):

    fac_scores = {f:0 for f in set(scoring_key.keys())}
    total = 0

    for it,val in responses.items():
        if val!=-1:
            total += scoring_key[it][val]

    st.write("Score total brut :", total)
