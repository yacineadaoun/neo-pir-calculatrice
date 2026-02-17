import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

# Scoring key embedded (from your CSV)
scoring_data = """item,FD,D,N,A,FA
1,4,3,2,1,0
2,0,1,2,3,4
3,0,1,2,3,4
4,4,3,2,1,0
5,0,1,2,3,4
6,0,1,2,3,4
7,4,3,2,1,0
8,4,3,2,1,0
9,0,1,2,3,4
10,4,3,2,1,0
11,4,3,2,1,0
12,0,1,2,3,4
13,0,1,2,3,4
14,4,3,2,1,0
15,0,1,2,3,4
16,0,1,2,3,4
17,4,3,2,1,0
18,4,3,2,1,0
19,0,1,2,3,4
20,4,3,2,1,0
21,4,3,2,1,0
22,0,1,2,3,4
23,0,1,2,3,4
24,4,3,2,1,0
25,0,1,2,3,4
26,0,1,2,3,4
27,4,3,2,1,0
28,4,3,2,1,0
29,0,1,2,3,4
30,4,3,2,1,0
31,0,1,2,3,4
32,4,3,2,1,0
33,4,3,2,1,0
34,0,1,2,3,4
35,4,3,2,1,0
36,4,3,2,1,0
37,0,1,2,3,4
38,0,1,2,3,4
39,4,3,2,1,0
40,0,1,2,3,4
41,0,1,2,3,4
42,4,3,2,1,0
43,4,3,2,1,0
44,0,1,2,3,4
45,4,3,2,1,0
46,4,3,2,1,0
47,0,1,2,3,4
48,0,1,2,3,4
49,4,3,2,1,0
50,0,1,2,3,4
51,0,1,2,3,4
52,4,3,2,1,0
53,4,3,2,1,0
54,0,1,2,3,4
55,4,3,2,1,0
56,4,3,2,1,0
57,0,1,2,3,4
58,0,1,2,3,4
59,4,3,2,1,0
60,0,1,2,3,4
61,4,3,2,1,0
62,0,1,2,3,4
63,0,1,2,3,4
64,4,3,2,1,0
65,0,1,2,3,4
66,0,1,2,3,4
67,4,3,2,1,0
68,4,3,2,1,0
69,0,1,2,3,4
70,4,3,2,1,0
71,4,3,2,1,0
72,0,1,2,3,4
73,0,1,2,3,4
74,4,3,2,1,0
75,0,1,2,3,4
76,0,1,2,3,4
77,4,3,2,1,0
78,4,3,2,1,0
79,0,1,2,3,4
80,4,3,2,1,0
81,4,3,2,1,0
82,0,1,2,3,4
83,0,1,2,3,4
84,4,3,2,1,0
85,0,1,2,3,4
86,0,1,2,3,4
87,4,3,2,1,0
88,4,3,2,1,0
89,0,1,2,3,4
90,4,3,2,1,0
91,0,1,2,3,4
92,4,3,2,1,0
93,4,3,2,1,0
94,0,1,2,3,4
95,4,3,2,1,0
96,4,3,2,1,0
97,0,1,2,3,4
98,0,1,2,3,4
99,4,3,2,1,0
100,0,1,2,3,4
101,0,1,2,3,4
102,4,3,2,1,0
103,4,3,2,1,0
104,0,1,2,3,4
105,4,3,2,1,0
106,4,3,2,1,0
107,0,1,2,3,4
108,0,1,2,3,4
109,4,3,2,1,0
110,0,1,2,3,4
111,0,1,2,3,4
112,4,3,2,1,0
113,4,3,2,1,0
114,0,1,2,3,4
115,4,3,2,1,0
116,4,3,2,1,0
117,0,1,2,3,4
118,0,1,2,3,4
119,4,3,2,1,0
120,0,1,2,3,4
121,4,3,2,1,0
122,0,1,2,3,4
123,0,1,2,3,4
124,4,3,2,1,0
125,0,1,2,3,4
126,0,1,2,3,4
127,4,3,2,1,0
128,4,3,2,1,0
129,0,1,2,3,4
130,4,3,2,1,0
131,4,3,2,1,0
132,0,1,2,3,4
133,0,1,2,3,4
134,4,3,2,1,0
135,0,1,2,3,4
136,0,1,2,3,4
137,4,3,2,1,0
138,4,3,2,1,0
139,0,1,2,3,4
140,4,3,2,1,0
141,4,3,2,1,0
142,0,1,2,3,4
143,0,1,2,3,4
144,4,3,2,1,0
145,0,1,2,3,4
146,0,1,2,3,4
147,4,3,2,1,0
148,4,3,2,1,0
149,0,1,2,3,4
150,4,3,2,1,0
151,0,1,2,3,4
152,4,3,2,1,0
153,4,3,2,1,0
154,0,1,2,3,4
155,4,3,2,1,0
156,4,3,2,1,0
157,0,1,2,3,4
158,0,1,2,3,4
159,4,3,2,1,0
160,0,1,2,3,4
161,0,1,2,3,4
162,4,3,2,1,0
163,4,3,2,1,0
164,0,1,2,3,4
165,4,3,2,1,0
166,4,3,2,1,0
167,0,1,2,3,4
168,0,1,2,3,4
169,4,3,2,1,0
170,0,1,2,3,4
171,0,1,2,3,4
172,4,3,2,1,0
173,4,3,2,1,0
174,0,1,2,3,4
175,4,3,2,1,0
176,4,3,2,1,0
177,0,1,2,3,4
178,0,1,2,3,4
179,4,3,2,1,0
180,0,1,2,3,4
181,4,3,2,1,0
182,0,1,2,3,4
183,0,1,2,3,4
184,4,3,2,1,0
185,0,1,2,3,4
186,0,1,2,3,4
187,4,3,2,1,0
188,4,3,2,1,0
189,0,1,2,3,4
190,4,3,2,1,0
191,4,3,2,1,0
192,0,1,2,3,4
193,0,1,2,3,4
194,4,3,2,1,0
195,0,1,2,3,4
196,0,1,2,3,4
197,4,3,2,1,0
198,4,3,2,1,0
199,0,1,2,3,4
200,4,3,2,1,0
201,4,3,2,1,0
202,0,1,2,3,4
203,0,1,2,3,4
204,4,3,2,1,0
205,0,1,2,3,4
206,0,1,2,3,4
207,4,3,2,1,0
208,4,3,2,1,0
209,0,1,2,3,4
210,4,3,2,1,0
211,0,1,2,3,4
212,4,3,2,1,0
213,4,3,2,1,0
214,0,1,2,3,4
215,4,3,2,1,0
216,4,3,2,1,0
217,0,1,2,3,4
218,0,1,2,3,4
219,4,3,2,1,0
220,0,1,2,3,4
221,0,1,2,3,4
222,4,3,2,1,0
223,4,3,2,1,0
224,0,1,2,3,4
225,4,3,2,1,0
226,4,3,2,1,0
227,0,1,2,3,4
228,0,1,2,3,4
229,4,3,2,1,0
230,0,1,2,3,4
231,0,1,2,3,4
232,4,3,2,1,0
233,4,3,2,1,0
234,0,1,2,3,4
235,4,3,2,1,0
236,4,3,2,1,0
237,0,1,2,3,4
238,0,1,2,3,4
239,4,3,2,1,0
240,0,1,2,3,4
"""

@st.cache_resource
def load_scoring_key():
    reader = pd.read_csv(io.StringIO(scoring_data))
    key = {}
    for row in reader.itertuples():
        item = row.item
        key[item] = [row.FD, row.D, row.N, row.A, row.FA]
    return key

scoring_key = load_scoring_key()

# Item to facette (from your code)
facet_bases = {
    "N1": [1],  "N2": [6],  "N3": [11], "N4": [16], "N5": [21], "N6": [26],
    "E1": [2],  "E2": [7],  "E3": [12], "E4": [17], "E5": [22], "E6": [27],
    "O1": [3],  "O2": [8],  "O3": [13], "O4": [18], "O5": [23], "O6": [28],
    "A1": [4],  "A2": [9],  "A3": [14], "A4": [19], "A5": [24], "A6": [29],
    "C1": [5],  "C2": [10], "C3": [15], "C4": [20], "C5": [25], "C6": [30],
}

item_to_facette = {}
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

# Protocol
class ProtocolRules:
    def __init__(self, max_blank_invalid=15, max_N_invalid=42, impute_blank_if_leq=10, impute_option_index=2):
        self.max_blank_invalid = max_blank_invalid
        self.max_N_invalid = max_N_invalid
        self.impute_blank_if_leq = impute_blank_if_leq
        self.impute_option_index = impute_option_index

def compute_scores(responses):
    facette_scores = {fac: 0 for fac in facette_labels}
    for item_id, idx in responses.items():
        if idx == -1: continue
        fac = item_to_facette.get(item_id)
        if fac is None: continue
        facette_scores[fac] += scoring_key[item_id][idx]
    domain_scores = {d: 0 for d in domain_labels}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc
    return facette_scores, domain_scores

def apply_protocol_rules(responses, rules):
    blanks = [i for i, v in responses.items() if v == -1]
    n_blank = len(blanks)
    n_count = sum(1 for v in responses.values() if v == rules.impute_option_index)
    status = {
        "valid": True,
        "reasons": [],
        "n_blank": n_blank,
        "n_count": n_count,
        "imputed": 0
    }
    if n_blank >= rules.max_blank_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop d'items vides : {n_blank} (>= {rules.max_blank_invalid})")
    if n_count >= rules.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop de réponses 'N' : {n_count} (>= {rules.max_N_invalid})")
    new_resp = dict(responses)
    if status["valid"] and 0 < n_blank <= rules.impute_blank_if_leq:
        for item_id in blanks:
            new_resp[item_id] = rules.impute_option_index
            status["imputed"] += 1
    return new_resp, status

def plot_profile(facette_scores, domain_scores):
    x_labels = ["N", "E", "O", "A", "C"] + [f for f in facette_labels.keys()]
    y = [domain_scores[d] for d in ["N", "E", "O", "A", "C"]] + [facette_scores[f] for f in facette_labels]
    fig = plt.figure(figsize=(16, 5))
    ax = plt.gca()
    ax.plot(range(len(x_labels)), y, marker="o", linewidth=2)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=60, ha="right")
    ax.set_title("Profil NEO PI-R — Scores bruts (Domaines + 30 facettes)")
    ax.set_ylabel("Score brut")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig

def fig_to_bytes(fig, fmt):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# UI
st.set_page_config(page_title="Calculatrice NEO PI-R Manuelle", layout="wide")

st.title("Calculatrice NEO PI-R — Saisie Manuelle des Réponses")
st.caption("Saisissez les réponses (FD/D/N/A/FA) pour obtenir les scores. Idéal pour recopier 200 copies rapidement.")

with st.sidebar:
    st.subheader("Protocole")
    max_blank_invalid = st.number_input("Items vides ⇒ invalide si ≥", 0, 240, 15)
    max_N_invalid = st.number_input("Réponses 'N' ⇒ invalide si ≥", 0, 240, 42)
    impute_blank_if_leq = st.number_input("Imputation si blancs ≤", 0, 240, 10)

RULES = ProtocolRules(max_blank_invalid, max_N_invalid, impute_blank_if_leq, 2)  # N = index 2

# CSS for big round buttons
st.markdown("""
    <style>
    div.row-widget.stRadio > div {
        flex-direction: row;
        justify-content: center;
    }
    div.row-widget.stRadio > div > label > div {
        width: 60px !important;
        height: 60px !important;
        border-radius: 50% !important;
        background-color: #f0f0f0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 5px !important;
        font-size: 24px !important;
        cursor: pointer !important;
    }
    div.row-widget.stRadio > div > label > input:checked + div {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

options = ['FD', 'D', 'N', 'A', 'FA']

responses = {}
with st.expander("Items 1-60"):
    for i in range(1, 61):
        selected = st.radio(f"Item {i}", options, horizontal=True, index=2, key=f"item{i}")
        responses[i] = options.index(selected)

with st.expander("Items 61-120"):
    for i in range(61, 121):
        selected = st.radio(f"Item {i}", options, horizontal=True, index=2, key=f"item{i}")
        responses[i] = options.index(selected)

with st.expander("Items 121-180"):
    for i in range(121, 181):
        selected = st.radio(f"Item {i}", options, horizontal=True, index=2, key=f"item{i}")
        responses[i] = options.index(selected)

with st.expander("Items 181-240"):
    for i in range(181, 241):
        selected = st.radio(f"Item {i}", options, horizontal=True, index=2, key=f"item{i}")
        responses[i] = options.index(selected)

if st.button("Calculer les scores", type="primary"):
    # Replace missing with -1 (if not all answered)
    for i in range(1, 241):
        if i not in responses:
            responses[i] = -1

    final_responses, status = apply_protocol_rules(responses, RULES)
    facette_scores, domain_scores = compute_scores(final_responses)

    st.subheader("Statut Protocole")
    cols = st.columns(4)
    cols[0].metric("Vides", status["n_blank"])
    cols[1].metric("N", status["n_count"])
    cols[2].metric("Imputés", status["imputed"])
    cols[3].metric("Valide", "Oui" if status["valid"] else "Non")

    if not status["valid"]:
        st.error("\n".join(status["reasons"]))

    tab1, tab2, tab3, tab4 = st.tabs(["Facettes", "Domaines", "Profil", "Exports"])

    with tab1:
        data = [{"Facette": facette_labels[fac], "Score": facette_scores[fac]} for fac in sorted(facette_labels)]
        st.dataframe(data, use_container_width=True, hide_index=True)

    with tab2:
        dom_data = [{"Domaine": domain_labels[d], "Score": domain_scores[d]} for d in ["N", "E", "O", "A", "C"]]
        st.dataframe(dom_data, use_container_width=True, hide_index=True)

    with tab3:
        fig = plot_profile(facette_scores, domain_scores)
        st.pyplot(fig)

    with tab4:
        df_fac = pd.DataFrame(data)
        df_dom = pd.DataFrame(dom_data)
        csv = pd.concat([df_fac, df_dom]).to_csv(index=False)
        st.download_button("Télécharger CSV", csv, "neo_scores.csv")

        st.download_button("Télécharger Profil PNG", fig_to_bytes(fig, "png"), "neo_profil.png")

if st.button("Réinitialiser pour nouvelle copy"):
    st.experimental_rerun()
