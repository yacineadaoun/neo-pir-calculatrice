# NEO PI-R — Calculatrice Manuelle (Streamlit)

Application Streamlit pour saisir manuellement les 240 réponses (FD/D/N/A/FA),
appliquer les règles du protocole, calculer les scores (30 facettes + 5 domaines),
afficher un profil graphique, et exporter CSV/PDF.

## Fichiers requis
- `app.py`
- `scoring_key.csv` (obligatoire, 240 items, colonnes: item, FD, D, N, A, FA)
- `requirements.txt`

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
