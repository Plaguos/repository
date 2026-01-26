import streamlit as st

st.set_page_config(page_title="Suite Streamlit", layout="wide")

st.title("Suite Streamlit — Outils")
st.markdown("""
Choisis une page dans la barre latérale :

- **YouTrack Dashboard (CSV)** : analyse d’un export CSV YouTrack.
- **Tendances réclamations (Excel)** : multi-fichiers, anomalies, STL, comparaisons.
- **Référencement Pannes (Excel)** : saisie + append dans une table Excel.
- **Comparateur GTFS** : visualiser les différences entre deux GTFS.

👉 Ouvre le menu en haut à gauche (ou la barre latérale) pour naviguer.
""")

st.info("Astuce : si une page dépend d’un fichier local, charge-le depuis la page correspondante.")
