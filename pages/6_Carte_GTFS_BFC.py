import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Carte GTFS — BFC (par département)")

# Chemin robuste : la carte est à la racine du projet (au même niveau que home.py)
base_dir = Path(__file__).resolve().parents[1]
html_path = base_dir / "gtfs_routes_BFC_admin_by_dept.html"

if not html_path.exists():
    st.error(f"Fichier introuvable : {html_path}")
    st.info("Place le .html à la racine du projet, ou adapte html_path.")
else:
    components.html(html_path.read_text(encoding="utf-8"), height=900, scrolling=True)
