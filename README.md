# Suite Streamlit (multi-pages)

## Lancer
Dans ce dossier :

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Pages
- YouTrack Dashboard (CSV)
- Tendances réclamations (Excel)
- Référencement Pannes (Excel)

## Important (Référencement Pannes)
- `table_append.py` est fourni avec une implémentation openpyxl.
  Si tu as déjà ton propre `table_append.py`, remplace-le (même nom de fichier / même fonction).
