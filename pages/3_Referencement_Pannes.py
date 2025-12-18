import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta

# --------------------
# Config / constantes
# --------------------
FEUILLE_DEFAULT = "Pannes_matériel"

st.title("📊 Pannes — Graphiques (lecture seule)")
st.caption("Version sans SharePoint et sans écriture : charge un fichier Excel et explore les graphiques.")

# --------------------
# Upload + lecture
# --------------------
uploaded = st.file_uploader("Charger un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Charge un fichier Excel pour démarrer.")
    st.stop()

# Choix feuille
try:
    xls = pd.ExcelFile(uploaded)
    sheets = xls.sheet_names
except Exception as e:
    st.error(f"Impossible de lire le fichier Excel : {e}")
    st.stop()

sheet = st.selectbox(
    "Feuille à lire",
    options=sheets,
    index=(sheets.index(FEUILLE_DEFAULT) if FEUILLE_DEFAULT in sheets else 0),
)

try:
    df = pd.read_excel(uploaded, sheet_name=sheet)
except Exception as e:
    st.error(f"Impossible de lire la feuille '{sheet}' : {e}")
    st.stop()

if df.empty:
    st.warning("La feuille est vide.")
    st.stop()

st.subheader("Aperçu des données")
st.dataframe(df, use_container_width=True, height=280)

# --------------------
# Helpers dates
# --------------------
def parse_date_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(s)), errors="coerce")

# Construire DateRef (tolérant)
df_stats = df.copy()

dateref_candidates = [
    "Date signalement",
    "Date signalement/envoi",
    "Date de soumission",
    "Date",
]
date_col_found = next((c for c in dateref_candidates if c in df_stats.columns), None)

if date_col_found:
    df_stats["DateRef"] = parse_date_series(df_stats[date_col_found])
else:
    df_stats["DateRef"] = pd.NaT

# --------------------
# Filtres
# --------------------
st.markdown("---")
st.subheader("Filtres")

min_d = df_stats["DateRef"].min()
max_d = df_stats["DateRef"].max()

if pd.isna(min_d) or pd.isna(max_d):
    # fallback si pas de dates exploitables
    min_d = datetime.today() - timedelta(days=90)
    max_d = datetime.today()

cfa, cfb, cfc, cfd, cfe = st.columns(5)
with cfa:
    d1 = st.date_input("Du", value=min_d.date() if not pd.isna(min_d) else date.today())
with cfb:
    d2 = st.date_input("Au", value=max_d.date() if not pd.isna(max_d) else date.today())
with cfc:
    f_type_eq = st.multiselect(
        "Type équipement",
        sorted([x for x in df_stats["Type équipement"].dropna().astype(str).unique()])
    ) if "Type équipement" in df_stats.columns else []
with cfd:
    f_type_err = st.multiselect(
        "Type erreur",
        sorted([x for x in df_stats["Type erreur"].dropna().astype(str).unique()])
    ) if "Type erreur" in df_stats.columns else []
with cfe:
    f_sav = st.multiselect(
        "SAV",
        sorted([x for x in df_stats["SAV"].dropna().astype(str).unique()])
    ) if "SAV" in df_stats.columns else []

mask = (df_stats["DateRef"].dt.date >= d1) & (df_stats["DateRef"].dt.date <= d2)
if f_type_eq and "Type équipement" in df_stats.columns:
    mask &= df_stats["Type équipement"].astype(str).isin(f_type_eq)
if f_type_err and "Type erreur" in df_stats.columns:
    mask &= df_stats["Type erreur"].astype(str).isin(f_type_err)
if f_sav and "SAV" in df_stats.columns:
    mask &= df_stats["SAV"].astype(str).isin(f_sav)

df_f = df_stats[mask].copy()

if df_f.empty:
    st.warning("Aucune ligne après filtres.")
    st.stop()

# Plotly si dispo
try:
    import plotly.express as px
    use_plotly = True
except Exception:
    use_plotly = False

# --------------------
# Graphiques
# --------------------
st.markdown("---")

# 1) Pannes par type d’erreur
st.markdown("### Pannes par type d’erreur")
if "Type erreur" in df_f.columns:
    by_type = (
        df_f.groupby("Type erreur").size()
        .reset_index(name="Nb")
        .sort_values("Nb", ascending=False)
    )
else:
    by_type = pd.DataFrame(columns=["Type erreur", "Nb"])

if use_plotly and not by_type.empty:
    fig1 = px.bar(by_type, x="Type erreur", y="Nb", color="Type erreur", height=420)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.dataframe(by_type, use_container_width=True)

# 2) Évolution mensuelle
st.markdown("### Évolution mensuelle")
df_f["Mois"] = df_f["DateRef"].dt.to_period("M").astype(str)
monthly = df_f.groupby("Mois").size().reset_index(name="Nb").sort_values("Mois")

if use_plotly and not monthly.empty:
    fig2 = px.line(monthly, x="Mois", y="Nb", markers=True, height=420)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.dataframe(monthly, use_container_width=True)

# 3) Répartition par acteur / SAV
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Répartition par acteur")
    if "Acteur" in df_f.columns:
        by_acteur = df_f.groupby("Acteur").size().reset_index(name="Nb").sort_values("Nb", ascending=False)
    else:
        by_acteur = pd.DataFrame(columns=["Acteur", "Nb"])
    if use_plotly and not by_acteur.empty:
        fig3 = px.pie(by_acteur, names="Acteur", values="Nb", height=360)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.dataframe(by_acteur, use_container_width=True)

with c2:
    st.markdown("#### Répartition par SAV")
    if "SAV" in df_f.columns:
        by_sav = df_f.groupby("SAV").size().reset_index(name="Nb").sort_values("Nb", ascending=False)
    else:
        by_sav = pd.DataFrame(columns=["SAV", "Nb"])
    if use_plotly and not by_sav.empty:
        fig4 = px.pie(by_sav, names="SAV", values="Nb", height=360)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.dataframe(by_sav, use_container_width=True)

# --------------------
# Insights automatiques
# --------------------
st.markdown("---")
st.markdown("### 🧠 Insights automatiques")

end = datetime.combine(d2, datetime.min.time())
start = end - timedelta(days=29)

win = (df_stats["DateRef"] >= start) & (df_stats["DateRef"] <= end)
prev_win = (df_stats["DateRef"] >= start - timedelta(days=30)) & (df_stats["DateRef"] < start)

cur_n = int(df_stats[win].shape[0])
prev_n = int(df_stats[prev_win].shape[0])
delta = cur_n - prev_n
delta_pct = (delta / prev_n * 100.0) if prev_n > 0 else None

bullets = []

if delta_pct is not None:
    bullets.append(f"• **Volume 30j** : {cur_n} (Δ {delta:+d}, {delta_pct:+.1f}%)")
else:
    bullets.append(f"• **Volume 30j** : {cur_n} (Δ {delta:+d})")

if "Type erreur" in df_stats.columns:
    top_err = df_stats[win].groupby("Type erreur").size().sort_values(ascending=False)
    top_err_text = ", ".join([f"{k} ({v})" for k, v in top_err.head(3).items()]) if not top_err.empty else "—"
    bullets.append(f"• **Top erreurs (30j)** : {top_err_text}")

if "SAV" in df_stats.columns:
    rep_sav = df_stats[win].groupby("SAV").size().sort_values(ascending=False)
    rep_sav_text = ", ".join([f"{k} ({v})" for k, v in rep_sav.head(3).items()]) if not rep_sav.empty else "—"
    bullets.append(f"• **SAV les plus concernés (30j)** : {rep_sav_text}")

st.markdown("\n".join(bullets))

# --------------------
# Export CSV filtré
# --------------------
st.markdown("---")
st.subheader("Données filtrées")
st.dataframe(df_f, use_container_width=True, height=320)

st.download_button(
    "⬇️ Télécharger le CSV filtré",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="pannes_filtre.csv",
    mime="text/csv",
)
