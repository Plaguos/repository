# Tendance — "mode caméra Plotly" (aucun export d'image serveur)
# NOTE: st.set_page_config est centralisé dans Home.py

import re, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", message="Data Validation extension is not supported and will be removed")

MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

def mois_label(ts) -> str:
    if isinstance(ts, str):
        return ts
    if hasattr(ts, "to_timestamp"):
        try:
            ts = ts.to_timestamp()
        except Exception:
            pass
    if not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.to_datetime(ts, errors="coerce")
        except Exception:
            ts = pd.NaT
    if pd.isna(ts):
        return "n/a"
    return f"{MOIS_FR[int(ts.month)]} {int(ts.year)}"

def ensure_month_french(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns and not pd.api.types.is_string_dtype(out[col]):
        out[col] = out[col].apply(mois_label)
    return out

def parse_year_from_filename(name: str) -> int | None:
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else None

def main():
    st.title("Tendances réclamations — Multi-périodes")

    st.sidebar.title("Paramètres")

    uploaded_files = st.sidebar.file_uploader(
        "Fichiers Excel (plusieurs autorisés) — onglet `Liste_mantis`",
        type=["xlsx", "xlsm"], accept_multiple_files=True
    )

    header_row = st.sidebar.number_input("Ligne d'en-têtes (1 = première ligne)", min_value=1, value=45, step=1)
    header_idx = header_row - 1

    date_col = st.sidebar.text_input("Nom de la colonne Date", value="Date de soumission")
    type_col = st.sidebar.text_input("Nom de la colonne Type", value="Type ano")
    id_col_guess = st.sidebar.text_input("Nom de la colonne ID (si dispo)", value="ID")

    top_n_types = st.sidebar.slider("Top N types (graphe empilé)", 3, 20, 5, 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Anomalies (IsolationForest)")
    contamination = st.sidebar.slider("Contamination (proportion d'anomalies)", 0.01, 0.40, 0.15, 0.01)

    st.sidebar.markdown("---")
    show_stl = st.sidebar.checkbox("Afficher la décomposition STL (tendance/saisonnalité)", value=True)

    @st.cache_data(show_spinner=False, ttl=300)
    def load_single(file, header_idx, date_col, type_col, id_col_guess):
        df = pd.read_excel(file, sheet_name="Liste_mantis", header=header_idx)
        if date_col not in df.columns or type_col not in df.columns:
            raise ValueError(f"Colonnes introuvables : '{date_col}' / '{type_col}' dans {getattr(file, 'name', 'fichier')}")
        src_name = getattr(file, "name", "fichier")
        df["Source_fichier"] = src_name
        df["Annee_fichier"] = parse_year_from_filename(src_name)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()
        df["Mois"] = df[date_col].dt.to_period("M")
        df["Mois_debut"] = df["Mois"].dt.to_timestamp()
        id_present = id_col_guess in df.columns
        return df, id_present

    if not uploaded_files:
        st.info("💡 Charge un ou plusieurs fichiers Excel (.xlsx/.xlsm) à gauche pour démarrer.")
        st.stop()

    dfs, any_id_present, err_files = [], False, []
    for f in uploaded_files:
        try:
            tmp, has_id = load_single(f, header_idx, date_col, type_col, id_col_guess)
            dfs.append(tmp)
            any_id_present |= has_id
        except Exception as e:
            err_files.append(f"{f.name} → {e}")

    for m in err_files:
        st.warning(f"⚠️ {m}")

    if not dfs:
        st.error("Aucun fichier valide chargé.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)

    with st.sidebar.expander("Filtrer par fichier"):
        sel_sources = st.multiselect(
            "Fichiers pris en compte",
            options=sorted(df["Source_fichier"].unique()),
            default=sorted(df["Source_fichier"].unique())
        )
    if sel_sources:
        df = df[df["Source_fichier"].isin(sel_sources)].copy()

    # Déduplication
    if any_id_present and id_col_guess in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col_guess])
        after = len(df)
        dedup_info = f"Déduplication par '{id_col_guess}' : {before-after} doublon(s)."
    else:
        cols_for_key = [c for c in [date_col, type_col, "Source_fichier"] if c in df.columns]
        if "Résumé" in df.columns:
            cols_for_key.append("Résumé")
        if "Description" in df.columns:
            cols_for_key.append("Description")
        if not cols_for_key:
            cols_for_key = [date_col, type_col]
        before = len(df)
        key = df[cols_for_key].astype(str).agg("||".join, axis=1)
        df = df.loc[~key.duplicated()].copy()
        after = len(df)
        dedup_info = f"Déduplication composite ({', '.join(cols_for_key)}) : {before-after} doublon(s)."

    st.success(f"{len(uploaded_files)} fichier(s) — {len(df):,} lignes après fusion. {dedup_info}")

    # Filtres dynamiques
    all_types = sorted(df[type_col].dropna().astype(str).unique().tolist())
    sel_types = st.sidebar.multiselect("Filtrer Type", options=all_types, default=all_types)
    df = df[df[type_col].astype(str).isin(sel_types)].copy()

    min_d, max_d = df["Mois_debut"].min(), df["Mois_debut"].max()
    date_range = st.sidebar.date_input(
        "Période (Mois début)",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["Mois_debut"] >= start_d) & (df["Mois_debut"] <= end_d)].copy()

    if df.empty:
        st.warning("Aucune ligne après filtres.")
        st.stop()

    # Agrégations + réindexation calendrier
    monthly = df.groupby("Mois_debut").size().rename("Réclamations").sort_index()

    if not monthly.empty:
        full_months = pd.date_range(start=monthly.index.min(), end=monthly.index.max(), freq="MS")
    else:
        full_months = pd.date_range(freq="MS", periods=0, start=pd.Timestamp.today())
    monthly_full = monthly.reindex(full_months, fill_value=0)
    monthly_full.index.name = "Mois_debut"

    top_types = df[type_col].value_counts().head(top_n_types).index.tolist()
    monthly_by_type_raw = (
        df[df[type_col].isin(top_types)]
        .groupby(["Mois_debut", type_col]).size()
        .unstack(fill_value=0)
        .sort_index()
    )
    if not monthly_by_type_raw.empty:
        monthly_by_type = monthly_by_type_raw.reindex(full_months, fill_value=0)
        monthly_by_type.index.name = "Mois_debut"
    else:
        monthly_by_type = monthly_by_type_raw

    counts_by_type = (
        df.groupby(["Mois_debut", type_col]).size()
        .unstack(fill_value=0)
        .sort_index()
    )

    # Anomalies (IsolationForest)
    if len(monthly_full) >= 3:
        X = monthly_full.to_frame()
        cont = contamination if len(monthly_full) >= 12 else min(0.25, max(0.05, 2.0 / len(monthly_full)))
        iso = IsolationForest(n_estimators=200, contamination=cont, random_state=42)
        iso.fit(X)
        scores = pd.Series(iso.decision_function(X), index=monthly_full.index, name="score")
        preds = pd.Series(iso.predict(X), index=monthly_full.index, name="anomaly")  # -1 = anomalie
    else:
        scores = pd.Series(index=monthly_full.index, dtype=float, name="score")
        preds = pd.Series(1, index=monthly_full.index, name="anomaly")

    anomalies = pd.concat([monthly_full.rename("Réclamations"), scores, preds], axis=1)
    anomalies = anomalies[anomalies["anomaly"] == -1]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total réclamations", f"{int(monthly_full.sum())}")
    c2.metric("Moyenne mensuelle", f"{monthly_full.mean():.1f}")
    if len(monthly_full) > 0:
        mois_pic_dt = monthly_full.idxmax()
        c3.metric("Mois pic", mois_label(mois_pic_dt), delta=int(monthly_full.max()))
    else:
        c3.metric("Mois pic", "n/a")
    if not df.empty:
        top_overall = df[type_col].value_counts().idxmax()
        c4.metric("Type le plus fréquent", str(top_overall), delta=int(df[type_col].value_counts().max()))
    else:
        c4.metric("Type le plus fréquent", "n/a")

    with st.expander("ℹ️ Méthodologie"):
        st.markdown(f"""
        - Multi-fichiers (année détectée via nom) ; dédup par **{id_col_guess}** ou clé composite.
        - Agrégation mensuelle ; anomalies **IsolationForest**.
        - **STL (12)** pour tendance/saisonnalité/résiduel.
        - Comparaisons : **A↔B** et **même mois sur 2 années**.
        - Export d'images: **utilisez l’icône caméra de Plotly** sur chaque graphique.
        """)

    st.markdown("---")

    # Graphiques
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=monthly_full.index, y=monthly_full.values, mode="lines+markers", name="Réclamations"))
    if not anomalies.empty:
        fig1.add_trace(go.Scatter(
            x=anomalies.index, y=anomalies["Réclamations"], mode="markers",
            name="Anomalies", marker=dict(size=12, symbol="diamond")
        ))
    if len(monthly_full) >= 2:
        roll = monthly_full.rolling(3).mean()
        fig1.add_trace(go.Scatter(x=roll.index, y=roll.values, name="Moy. mobile 3m", line=dict(dash="dash")))
    if len(monthly_full) >= 3:
        med12 = monthly_full.tail(12).median() if len(monthly_full) >= 12 else monthly_full.median()
        fig1.add_hline(y=med12, line=dict(dash="dot"), annotation_text=f"Médiane : {med12:.0f}")

    fig1.update_layout(
        title="Tendance mensuelle (anomalies & références)",
        xaxis_title="Mois", yaxis_title="Réclamations",
        xaxis=dict(tickmode="array", tickvals=monthly_full.index, ticktext=[mois_label(x) for x in monthly_full.index])
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Export : cliquez sur l’icône caméra (barre du graphique).")

    # ✅ (1) Toggle d'affichage + (2) Normalisation % + (3) Ligne moyenne multi-années
    st.subheader("Total mensuel — vues & comparaison inter-années")

    view_total = st.radio(
        "Vue",
        ["Chronologique", "Comparer mêmes mois (Janvier avec Janvier, …)"],
        horizontal=True
    )

    if view_total == "Chronologique":
        df_total = monthly_full.rename("Réclamations").reset_index()
        df_total["Mois_label"] = df_total["Mois_debut"].apply(mois_label)
        cat_order_total = df_total.sort_values("Mois_debut")["Mois_label"].unique().tolist()

        fig_total = px.bar(
            df_total,
            x="Mois_label",
            y="Réclamations",
            title="Total des réclamations par mois (chronologique)"
        )
        fig_total.update_layout(
            xaxis_title="Mois",
            yaxis_title="Réclamations",
            xaxis=dict(categoryorder="array", categoryarray=cat_order_total)
        )
        st.plotly_chart(fig_total, use_container_width=True)
        st.caption("Export : cliquez sur l’icône caméra (barre du graphique).")

    else:
        copt1, copt2 = st.columns(2)
        with copt1:
            normalize_pct = st.toggle("Normaliser en % du total annuel", value=False)
        with copt2:
            show_avg_line = st.toggle("Afficher la ligne moyenne multi-années", value=True)

        df_cmp = monthly_full.rename("Valeur").reset_index()
        df_cmp["Année"] = df_cmp["Mois_debut"].dt.year.astype(int)
        df_cmp["Mois_num"] = df_cmp["Mois_debut"].dt.month.astype(int)
        df_cmp["Mois"] = df_cmp["Mois_num"].map(MOIS_FR)

        mois_order = [MOIS_FR[m] for m in range(1, 13)]
        years_order = sorted(df_cmp["Année"].unique().tolist())

        if normalize_pct:
            # % du total annuel (chaque année somme à 100%)
            denom = df_cmp.groupby("Année")["Valeur"].transform("sum").replace(0, np.nan)
            df_cmp["Réclamations"] = (df_cmp["Valeur"] / denom) * 100
            y_title = "% du total annuel"
        else:
            df_cmp["Réclamations"] = df_cmp["Valeur"]
            y_title = "Réclamations"

        # Figure combinée (barres par année + ligne moyenne optionnelle)
        fig_cmp_total = go.Figure()

        for y in years_order:
            d = df_cmp[df_cmp["Année"] == y].copy()
            # force ordre mois 1..12
            d = d.set_index("Mois_num").reindex(range(1, 13), fill_value=0).reset_index()
            d["Mois"] = d["Mois_num"].map(MOIS_FR)
            fig_cmp_total.add_trace(
                go.Bar(
                    x=d["Mois"],
                    y=d["Réclamations"],
                    name=str(y)
                )
            )

        if show_avg_line:
            avg = (
                df_cmp.groupby("Mois_num")["Réclamations"]
                .mean()
                .reindex(range(1, 13), fill_value=0)
            )
            fig_cmp_total.add_trace(
                go.Scatter(
                    x=[MOIS_FR[m] for m in avg.index],
                    y=avg.values,
                    name="Moyenne (multi-années)",
                    mode="lines+markers"
                )
            )

        fig_cmp_total.update_layout(
            title="Total par mois — comparaison inter-années (mois équivalents côte à côte)",
            barmode="group",
            xaxis_title="Mois",
            yaxis_title=y_title,
            xaxis=dict(categoryorder="array", categoryarray=mois_order),
            legend_title_text=""
        )

        st.plotly_chart(fig_cmp_total, use_container_width=True)
        st.caption("Export : cliquez sur l’icône caméra (barre du graphique).")

    # Empilé par type
    if not monthly_by_type.empty:
        df_stack = monthly_by_type.reset_index().melt(id_vars="Mois_debut", var_name="Type", value_name="Réclamations")
        df_stack["Mois_label"] = df_stack["Mois_debut"].apply(mois_label)
        cat_order = df_stack.sort_values("Mois_debut")["Mois_label"].unique().tolist()
        fig2 = px.bar(
            df_stack, x="Mois_label", y="Réclamations", color="Type",
            barmode="stack", title=f"Réclamations par type (Top {top_n_types})"
        )
        fig2.update_layout(
            xaxis_title="Mois", yaxis_title="Réclamations",
            xaxis=dict(categoryorder="array", categoryarray=cat_order)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Export : cliquez sur l’icône caméra (barre du graphique).")

    st.markdown("---")

    # Comparaison A ↔ B
    st.subheader("Comparaison manuelle entre deux mois")
    mois_options = list(monthly_full.index)
    mois_labels = [mois_label(m) for m in mois_options]
    colA, colB = st.columns(2)
    with colA:
        idx_a = st.selectbox(
            "Mois A (référence)",
            options=list(range(len(mois_options))),
            format_func=lambda i: mois_labels[i],
            index=0 if len(mois_options) > 0 else 0
        )
    with colB:
        idx_b = st.selectbox(
            "Mois B (à comparer)",
            options=list(range(len(mois_options))),
            format_func=lambda i: mois_labels[i],
            index=(len(mois_options) - 1 if len(mois_options) > 0 else 0)
        )

    if mois_options:
        mA, mB = mois_options[idx_a], mois_options[idx_b]
        valA = int(monthly_full.loc[mA]) if mA in monthly_full.index else 0
        valB = int(monthly_full.loc[mB]) if mB in monthly_full.index else 0
        delta = valB - valA
        pct = (delta / valA * 100) if valA != 0 else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Réclamations {mois_label(mA)}", valA)
        c2.metric(f"Réclamations {mois_label(mB)}", valB, delta=f"{delta:+d}")
        c3.metric("Écart (%)", f"{pct:+.1f}%" if not np.isnan(pct) else "n/a")

        if not counts_by_type.empty:
            sA = counts_by_type.loc[mA] if mA in counts_by_type.index else pd.Series(dtype=int)
            sB = counts_by_type.loc[mB] if mB in counts_by_type.index else pd.Series(dtype=int)
            all_cols = sorted(set(sA.index).union(set(sB.index)))
            sA = sA.reindex(all_cols, fill_value=0)
            sB = sB.reindex(all_cols, fill_value=0)
            diff = (sB - sA).sort_values(ascending=False)
            pct_diff = pd.Series(
                np.where(sA.values != 0, (sB.values - sA.values) / sA.values * 100, np.nan),
                index=all_cols
            )

            df_diff_plot = pd.DataFrame({
                "Type": diff.index.astype(str),
                "Écart (B - A)": diff.values,
                "Écart (%)": pct_diff.values
            })
            df_diff_plot["Écart (%)"] = df_diff_plot["Écart (%)"].round(1)
            df_diff_plot["couleur"] = np.where(df_diff_plot["Écart (B - A)"] >= 0, "Hausse", "Baisse")

            fig_cmp = px.bar(
                df_diff_plot, x="Type", y="Écart (B - A)", color="couleur",
                color_discrete_map={"Hausse": "#2ca02c", "Baisse": "#d62728"},
                title=f"Écarts par type — {mois_label(mA)} → {mois_label(mB)}"
            )
            fig_cmp.update_layout(xaxis_title="Type", yaxis_title="Écart de réclamations", legend_title="")
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption("Export : cliquez sur l’icône caméra (barre du graphique).")

    st.markdown("---")

    # Même mois sur deux années
    st.subheader("Comparer le même mois sur deux années")
    years_available = sorted(monthly_full.index.year.unique())
    months_available = sorted(monthly_full.index.month.unique())

    cmo1, cmo2, cmo3 = st.columns(3)
    with cmo1:
        month_pick = st.selectbox("Mois", options=months_available, format_func=lambda m: MOIS_FR[int(m)])
    with cmo2:
        y1 = st.selectbox("Année A", options=years_available, index=0 if years_available else 0)
    with cmo3:
        y2 = st.selectbox("Année B", options=years_available, index=(len(years_available) - 1 if years_available else 0))

    try:
        mA2 = pd.Timestamp(int(y1), int(month_pick), 1)
        mB2 = pd.Timestamp(int(y2), int(month_pick), 1)
        valA2 = int(monthly_full.get(mA2, 0))
        valB2 = int(monthly_full.get(mB2, 0))
        delta2 = valB2 - valA2
        pct2 = (delta2 / valA2 * 100) if valA2 != 0 else np.nan

        d1, d2, d3 = st.columns(3)
        d1.metric(f"Réclamations {MOIS_FR[int(month_pick)]} {y1}", valA2)
        d2.metric(f"Réclamations {MOIS_FR[int(month_pick)]} {y2}", valB2, delta=f"{delta2:+d}")
        d3.metric("Écart (%)", f"{pct2:+.1f}%" if not np.isnan(pct2) else "n/a")

        if not counts_by_type.empty:
            sA2 = counts_by_type.loc[mA2] if mA2 in counts_by_type.index else pd.Series(dtype=int)
            sB2 = counts_by_type.loc[mB2] if mB2 in counts_by_type.index else pd.Series(dtype=int)
            all_cols2 = sorted(set(sA2.index).union(set(sB2.index)))
            sA2 = sA2.reindex(all_cols2, fill_value=0)
            sB2 = sB2.reindex(all_cols2, fill_value=0)
            diff2 = (sB2 - sA2).sort_values(ascending=False)
            pct_diff2 = pd.Series(
                np.where(sA2.values != 0, (sB2.values - sA2.values) / sA2.values * 100, np.nan),
                index=all_cols2
            )

            df_same_month = pd.DataFrame({
                "Type": diff2.index.astype(str),
                f"Réclamations {y1}": sA2.values,
                f"Réclamations {y2}": sB2.values,
                "Écart (B - A)": diff2.values,
                "Écart (%)": pct_diff2.values
            })
            df_same_month["Écart (%)"] = df_same_month["Écart (%)"].round(1)

            fig_same = px.bar(
                df_same_month.assign(couleur=lambda d: np.where(d["Écart (B - A)"] >= 0, "Hausse", "Baisse")),
                x="Type", y="Écart (B - A)", color="couleur",
                color_discrete_map={"Hausse": "#2ca02c", "Baisse": "#d62728"},
                title=f"Écarts par type — {MOIS_FR[int(month_pick)]} {y1} vs {MOIS_FR[int(month_pick)]} {y2}"
            )
            fig_same.update_layout(xaxis_title="Type", yaxis_title="Écart de réclamations", legend_title="")
            st.plotly_chart(fig_same, use_container_width=True)
            st.caption("Export : cliquez sur l’icône caméra (barre du graphique).")

            st.dataframe(df_same_month)
    except Exception as e:
        st.info(f"Comparaison même mois : {e}")

    st.markdown("---")

    # Tableaux & exports CSV
    st.subheader("Tableaux")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tendance mensuelle (global)**")
        monthly_fmt = ensure_month_french(monthly_full.rename_axis("Mois").reset_index(), "Mois")
        st.dataframe(monthly_fmt)
    with c2:
        st.markdown(f"**Tendance par type (Top {top_n_types})**")
        if not monthly_by_type.empty:
            mbt_fmt = ensure_month_french(monthly_by_type.rename_axis("Mois").reset_index(), "Mois")
            st.dataframe(mbt_fmt)
        else:
            mbt_fmt = pd.DataFrame()
            st.write("n/a")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Anomalies détectées (IsolationForest)**")
        if anomalies.empty:
            tmp = pd.DataFrame(columns=["Mois", "Volume", "score", "anomaly"])
            st.write("Aucune anomalie détectée.")
        else:
            tmp = anomalies.copy()
            tmp.index.name = "Mois"
            tmp = tmp.reset_index().rename(columns={"Réclamations": "Volume"})
            tmp = ensure_month_french(tmp, "Mois")
            st.dataframe(tmp)
    with c4:
        st.markdown("**Table par type (complète)**")
        if not counts_by_type.empty:
            cbt_fmt = ensure_month_french(counts_by_type.rename_axis("Mois").reset_index(), "Mois")
            st.dataframe(cbt_fmt)
        else:
            cbt_fmt = pd.DataFrame()
            st.write("n/a")

    st.markdown("---")
    st.subheader("Exports CSV")

    def to_csv_download(df_obj, name):
        csv = df_obj.to_csv(index=False).encode("utf-8")
        st.download_button(f"Télécharger — {name}.csv", data=csv, file_name=f"{name}.csv", mime="text/csv")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        to_csv_download(monthly_fmt, "tendance_mensuelle")
    with d2:
        if not mbt_fmt.empty:
            to_csv_download(mbt_fmt, "tendance_par_type_topN")
    with d3:
        if not tmp.empty:
            to_csv_download(tmp, "anomalies_isolationforest")
    with d4:
        if not cbt_fmt.empty:
            to_csv_download(cbt_fmt, "table_par_type_complete")

    st.markdown("---")
    st.subheader("Détail par mois")
    if len(monthly_full) > 0:
        mois_det = st.selectbox("Choisir un mois", options=monthly_full.index, format_func=mois_label)
        detail = df[df["Mois_debut"] == mois_det].copy()
        cols_show = [c for c in [date_col, type_col, "Source_fichier", "Annee_fichier"] if c in detail.columns]
        if not cols_show:
            cols_show = detail.columns.tolist()
        st.dataframe(detail[cols_show].sort_values(date_col))
        st.download_button(
            f"⬇️ Exporter le détail — {mois_label(mois_det)}",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name=f"reclamations_detail_{mois_det.strftime('%Y-%m')}.csv",
            mime="text/csv"
        )

    # ✅ STL — déplacé à la fin
    if show_stl and len(monthly_full) >= 6:
        s = monthly_full.asfreq("MS").fillna(0)
        try:
            res = STL(s, period=12, robust=True).fit()
            fig_stl = go.Figure()
            fig_stl.add_trace(go.Scatter(x=s.index, y=res.trend, name="Tendance"))
            fig_stl.add_trace(go.Scatter(x=s.index, y=res.seasonal, name="Saisonnalité"))
            fig_stl.add_trace(go.Scatter(x=s.index, y=res.resid, name="Résiduel"))
            fig_stl.update_layout(title="Décomposition STL (12 mois)", xaxis_title="Mois")
            st.plotly_chart(fig_stl, use_container_width=True)
            st.caption("Export : cliquez sur l’icône caméra (barre du graphique).")

            trend = res.trend.dropna()
            evol = (trend.iloc[-1] - trend.iloc[0]) / max(abs(trend.iloc[0]), 1e-9) * 100 if len(trend) >= 2 else 0.0
            saison_amplitude = (res.seasonal.max() - res.seasonal.min()) / 2
            resid_abs = res.resid.abs()
            resid_peak_idx = resid_abs.idxmax()
            resid_peak_val = res.resid.loc[resid_peak_idx]
            resid_dir = "positive" if resid_peak_val > 0 else "négative"

            phrase = []
            if evol > 2:
                phrase.append(f"Tendance générale à la hausse (+{evol:.1f} %).")
            elif evol < -2:
                phrase.append(f"Tendance générale à la baisse ({evol:.1f} %).")
            else:
                phrase.append("Tendance globale stable.")
            phrase.append(
                "Saisonnalité marquée (écart typique ±{:.1f}).".format(saison_amplitude)
                if saison_amplitude > 5 else "Peu de saisonnalité observable."
            )
            phrase.append(
                f"Anomalie {resid_dir} détectée autour de {mois_label(resid_peak_idx)} "
                f"(écart de {resid_peak_val:+.0f} par rapport à la tendance)."
            )

            st.markdown("### Interprétation automatique (STL)")
            st.info(" ".join(phrase))

        except Exception as e:
            st.info(f"STL non affichée : {e}")

main()
