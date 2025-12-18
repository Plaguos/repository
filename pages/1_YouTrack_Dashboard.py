import pandas as pd
import streamlit as st
import altair as alt

# NOTE: st.set_page_config est centralisé dans Home.py

def read_csv_with_fallback(uploaded_file) -> pd.DataFrame:
    """
    Lit le CSV en testant plusieurs encodages et choisit celui qui minimise
    les caractères de remplacement '�' (symptôme classique d'encodage).
    """
    candidates = ["utf-8-sig", "utf-8", "cp1252", "cp850", "latin1"]

    best_df = None
    best_bad = None

    for enc in candidates:
        try:
            uploaded_file.seek(0)
            df_try = pd.read_csv(uploaded_file, dtype=str, encoding=enc)

            sample = df_try.head(200).astype(str)
            bad = sample.apply(lambda s: s.str.count("�").sum()).sum()

            if best_bad is None or bad < best_bad:
                best_bad = bad
                best_df = df_try
                if bad == 0:
                    break
        except Exception:
            continue

    if best_df is None:
        uploaded_file.seek(0)
        best_df = pd.read_csv(uploaded_file, dtype=str)

    best_df.columns = [c.strip() for c in best_df.columns]
    return best_df


def parse_youtrack_date(series: pd.Series) -> pd.Series:
    """Convertit une colonne date YouTrack: timestamp ms ou ISO/texte."""
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        dt = pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
        return dt.dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    return pd.to_datetime(series, errors="coerce")


STATE_FR_FALLBACK = {
    "Open": "Ouvert",
    "In Progress": "En cours",
    "Submitted": "Soumis",
    "Reopened": "Réouvert",
    "Solved": "Résolu",
    "Resolved": "Résolu",
    "Done": "Terminé",
    "Closed": "Fermé",
    "Canceled": "Annulé",
}


def to_state_fr(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return "—"
    # si déjà français (ex: "Résolu"), garder
    if any(ch in s for ch in "éèêàçôûîïùÉÈÊÀÇÔÛÎÏÙ"):
        return s
    return STATE_FR_FALLBACK.get(s, s)


def multiselect_filter(df: pd.DataFrame, label: str, col: str):
    if col not in df.columns:
        return []
    options = sorted([x for x in df[col].dropna().unique().tolist() if str(x).strip() != ""])
    return st.sidebar.multiselect(label, options, default=[])


def main():
    st.title("Dashboard YouTrack (CSV)")

    uploaded = st.file_uploader("Charge ton CSV (ex: youtrack_issues_brut_clean.csv)", type=["csv"])
    if uploaded is None:
        st.info("Charge un fichier CSV pour commencer.")
        st.stop()

    df = read_csv_with_fallback(uploaded)

    # Dates
    for col in ["created", "updated"]:
        if col in df.columns:
            df[col] = parse_youtrack_date(df[col])

    # State en FR (colonne d'affichage)
    if "state" in df.columns:
        df["state_display"] = df["state"].apply(to_state_fr)
    elif "State" in df.columns:
        df["state_display"] = df["State"].apply(to_state_fr)
    else:
        df["state_display"] = "—"

    # Sidebar filtres
    st.sidebar.header("Filtres")

    f_state = multiselect_filter(df, "State (FR)", "state_display")
    f_type_mobigo = multiselect_filter(df, "Type (Mobigo)", "type_mobigo")
    f_interlocuteur = multiselect_filter(df, "Interlocuteur", "interlocuteur")
    f_ref = multiselect_filter(df, "Référent", "referent")
    f_prio = multiselect_filter(df, "Priority", "priority")

    text_search = st.sidebar.text_input("Recherche texte (summary contient)", "")

    date_from = None
    date_to = None
    if "created" in df.columns and df["created"].notna().any():
        min_d = df["created"].dropna().min()
        max_d = df["created"].dropna().max()
        st.sidebar.subheader("Période (created)")
        date_from, date_to = st.sidebar.date_input(
            "Du / au",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )

    # Application filtres
    dff = df.copy()

    def apply_filter(_dff: pd.DataFrame, col: str, values):
        if values and col in _dff.columns:
            return _dff[_dff[col].isin(values)]
        return _dff

    dff = apply_filter(dff, "state_display", f_state)
    dff = apply_filter(dff, "type_mobigo", f_type_mobigo)
    dff = apply_filter(dff, "interlocuteur", f_interlocuteur)
    dff = apply_filter(dff, "referent", f_ref)
    dff = apply_filter(dff, "priority", f_prio)

    if text_search.strip() and "summary" in dff.columns:
        dff = dff[dff["summary"].fillna("").str.contains(text_search, case=False, na=False)]

    if date_from and date_to and "created" in dff.columns:
        start = pd.to_datetime(date_from)
        end = pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        dff = dff[(dff["created"] >= start) & (dff["created"] <= end)]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets (filtrés)", len(dff))
    c2.metric("États (FR) uniques", int(dff["state_display"].nunique(dropna=True)) if "state_display" in dff.columns else 0)
    c3.metric("Types Mobigo uniques", int(dff["type_mobigo"].nunique(dropna=True)) if "type_mobigo" in dff.columns else 0)
    c4.metric("Interlocuteurs uniques", int(dff["interlocuteur"].nunique(dropna=True)) if "interlocuteur" in dff.columns else 0)

    st.divider()

    # Graphs
    left, right = st.columns(2)

    with left:
        st.subheader("Répartition par State (FR)")
        agg = dff["state_display"].fillna("—").value_counts().reset_index()
        agg.columns = ["state_display", "count"]
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X("state_display:N", sort="-y", title="State (FR)"),
            y=alt.Y("count:Q", title="Nb tickets"),
            tooltip=["state_display", "count"],
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.subheader("Top 15 Type (Mobigo)")
        if "type_mobigo" in dff.columns:
            agg = dff["type_mobigo"].fillna("—").value_counts().head(15).reset_index()
            agg.columns = ["type_mobigo", "count"]
            chart = alt.Chart(agg).mark_bar().encode(
                x=alt.X("count:Q", title="Nb tickets"),
                y=alt.Y("type_mobigo:N", sort="-x", title="Type (Mobigo)"),
                tooltip=["type_mobigo", "count"],
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Colonne 'type_mobigo' absente.")

    st.subheader("Graphique empilé : State (FR) empilé par Type (Mobigo)")
    normalize_type = st.toggle("Afficher en 100% empilé (Type Mobigo)", value=False)

    if "type_mobigo" in dff.columns:
        tmp = dff.copy()
        tmp["type_mobigo"] = tmp["type_mobigo"].fillna("—")
        tmp["state_display"] = tmp["state_display"].fillna("—")
        grp = tmp.groupby(["state_display", "type_mobigo"]).size().reset_index(name="count")

        chart = alt.Chart(grp).mark_bar().encode(
            x=alt.X("state_display:N", title="State (FR)", sort="-y"),
            y=alt.Y("count:Q", title=("Part" if normalize_type else "Nb tickets"),
                    stack="normalize" if normalize_type else "zero"),
            color=alt.Color("type_mobigo:N", title="Type (Mobigo)"),
            tooltip=["state_display", "type_mobigo", "count"],
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Pour l’empilé, il faut la colonne 'type_mobigo'.")

    st.subheader("Graphique empilé : State (FR) empilé par Interlocuteur")
    normalize_inter = st.toggle("Afficher en 100% empilé (Interlocuteur)", value=False)

    if "interlocuteur" in dff.columns:
        tmp = dff.copy()
        tmp["interlocuteur"] = tmp["interlocuteur"].fillna("—")
        tmp["state_display"] = tmp["state_display"].fillna("—")
        grp = tmp.groupby(["state_display", "interlocuteur"]).size().reset_index(name="count")

        chart = alt.Chart(grp).mark_bar().encode(
            x=alt.X("state_display:N", title="State (FR)", sort="-y"),
            y=alt.Y("count:Q", title=("Part" if normalize_inter else "Nb tickets"),
                    stack="normalize" if normalize_inter else "zero"),
            color=alt.Color("interlocuteur:N", title="Interlocuteur"),
            tooltip=["state_display", "interlocuteur", "count"],
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Colonne 'interlocuteur' absente.")

    st.subheader("Évolution des créations (par jour)")
    if "created" in dff.columns and dff["created"].notna().any():
        tmp = dff.dropna(subset=["created"]).copy()
        tmp["day"] = tmp["created"].dt.date
        agg = tmp.groupby("day").size().reset_index(name="count")
        chart = alt.Chart(agg).mark_line(point=True).encode(
            x=alt.X("day:T", title="Jour"),
            y=alt.Y("count:Q", title="Nb tickets créés"),
            tooltip=["day", "count"],
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("Colonne 'created' absente ou pas de dates exploitables.")

    st.subheader("Graphique empilé par jour (créations)")
    if "created" in dff.columns and dff["created"].notna().any():
        stack_by = st.selectbox(
            "Empiler par",
            options=[
                ("State (FR)", "state_display"),
                ("Type (Mobigo)", "type_mobigo"),
                ("Interlocuteur", "interlocuteur"),
                ("Priority", "priority"),
            ],
            format_func=lambda x: x[0],
            index=0,
        )[1]

        normalize_day = st.toggle("Afficher en 100% empilé (par jour)", value=False)

        tmp = dff.dropna(subset=["created"]).copy()
        tmp["day"] = tmp["created"].dt.date

        if stack_by not in tmp.columns:
            st.info(f"Colonne '{stack_by}' absente.")
        else:
            tmp[stack_by] = tmp[stack_by].fillna("—")
            grp = tmp.groupby(["day", stack_by]).size().reset_index(name="count")

            chart = alt.Chart(grp).mark_bar().encode(
                x=alt.X("day:T", title="Jour"),
                y=alt.Y("count:Q", title=("Part" if normalize_day else "Nb tickets créés"),
                        stack="normalize" if normalize_day else "zero"),
                color=alt.Color(f"{stack_by}:N", title="Empilé par"),
                tooltip=["day", stack_by, "count"],
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("Colonne 'created' absente ou pas de dates exploitables.")

    st.divider()

    # Tableau + export
    st.subheader("Tableau filtré")
    preferred_cols = [
        "idReadable",
        "summary",
        "state_display",
        "priority",
        "type",
        "type_mobigo",
        "interlocuteur",
        "referent",
        "email_demandeur",
        "created",
        "updated",
    ]
    cols_to_show = [c for c in preferred_cols if c in dff.columns]
    st.dataframe(dff[cols_to_show] if cols_to_show else dff, use_container_width=True, height=420)

    csv_bytes = dff.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Télécharger le CSV filtré",
        data=csv_bytes,
        file_name="youtrack_filtre.csv",
        mime="text/csv",
    )

main()
