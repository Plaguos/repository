# app.py
# Comparateur GTFS PROD vs TEST (Streamlit) + module "Fiches horaires" (PROD vs TEST)
# - Upload PROD et TEST à chaque fois
# - Mode 1: Comparaison Jour (écarts + ajoutées/supprimées + drill-down arrêt par arrêt)
# - Mode 2: Fiches horaires : comparer PROD vs TEST sur une semaine "Scolaire" et une semaine "Vacances"
#          -> basé sur le départ de la course (1er arrêt), comme tes fiches
# - Par défaut : liste toutes les lignes (🔁 communes / 🆕 nouvelles TEST / ❌ absentes TEST)

import io
import zipfile
import pandas as pd
import streamlit as st
from datetime import date, timedelta


# ---------------------- Lecture GTFS ----------------------
def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    try:
        with zf.open(name) as f:
            return pd.read_csv(f)
    except UnicodeDecodeError:
        with zf.open(name) as f:
            return pd.read_csv(f, encoding="latin-1")


def load_gtfs_from_bytes(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    tables = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for n in zf.namelist():
            if n.endswith(".txt"):
                tables[n] = _read_csv_from_zip(zf, n)
    return tables


# ---------------------- Calendrier (service_id actifs) ----------------------
def active_services(tables: dict[str, pd.DataFrame], d: date) -> set[str]:
    d_yyyymmdd = int(d.strftime("%Y%m%d"))
    weekday = d.strftime("%A").lower()
    day_col = {
        "monday": "monday",
        "tuesday": "tuesday",
        "wednesday": "wednesday",
        "thursday": "thursday",
        "friday": "friday",
        "saturday": "saturday",
        "sunday": "sunday",
    }[weekday]

    services: set[str] = set()

    if "calendar.txt" in tables:
        cal = tables["calendar.txt"].copy()
        cal["start_date"] = cal["start_date"].astype(int)
        cal["end_date"] = cal["end_date"].astype(int)
        cal = cal[
            (cal["start_date"] <= d_yyyymmdd)
            & (cal["end_date"] >= d_yyyymmdd)
            & (cal[day_col].astype(int) == 1)
        ]
        services |= set(cal["service_id"].astype(str))

    if "calendar_dates.txt" in tables:
        ex = tables["calendar_dates.txt"].copy()
        ex["date"] = ex["date"].astype(int)
        ex = ex[ex["date"] == d_yyyymmdd]
        adds = set(ex[ex["exception_type"].astype(int) == 1]["service_id"].astype(str))
        removes = set(ex[ex["exception_type"].astype(int) == 2]["service_id"].astype(str))
        services |= adds
        services -= removes

    return services


# ---------------------- Helpers temps / routes ----------------------
def parse_gtfs_time_to_minutes(t: str | float | int | None) -> int | None:
    """Convertit 'HH:MM:SS' (HH peut dépasser 24) en minutes depuis 00:00."""
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    s = str(t).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        hh, mm, _ss = s.split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return None


def get_routes(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    r = tables["routes.txt"].copy()
    r["route_id"] = r["route_id"].astype(str)
    for col in ["route_short_name", "route_long_name"]:
        if col not in r.columns:
            r[col] = ""
    r["label"] = (
        r["route_short_name"].astype(str).fillna("")
        + " — "
        + r["route_long_name"].astype(str).fillna("")
    )
    return r.sort_values("label")


def week_dates(any_day: date) -> list[date]:
    monday = any_day - timedelta(days=any_day.weekday())
    return [monday + timedelta(days=i) for i in range(7)]


# ---------------------- Résumés "courses" ----------------------
def compute_trip_summary_for_route_day(tables: dict[str, pd.DataFrame], d: date, route_id: str) -> pd.DataFrame:
    """
    Résumé par course (trip) pour comparaison visuelle :
    - départ = heure au 1er arrêt
    - arrivée = heure au dernier arrêt
    - + signature robuste pour matcher PROD/TEST même si trip_id change
    """
    services = active_services(tables, d)

    trips = tables["trips.txt"].copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    trips["service_id"] = trips["service_id"].astype(str)
    trips = trips[(trips["route_id"] == str(route_id)) & (trips["service_id"].isin(services))]

    stt = tables["stop_times.txt"][["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]].copy()
    stt["trip_id"] = stt["trip_id"].astype(str)
    stt["stop_id"] = stt["stop_id"].astype(str)

    stops = tables.get("stops.txt")
    if stops is not None:
        stops = stops[["stop_id", "stop_name"]].copy()
        stops["stop_id"] = stops["stop_id"].astype(str)

    # trips_extensions + companies (transporteur réel)
    te = tables.get("trips_extensions.txt")
    comp = tables.get("companies.txt")
    if te is not None:
        te = te.copy()
        te["trip_id"] = te["trip_id"].astype(str)
        if "exec_company_id" in te.columns:
            te["exec_company_id"] = te["exec_company_id"].astype(str)
        if comp is not None and "company_id" in comp.columns:
            comp = comp.copy()
            comp["company_id"] = comp["company_id"].astype(str)
            te = te.merge(comp[["company_id", "company_name"]], left_on="exec_company_id", right_on="company_id", how="left")
        else:
            te["company_name"] = None

    sorted_stt = stt.sort_values(["trip_id", "stop_sequence"], kind="mergesort")
    first = (
        sorted_stt.groupby("trip_id", as_index=False)
        .head(1)
        .rename(columns={"stop_id": "start_stop_id", "departure_time": "start_departure"})
    )[["trip_id", "start_stop_id", "start_departure"]]
    last = (
        sorted_stt.groupby("trip_id", as_index=False)
        .tail(1)
        .rename(columns={"stop_id": "end_stop_id", "arrival_time": "end_arrival"})
    )[["trip_id", "end_stop_id", "end_arrival"]]

    out_cols = ["trip_id"]
    if "direction_id" in trips.columns:
        out_cols.append("direction_id")
    out = trips[out_cols].copy()
    if "direction_id" not in out.columns:
        out["direction_id"] = None

    out = out.merge(first, on="trip_id", how="left").merge(last, on="trip_id", how="left")

    if stops is not None:
        out = out.merge(stops.rename(columns={"stop_id": "start_stop_id", "stop_name": "start_stop_name"}), on="start_stop_id", how="left")
        out = out.merge(stops.rename(columns={"stop_id": "end_stop_id", "stop_name": "end_stop_name"}), on="end_stop_id", how="left")
    else:
        out["start_stop_name"] = None
        out["end_stop_name"] = None

    if te is not None:
        keep = [c for c in ["trip_id", "company_name", "indic_reservation"] if c in te.columns]
        out = out.merge(te[keep], on="trip_id", how="left")
    else:
        out["company_name"] = None
        out["indic_reservation"] = None

    out["dep_min"] = out["start_departure"].map(parse_gtfs_time_to_minutes)
    out["arr_min"] = out["end_arrival"].map(parse_gtfs_time_to_minutes)

    out["sig_start"] = out["start_stop_id"].astype(str) + "|" + out["start_stop_name"].fillna("").astype(str)
    out["sig_end"] = out["end_stop_id"].astype(str) + "|" + out["end_stop_name"].fillna("").astype(str)

    # Tolérance : arrondi à 2 minutes
    def round2(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return int(2 * round(x / 2))

    out["dep_round"] = out["dep_min"].apply(round2)
    out["arr_round"] = out["arr_min"].apply(round2)

    out["signature"] = (
        "r=" + str(route_id)
        + "|dir=" + out["direction_id"].astype(str)
        + "|A=" + out["sig_start"].astype(str)
        + "|B=" + out["sig_end"].astype(str)
        + "|d=" + out["dep_round"].astype(str)
        + "|a=" + out["arr_round"].astype(str)
    )

    out = out.rename(
        columns={
            "company_name": "Transporteur",
            "indic_reservation": "Réservation",
            "direction_id": "Direction",
            "start_stop_name": "Départ",
            "end_stop_name": "Arrivée",
            "start_departure": "Heure départ",
            "end_arrival": "Heure arrivée",
        }
    )
    out["service_id"] = trips["service_id"].values  # utile en debug/recette

    return out.sort_values(["dep_min", "trip_id"], kind="mergesort").reset_index(drop=True)


def trip_starts_for_route_day(tables: dict[str, pd.DataFrame], d: date, route_id: str) -> pd.DataFrame:
    """
    Pour fiches horaires : départ de la course = stop_times au 1er arrêt.
    On sort aussi service_id (utile pour contrôler scolaire/vacances).
    """
    services = active_services(tables, d)

    trips = tables["trips.txt"].copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    trips["service_id"] = trips["service_id"].astype(str)
    trips = trips[(trips["route_id"] == str(route_id)) & (trips["service_id"].isin(services))]

    stt = tables["stop_times.txt"][["trip_id", "stop_sequence", "departure_time"]].copy()
    stt["trip_id"] = stt["trip_id"].astype(str)

    first = (
        stt.sort_values(["trip_id", "stop_sequence"], kind="mergesort")
        .groupby("trip_id", as_index=False)
        .head(1)[["trip_id", "departure_time"]]
        .rename(columns={"departure_time": "Départ course"})
    )

    out = trips.merge(first, on="trip_id", how="left")
    if "direction_id" in out.columns:
        out = out.rename(columns={"direction_id": "Direction"})
    else:
        out["Direction"] = None

    out["dep_min"] = out["Départ course"].map(parse_gtfs_time_to_minutes)
    out = out.sort_values(["dep_min", "trip_id"], kind="mergesort")

    return out[["Départ course", "Direction", "service_id", "trip_id"]].reset_index(drop=True)


def get_stop_times_detail(tables: dict[str, pd.DataFrame], trip_id: str) -> pd.DataFrame:
    stt = tables["stop_times.txt"].copy()
    stt["trip_id"] = stt["trip_id"].astype(str)
    stt["stop_id"] = stt["stop_id"].astype(str)

    df = stt[stt["trip_id"] == str(trip_id)][["stop_sequence", "stop_id", "arrival_time", "departure_time"]].copy()

    stops = tables.get("stops.txt")
    if stops is not None:
        stops = stops[["stop_id", "stop_name"]].copy()
        stops["stop_id"] = stops["stop_id"].astype(str)
        df = df.merge(stops, on="stop_id", how="left")

    df = df.sort_values("stop_sequence", kind="mergesort").rename(
        columns={
            "stop_sequence": "Ordre",
            "stop_name": "Arrêt",
            "arrival_time": "Arrivée",
            "departure_time": "Départ",
            "stop_id": "stop_id",
        }
    )
    cols = ["Ordre", "Arrêt", "Arrivée", "Départ", "stop_id"]
    return df[cols] if "Arrêt" in df.columns else df[["Ordre", "stop_id", "Arrivée", "Départ"]]


# ---------------------- Comparaison PROD vs TEST ----------------------
def compare_prod_test(prod: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_cols = [
        "trip_id",
        "signature",
        "Heure départ",
        "Heure arrivée",
        "dep_min",
        "arr_min",
        "Départ",
        "Arrivée",
        "Direction",
        "Transporteur",
        "Réservation",
    ]
    p = prod[[c for c in base_cols if c in prod.columns]].copy()
    t = test[[c for c in base_cols if c in test.columns]].copy()

    p_map = p.set_index("signature")
    t_map = t.set_index("signature")

    common = p_map.index.intersection(t_map.index)
    only_p = p_map.index.difference(t_map.index)
    only_t = t_map.index.difference(p_map.index)

    matched = p_map.loc[common].reset_index().merge(
        t_map.loc[common].reset_index(),
        on="signature",
        how="inner",
        suffixes=("_PROD", "_TEST"),
    )

    if "dep_min_TEST" in matched.columns and "dep_min_PROD" in matched.columns:
        matched["Δ départ (min)"] = matched["dep_min_TEST"] - matched["dep_min_PROD"]
    if "arr_min_TEST" in matched.columns and "arr_min_PROD" in matched.columns:
        matched["Δ arrivée (min)"] = matched["arr_min_TEST"] - matched["arr_min_PROD"]

    cols = [
        "Direction_PROD",
        "Départ_PROD",
        "Heure départ_PROD",
        "Arrivée_PROD",
        "Heure arrivée_PROD",
        "Direction_TEST",
        "Départ_TEST",
        "Heure départ_TEST",
        "Arrivée_TEST",
        "Heure arrivée_TEST",
        "Δ départ (min)",
        "Δ arrivée (min)",
        "Transporteur_PROD",
        "Transporteur_TEST",
        "Réservation_PROD",
        "Réservation_TEST",
        "trip_id_PROD",
        "trip_id_TEST",
        "signature",
    ]
    matched = matched[[c for c in cols if c in matched.columns]].sort_values(["Heure départ_PROD", "trip_id_PROD"], kind="mergesort")

    removed = p_map.loc[only_p].reset_index().rename(columns={"trip_id": "trip_id_PROD"})
    added = t_map.loc[only_t].reset_index().rename(columns={"trip_id": "trip_id_TEST"})
    return matched.reset_index(drop=True), removed.reset_index(drop=True), added.reset_index(drop=True)


# ---------------------- UI ----------------------
st.set_page_config(page_title="Comparateur GTFS PROD vs TEST", layout="wide")
st.title("Comparateur GTFS — PROD vs TEST")

col_up1, col_up2 = st.columns(2)
with col_up1:
    prod_file = st.file_uploader("GTFS PROD (.zip)", type=["zip"], key="prod")
with col_up2:
    test_file = st.file_uploader("GTFS TEST (.zip)", type=["zip"], key="test")

if not prod_file or not test_file:
    st.info("Charge les deux fichiers (PROD et TEST) pour démarrer.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_cached(content: bytes) -> dict[str, pd.DataFrame]:
    return load_gtfs_from_bytes(content)

prod_tables = load_cached(prod_file.getvalue())
test_tables = load_cached(test_file.getvalue())

for lbl, tables in [("PROD", prod_tables), ("TEST", test_tables)]:
    missing = [x for x in ["routes.txt", "trips.txt", "stop_times.txt"] if x not in tables]
    if missing:
        st.error(f"{lbl} incomplet : fichiers manquants {', '.join(missing)}")
        st.stop()

mode = st.radio(
    "Mode",
    ["Comparaison (Jour)", "Fiches horaires (PROD vs TEST)"],
    horizontal=True,
)

# --- Liste des lignes avec statut (par défaut) ---
prod_routes = get_routes(prod_tables)
test_routes = get_routes(test_tables)

p_by_id = prod_routes.set_index("route_id")
t_by_id = test_routes.set_index("route_id")

prod_ids = set(p_by_id.index)
test_ids = set(t_by_id.index)

common_ids = sorted(prod_ids & test_ids)
only_test_ids = sorted(test_ids - prod_ids)
only_prod_ids = sorted(prod_ids - test_ids)

def _route_label(df_row) -> str:
    try:
        return str(df_row.get("label", "")).strip()
    except Exception:
        return ""

routes_all = []
for rid in common_ids:
    routes_all.append({"route_id": rid, "status": "COMMON", "status_tag": "🔁",
                       "label": _route_label(t_by_id.loc[rid]) or _route_label(p_by_id.loc[rid]) or rid})
for rid in only_test_ids:
    routes_all.append({"route_id": rid, "status": "TEST_ONLY", "status_tag": "🆕",
                       "label": _route_label(t_by_id.loc[rid]) or rid})
for rid in only_prod_ids:
    routes_all.append({"route_id": rid, "status": "PROD_ONLY", "status_tag": "❌",
                       "label": _route_label(p_by_id.loc[rid]) or rid})

routes_all_df = pd.DataFrame(routes_all)
status_order = {"TEST_ONLY": 0, "COMMON": 1, "PROD_ONLY": 2}
routes_all_df["order"] = routes_all_df["status"].map(status_order).fillna(9).astype(int)
routes_all_df = routes_all_df.sort_values(["order", "label"], kind="mergesort").reset_index(drop=True)

cA, cB, cC = st.columns(3)
cA.metric("Lignes communes", len(common_ids))
cB.metric("Nouvelles lignes (TEST)", len(only_test_ids))
cC.metric("Lignes absentes en TEST", len(only_prod_ids))

default_idx = 0
if len(only_test_ids) > 0:
    default_idx = int(routes_all_df.index[routes_all_df["status"] == "TEST_ONLY"][0])
elif len(common_ids) > 0:
    default_idx = int(routes_all_df.index[routes_all_df["status"] == "COMMON"][0])

chosen_idx = st.selectbox(
    "Ligne",
    options=list(range(len(routes_all_df))),
    index=int(default_idx),
    format_func=lambda i: f'{routes_all_df.loc[i,"status_tag"]} {routes_all_df.loc[i,"label"]}  (route_id={routes_all_df.loc[i,"route_id"]})'
)

route_id = routes_all_df.loc[chosen_idx, "route_id"]
route_status = routes_all_df.loc[chosen_idx, "status"]
chosen_label = routes_all_df.loc[chosen_idx, "label"]

if route_status == "TEST_ONLY":
    st.warning(f"🆕 Ligne nouvelle en TEST : **{chosen_label}** (absente en PROD).")
elif route_status == "PROD_ONLY":
    st.warning(f"❌ Ligne absente en TEST : **{chosen_label}** (présente en PROD).")
else:
    st.info(f"🔁 Ligne commune : **{chosen_label}**")

st.divider()

# ---------------------- MODE 1: Comparaison (Jour) ----------------------
if mode == "Comparaison (Jour)":
    top = st.columns([1, 1])
    with top[0]:
        chosen_date = st.date_input("Jour à comparer", value=date.today())
    with top[1]:
        show_only_diffs = st.toggle("Afficher uniquement les écarts", value=True)

    prod_sum = compute_trip_summary_for_route_day(prod_tables, chosen_date, route_id) if route_status != "TEST_ONLY" else pd.DataFrame()
    test_sum = compute_trip_summary_for_route_day(test_tables, chosen_date, route_id) if route_status != "PROD_ONLY" else pd.DataFrame()

    if route_status == "COMMON":
        matched, removed, added = compare_prod_test(prod_sum, test_sum)
    elif route_status == "TEST_ONLY":
        matched, removed = pd.DataFrame(), pd.DataFrame()
        added = test_sum.rename(columns={"trip_id": "trip_id_TEST"}).copy()
    elif route_status == "PROD_ONLY":
        matched, added = pd.DataFrame(), pd.DataFrame()
        removed = prod_sum.rename(columns={"trip_id": "trip_id_PROD"}).copy()
    else:
        matched = removed = added = pd.DataFrame()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Courses PROD", 0 if prod_sum.empty else len(prod_sum))
    k2.metric("Courses TEST", 0 if test_sum.empty else len(test_sum))
    k3.metric("Matchées", 0 if matched.empty else len(matched))
    k4.metric("Ajoutées / Supprimées", f"{0 if added.empty else len(added)} / {0 if removed.empty else len(removed)}")

    tab1, tab2, tab3, tab4 = st.tabs(["Comparaison", "Ajoutées (TEST)", "Supprimées (PROD)", "Détail course (arrêts)"])

    with tab1:
        st.subheader("Courses matchées (signature) — écarts de temps")
        if route_status != "COMMON":
            st.info("Pas de comparaison possible (ligne non commune).")
        else:
            view = matched.copy()
            if show_only_diffs and not view.empty and "Δ départ (min)" in view.columns:
                view = view[(view["Δ départ (min)"] != 0) | (view.get("Δ arrivée (min)", 0) != 0)]
            st.dataframe(view, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Courses présentes en TEST mais absentes en PROD")
        if route_status == "PROD_ONLY":
            st.info("Ligne absente en TEST (donc aucune course TEST).")
        else:
            if added.empty:
                st.write("Aucune.")
            else:
                cols = ["Direction", "Départ", "Heure départ", "Arrivée", "Heure arrivée",
                        "Transporteur", "Réservation", "trip_id_TEST", "signature"]
                st.dataframe(added[[c for c in cols if c in added.columns]], use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Courses présentes en PROD mais absentes en TEST")
        if route_status == "TEST_ONLY":
            st.info("Ligne absente en PROD (donc aucune course PROD).")
        else:
            if removed.empty:
                st.write("Aucune.")
            else:
                cols = ["Direction", "Départ", "Heure départ", "Arrivée", "Heure arrivée",
                        "Transporteur", "Réservation", "trip_id_PROD", "signature"]
                st.dataframe(removed[[c for c in cols if c in removed.columns]], use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Comparer une course arrêt par arrêt")
        if route_status != "COMMON":
            st.info("Drill-down PROD vs TEST disponible uniquement pour une ligne commune.")
        elif matched.empty:
            st.info("Aucune course matchée pour ce jour/ligne.")
        else:
            pick_labels = (
                matched["Heure départ_PROD"].fillna("").astype(str)
                + " — " + matched["Départ_PROD"].fillna("").astype(str)
                + " → " + matched["Arrivée_PROD"].fillna("").astype(str)
                + "  (PROD " + matched["trip_id_PROD"].astype(str)
                + " / TEST " + matched["trip_id_TEST"].astype(str) + ")"
            )
            idx = st.selectbox("Choisir une course matchée", list(range(len(matched))), format_func=lambda i: pick_labels.iloc[i])
            row = matched.iloc[idx]
            prod_trip = row["trip_id_PROD"]
            test_trip = row["trip_id_TEST"]

            left, right = st.columns(2)
            with left:
                st.markdown("**PROD — stop_times**")
                st.dataframe(get_stop_times_detail(prod_tables, prod_trip), use_container_width=True, hide_index=True)
            with right:
                st.markdown("**TEST — stop_times**")
                st.dataframe(get_stop_times_detail(test_tables, test_trip), use_container_width=True, hide_index=True)

# ---------------------- MODE 2: Fiches horaires (PROD vs TEST) ----------------------
else:
    st.subheader("Fiches horaires — PROD vs TEST (départs de courses / 1er arrêt)")

    colA, colB = st.columns(2)
    with colA:
        d_scol = st.date_input("Date dans une semaine SCOLAIRE", value=date.today(), key="ds")
    with colB:
        d_vac = st.date_input("Date dans une semaine VACANCES", value=date.today(), key="dv")

    ws = week_dates(d_scol)
    wv = week_dates(d_vac)

    def times_list(df: pd.DataFrame) -> list[str]:
        return df["Départ course"].dropna().astype(str).tolist() if not df.empty else []

    def render_day_compare(day: date, route_id_: str):
        prod_df = trip_starts_for_route_day(prod_tables, day, route_id_) if route_status != "TEST_ONLY" else pd.DataFrame()
        test_df = trip_starts_for_route_day(test_tables, day, route_id_) if route_status != "PROD_ONLY" else pd.DataFrame()

        prod_times = times_list(prod_df)
        test_times = times_list(test_df)

        prod_set = set(prod_times)
        test_set = set(test_times)
        added = sorted(test_set - prod_set)     # présents en TEST
        removed = sorted(prod_set - test_set)   # présents en PROD
        return prod_df, test_df, prod_times, test_times, added, removed

    def render_week_prod_vs_test(title: str, week: list[date]):
        st.markdown(f"## {title} (semaine du lundi {week[0].isoformat()})")

        export_rows = []

        for d in week:
            prod_df, test_df, prod_times, test_times, added, removed = render_day_compare(d, route_id)

            # Export : PROD
            for _, r in prod_df.iterrows():
                export_rows.append({
                    "week_type": title,
                    "source": "PROD",
                    "date": d.isoformat(),
                    "weekday": d.strftime("%a"),
                    "route_id": str(route_id),
                    "route_label": chosen_label,
                    "departure_time": r.get("Départ course"),
                    "direction": r.get("Direction"),
                    "service_id": r.get("service_id"),
                    "trip_id": r.get("trip_id"),
                })
            # Export : TEST
            for _, r in test_df.iterrows():
                export_rows.append({
                    "week_type": title,
                    "source": "TEST",
                    "date": d.isoformat(),
                    "weekday": d.strftime("%a"),
                    "route_id": str(route_id),
                    "route_label": chosen_label,
                    "departure_time": r.get("Départ course"),
                    "direction": r.get("Direction"),
                    "service_id": r.get("service_id"),
                    "trip_id": r.get("trip_id"),
                })

            lcol, rcol = st.columns(2)
            with lcol:
                st.markdown(f"**PROD — {d.strftime('%a %d/%m')} — {len(prod_times)} départs**")
                st.write(", ".join(prod_times) if prod_times else "—")
                if not prod_df.empty:
                    top_services = prod_df["service_id"].astype(str).value_counts().head(3)
                    st.caption("service_id (top): " + ", ".join([f"{sid}×{cnt}" for sid, cnt in top_services.items()]))

            with rcol:
                st.markdown(f"**TEST — {d.strftime('%a %d/%m')} — {len(test_times)} départs**")
                st.write(", ".join(test_times) if test_times else "—")
                if not test_df.empty:
                    top_services = test_df["service_id"].astype(str).value_counts().head(3)
                    st.caption("service_id (top): " + ", ".join([f"{sid}×{cnt}" for sid, cnt in top_services.items()]))

            if added or removed:
                st.warning(f"Diff {d.strftime('%a %d/%m')} — Ajoutés(TEST): {len(added)} | Supprimés(PROD): {len(removed)}")
                if added:
                    st.write("Ajoutés (ex): " + ", ".join(added[:20]) + (" ..." if len(added) > 20 else ""))
                if removed:
                    st.write("Supprimés (ex): " + ", ".join(removed[:20]) + (" ..." if len(removed) > 20 else ""))
            else:
                st.success(f"{d.strftime('%a %d/%m')} — mêmes départs (PROD = TEST)")

            st.divider()

        if export_rows:
            export_df = pd.DataFrame(export_rows)
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Export CSV — {title} (PROD vs TEST)",
                data=csv_bytes,
                file_name=f"fiches_{title.lower()}_route_{route_id}.csv",
                mime="text/csv",
            )

    render_week_prod_vs_test("Scolaire", ws)
    st.divider()
    render_week_prod_vs_test("Vacances", wv)
