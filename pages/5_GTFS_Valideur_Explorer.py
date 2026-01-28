# =========================
# GTFS_Valideur_Explorer.py
# SCRIPT COMPLET — PART 1/5
# (Imports + Vacances + Lecture GTFS + Helpers + CourseCode10)
# =========================

import io
import zipfile
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import streamlit as st

# -------- Optional exports --------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None  # PDF export will be disabled if reportlab not installed


# =======================
# Vacances 2025-2026
# =======================
VACANCES_2025_2026 = [
    ("TOUSSAINT", date(2025, 10, 18), date(2025, 11, 2)),
    ("NOËL", date(2025, 12, 20), date(2026, 1, 4)),
    ("HIVER", date(2026, 2, 7), date(2026, 2, 22)),
    ("PRINTEMPS", date(2026, 4, 4), date(2026, 4, 19)),
    ("ÉTÉ", date(2026, 7, 4), date(2099, 12, 31)),
]


def is_vacances(d: date) -> bool:
    return any(start <= d <= end for _name, start, end in VACANCES_2025_2026)


def vacances_label(d: date) -> str:
    for name, start, end in VACANCES_2025_2026:
        if start <= d <= end:
            return f"VACANCES ({name})"
    return "SCOLAIRE"


# =======================
# Utils (Excel / colonnes)
# =======================
def normalize_colname(s: str) -> str:
    """Normalise un nom de colonne: minuscules, sans accents, sans espaces/underscore."""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace(" ", "").replace("_", "")
    return s


def _normalize_hhmm_token(s: str) -> str | None:
    """Normalise 'H:MM' / 'HH:MM' -> 'HH:MM' ; accepte heures 0..47."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(s))
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 47 or mm < 0 or mm > 59:
        return None
    return f"{hh:02d}:{mm:02d}"


# =======================
# Core GTFS helpers
# =======================
def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """
    Lecture robuste + préservation des zéros non significatifs (ex: '0121').
    IMPORTANT : trips.txt => trip_headsign / trip_short_name en string dès la lecture.
    """
    dtype_map = None
    low = name.lower()

    if low.endswith("trips.txt"):
        dtype_map = {
            "trip_id": "string",
            "route_id": "string",
            "service_id": "string",
            "trip_headsign": "string",
            "trip_short_name": "string",
            "direction_id": "string",
        }
    elif low.endswith("routes.txt"):
        dtype_map = {
            "route_id": "string",
            "route_short_name": "string",
            "route_long_name": "string",
        }
    elif low.endswith("stop_times.txt"):
        dtype_map = {
            "trip_id": "string",
            "stop_id": "string",
            "arrival_time": "string",
            "departure_time": "string",
            "stop_sequence": "string",
        }
    elif low.endswith("calendar.txt"):
        dtype_map = {"service_id": "string", "start_date": "string", "end_date": "string"}
    elif low.endswith("calendar_dates.txt"):
        dtype_map = {"service_id": "string", "date": "string", "exception_type": "string"}

    try:
        with zf.open(name) as f:
            return pd.read_csv(
                f,
                dtype=dtype_map,
                keep_default_na=False,
                na_filter=False,
            )
    except UnicodeDecodeError:
        with zf.open(name) as f:
            return pd.read_csv(
                f,
                encoding="latin-1",
                dtype=dtype_map,
                keep_default_na=False,
                na_filter=False,
            )


def load_gtfs_from_bytes(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for n in zf.namelist():
            if n.endswith(".txt"):
                tables[n] = _read_csv_from_zip(zf, n)
    return tables


def parse_gtfs_time_to_minutes(t) -> int | None:
    """Convert 'HH:MM:SS' (HH can exceed 24) into minutes from 00:00."""
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


def week_dates(any_day: date) -> list[date]:
    monday = any_day - timedelta(days=any_day.weekday())
    return [monday + timedelta(days=i) for i in range(7)]


# =======================
# Calendar / services
# =======================
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
        # robust int cast
        cal["start_date"] = pd.to_numeric(cal.get("start_date", ""), errors="coerce")
        cal["end_date"] = pd.to_numeric(cal.get("end_date", ""), errors="coerce")
        if day_col in cal.columns:
            cal[day_col] = pd.to_numeric(cal[day_col], errors="coerce")
        cal = cal.dropna(subset=["start_date", "end_date"])
        cal = cal[
            (cal["start_date"].astype(int) <= d_yyyymmdd)
            & (cal["end_date"].astype(int) >= d_yyyymmdd)
            & (pd.to_numeric(cal.get(day_col, 0), errors="coerce").fillna(0).astype(int) == 1)
        ]
        if "service_id" in cal.columns:
            services |= set(cal["service_id"].astype(str))

    if "calendar_dates.txt" in tables:
        ex = tables["calendar_dates.txt"].copy()
        ex["date"] = pd.to_numeric(ex.get("date", ""), errors="coerce")
        ex = ex.dropna(subset=["date"])
        ex = ex[ex["date"].astype(int) == d_yyyymmdd]
        if "exception_type" in ex.columns and "service_id" in ex.columns:
            exc = pd.to_numeric(ex["exception_type"], errors="coerce").fillna(0).astype(int)
            adds = set(ex[exc == 1]["service_id"].astype(str))
            removes = set(ex[exc == 2]["service_id"].astype(str))
            services |= adds
            services -= removes

    return services


def active_services_filtered(
    tables: dict[str, pd.DataFrame],
    d: date,
    allowed_service_ids: set[str] | None,
) -> set[str]:
    s = active_services(tables, d)
    if allowed_service_ids is None:
        return s
    allowed = set(map(str, allowed_service_ids))
    return set(x for x in s if str(x) in allowed)


# =======================
# Routes
# =======================
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
    ).str.strip()

    return r.sort_values("label").reset_index(drop=True)


# =======================
# CourseCode (trip_headsign-trip_short_name) -> CourseCode10
# =======================
def add_course_code(trips: pd.DataFrame, max_len: int = 10) -> pd.DataFrame:
    """
    CourseCode_full = trip_headsign-trip_short_name
    CourseCode10     = 10 premiers caractères (ou max_len)
    CourseCode_tronqué = True si full dépasse max_len
    IMPORTANT : préserve les zéros non significatifs (lecture trips.txt en string).
    """
    t = trips.copy()

    for col in ["trip_headsign", "trip_short_name"]:
        if col not in t.columns:
            t[col] = ""
        t[col] = t[col].astype("string")

    t["trip_headsign"] = t["trip_headsign"].fillna("").astype(str).str.strip()
    t["trip_short_name"] = t["trip_short_name"].fillna("").astype(str).str.strip()

    t["CourseCode_full"] = (t["trip_headsign"] + "-" + t["trip_short_name"]).astype(str)
    t["CourseCode_full"] = t["CourseCode_full"].str.strip().str.strip("-").str.strip()

    t["CourseCode_tronqué"] = t["CourseCode_full"].str.len() > max_len
    t["CourseCode10"] = t["CourseCode_full"].str.slice(0, max_len)

    return t
# =======================
# Vacances 2025-2026
# =======================
VACANCES_2025_2026 = [
    ("TOUSSAINT", date(2025, 10, 18), date(2025, 11, 2)),
    ("NOËL", date(2025, 12, 20), date(2026, 1, 4)),
    ("HIVER", date(2026, 2, 7), date(2026, 2, 22)),
    ("PRINTEMPS", date(2026, 4, 4), date(2026, 4, 19)),
    ("ÉTÉ", date(2026, 7, 4), date(2099, 12, 31)),
]


def is_vacances(d: date) -> bool:
    return any(start <= d <= end for _, start, end in VACANCES_2025_2026)


def vacances_label(d: date) -> str:
    for name, start, end in VACANCES_2025_2026:
        if start <= d <= end:
            return f"VACANCES ({name})"
    return "SCOLAIRE"


# =======================
# CourseCode (trip_headsign-trip_short_name)
# =======================
def add_course_code(trips: pd.DataFrame, max_len: int = 10) -> pd.DataFrame:
    """
    CourseCode_full = trip_headsign-trip_short_name
    CourseCode10    = tronqué à max_len
    CourseCode_tronqué = True si dépassement
    """
    t = trips.copy()

    for col in ["trip_headsign", "trip_short_name"]:
        if col not in t.columns:
            t[col] = ""
        t[col] = t[col].astype("string")

    t["trip_headsign"] = t["trip_headsign"].fillna("").astype(str).str.strip()
    t["trip_short_name"] = t["trip_short_name"].fillna("").astype(str).str.strip()

    t["CourseCode_full"] = (t["trip_headsign"] + "-" + t["trip_short_name"]).str.strip("-")
    t["CourseCode_tronqué"] = t["CourseCode_full"].str.len() > max_len
    t["CourseCode10"] = t["CourseCode_full"].str.slice(0, max_len)

    return t


# =======================
# Fiche GTFS (départ + arrivée)
# =======================
def trip_dep_arr_for_route_day(
    tables: dict[str, pd.DataFrame],
    d: date,
    route_id: str,
    direction: str | None,
    allowed_service_ids: set[str] | None,
) -> pd.DataFrame:
    """
    Une ligne = une course
    - Départ : 1er arrêt
    - Arrivée : dernier arrêt
    - CourseCode10 inclus
    """
    services = active_services_filtered(tables, d, allowed_service_ids)

    trips = tables["trips.txt"].copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    trips["service_id"] = trips["service_id"].astype(str)

    trips = trips[
        (trips["route_id"] == str(route_id)) &
        (trips["service_id"].isin(services))
    ]

    if "direction_id" not in trips.columns:
        trips["direction_id"] = None
    if direction and direction != "(toutes)":
        trips = trips[trips["direction_id"].astype(str) == str(direction)]

    # stop_times
    stt = tables["stop_times.txt"].copy()
    stt["trip_id"] = stt["trip_id"].astype(str)
    stt["stop_sequence"] = pd.to_numeric(stt["stop_sequence"], errors="coerce")
    stt = stt.dropna(subset=["stop_sequence"])
    stt = stt.sort_values(["trip_id", "stop_sequence"], kind="mergesort")

    first = (
        stt.groupby("trip_id", as_index=False)
        .head(1)[["trip_id", "departure_time"]]
        .rename(columns={"departure_time": "Départ"})
    )

    if "arrival_time" in stt.columns:
        last_raw = (
            stt.groupby("trip_id", as_index=False)
            .tail(1)[["trip_id", "arrival_time", "departure_time"]]
        )
        last_raw["Arrivée"] = last_raw["arrival_time"].where(
            last_raw["arrival_time"].astype(str).str.len() > 0,
            last_raw["departure_time"],
        )
        last = last_raw[["trip_id", "Arrivée"]]
    else:
        last = (
            stt.groupby("trip_id", as_index=False)
            .tail(1)[["trip_id", "departure_time"]]
            .rename(columns={"departure_time": "Arrivée"})
        )

    out = (
        trips
        .merge(first, on="trip_id", how="left")
        .merge(last, on="trip_id", how="left")
    )

    # 🔑 AJOUT CourseCode10
    out = add_course_code(out, max_len=10)

    out["dep_min"] = out["Départ"].map(parse_gtfs_time_to_minutes)
    out = out.sort_values(["dep_min", "trip_id"], kind="mergesort")

    out = out.rename(columns={"direction_id": "Direction"})

    return out[
        [
            "Départ",
            "Arrivée",
            "CourseCode10",
            "CourseCode_tronqué",
            "Direction",
            "service_id",
            "trip_id",
        ]
    ].reset_index(drop=True)


def make_week_fiche_table(
    tables: dict[str, pd.DataFrame],
    route_id: str,
    week: list[date],
    direction: str | None,
    allowed_service_ids: set[str] | None,
) -> pd.DataFrame:
    rows = []
    for d in week:
        df = trip_dep_arr_for_route_day(
            tables,
            d,
            route_id,
            direction,
            allowed_service_ids,
        )
        for _, r in df.iterrows():
            rows.append({
                "date": d.isoformat(),
                "weekday": d.strftime("%A"),
                "departure_time": r["Départ"],
                "arrival_time": r["Arrivée"],
                "CourseCode10": r["CourseCode10"],
                "CourseCode_tronqué": r["CourseCode_tronqué"],
                "direction": r["Direction"],
                "service_id": r["service_id"],
                "trip_id": r["trip_id"],
            })
    return pd.DataFrame(rows)
# =========================
# SCRIPT COMPLET — PART 3/5
# (Exports Excel/PDF + Quality checks + Streamlit setup + TAB 1 complet)
# =========================

# =======================
# Exports (Excel / PDF)
# =======================
def export_excel(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
    return out.getvalue()


def export_excel_multi_sheets(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    Exporte un Excel multi-onglets (un onglet par jour).
    `sheets` : dict {sheet_name: dataframe}
    """
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = str(name)[:31] if name else "sheet"
            df.to_excel(writer, sheet_name=safe, index=False)
    return out.getvalue()


def export_pdf_simple(title: str, df: pd.DataFrame) -> bytes:
    if canvas is None:
        raise RuntimeError("reportlab non installé")

    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    width, height = A4

    x = 40
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 25

    c.setFont("Helvetica", 9)
    cols = ["date", "weekday", "departure_time", "arrival_time", "CourseCode10", "direction", "service_id", "trip_id"]
    header = " | ".join(cols)
    c.drawString(x, y, header)
    y -= 14

    for _, row in df.head(1500).iterrows():
        line = " | ".join([str(row.get(col, "")) for col in cols])
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)
        c.drawString(x, y, line[:120])
        y -= 12

    c.save()
    return out.getvalue()


# =======================
# Quality checks
# =======================
@dataclass
class Issue:
    level: str  # "ERROR" | "WARN" | "INFO"
    where: str
    message: str
    count: int | None = None


def run_quality_checks(tables: dict[str, pd.DataFrame]) -> tuple[list[Issue], dict[str, pd.DataFrame]]:
    issues: list[Issue] = []
    details: dict[str, pd.DataFrame] = {}

    for req in ["routes.txt", "trips.txt", "stop_times.txt"]:
        if req not in tables:
            issues.append(Issue("ERROR", req, "Fichier manquant (indispensable)."))
            return issues, details

    routes = tables["routes.txt"].copy()
    trips = tables["trips.txt"].copy()
    stt = tables["stop_times.txt"].copy()

    for df, col in [(routes, "route_id"), (trips, "route_id"), (trips, "trip_id"), (stt, "trip_id")]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # trips sans stop_times
    trip_ids = set(trips["trip_id"].astype(str))
    stt_trip_ids = set(stt["trip_id"].astype(str))
    missing_st = sorted(list(trip_ids - stt_trip_ids))
    if missing_st:
        issues.append(Issue("ERROR", "stop_times", "Trips sans stop_times.", count=len(missing_st)))
        details["trips_sans_stop_times"] = trips[trips["trip_id"].isin(missing_st)].head(200)

    # routes sans trips
    route_ids = set(routes["route_id"].astype(str))
    trip_route_ids = set(trips["route_id"].astype(str))
    no_trips_routes = sorted(list(route_ids - trip_route_ids))
    if no_trips_routes:
        issues.append(Issue("WARN", "routes/trips", "Routes sans trips.", count=len(no_trips_routes)))
        details["routes_sans_trips"] = routes[routes["route_id"].isin(no_trips_routes)].head(200)

    # stop_sequence invalides / doublons
    if "stop_sequence" in stt.columns:
        tmp = stt[["trip_id", "stop_sequence"]].copy()
        tmp["stop_sequence"] = pd.to_numeric(tmp["stop_sequence"], errors="coerce")
        bad_seq = tmp[tmp["stop_sequence"].isna()]
        if not bad_seq.empty:
            issues.append(Issue("ERROR", "stop_times.stop_sequence", "stop_sequence invalide.", count=len(bad_seq)))
            details["stop_sequence_invalide"] = bad_seq.head(200)

        dup = tmp.dropna().duplicated(subset=["trip_id", "stop_sequence"], keep=False)
        dup_df = tmp.dropna()[dup].sort_values(["trip_id", "stop_sequence"])
        if not dup_df.empty:
            issues.append(Issue("ERROR", "stop_times", "stop_sequence dupliqués dans un trip.", count=len(dup_df)))
            details["stop_sequence_doublons"] = dup_df.head(200)

    # times invalides / régressions
    if "departure_time" in stt.columns and "stop_sequence" in stt.columns:
        cols = ["trip_id", "stop_sequence", "departure_time"]
        if "arrival_time" in stt.columns:
            cols.append("arrival_time")

        tdf = stt[cols].copy()
        tdf["stop_sequence"] = pd.to_numeric(tdf["stop_sequence"], errors="coerce")
        tdf["dep_min"] = tdf["departure_time"].map(parse_gtfs_time_to_minutes)

        bad_time = tdf[tdf["dep_min"].isna()]
        if not bad_time.empty:
            issues.append(Issue("WARN", "stop_times.departure_time", "departure_time invalide/vide.", count=len(bad_time)))
            details["departure_time_invalide"] = bad_time.head(200)

        tdf2 = tdf.dropna(subset=["stop_sequence", "dep_min"]).sort_values(["trip_id", "stop_sequence"], kind="mergesort")
        tdf2["dep_prev"] = tdf2.groupby("trip_id")["dep_min"].shift(1)
        reg = tdf2[(tdf2["dep_prev"].notna()) & (tdf2["dep_min"] < tdf2["dep_prev"])]
        if not reg.empty:
            issues.append(Issue("ERROR", "stop_times", "Heures de départ régressives dans un trip.", count=len(reg)))
            details["depart_regressif"] = reg.head(200)

    # service_id orphelins
    if "service_id" in trips.columns:
        trip_services = set(trips["service_id"].astype(str))
    else:
        trip_services = set()

    cal_services = set()
    if "calendar.txt" in tables and "service_id" in tables["calendar.txt"].columns:
        cal_services |= set(tables["calendar.txt"]["service_id"].astype(str))
    if "calendar_dates.txt" in tables and "service_id" in tables["calendar_dates.txt"].columns:
        cal_services |= set(tables["calendar_dates.txt"]["service_id"].astype(str))

    if trip_services and cal_services:
        orphan = sorted(list(trip_services - cal_services))
        if orphan:
            issues.append(Issue("ERROR", "calendar/trips", "service_id dans trips mais absents calendar.", count=len(orphan)))
            details["service_id_orphelins"] = pd.DataFrame({"service_id": orphan}).head(200)

    if not issues:
        issues.append(Issue("INFO", "global", "Aucun problème détecté par ces contrôles."))

    return issues, details


# =======================
# Streamlit UI (setup)
# =======================
st.set_page_config(page_title="GTFS Explorer + Fiches + Compare Excel", layout="wide")
st.title("GTFS — Explorateur, Fiches horaires (Vue 2), Comparaison Excel Région")

uploaded = st.file_uploader("Choisir un GTFS (.zip)", type=["zip"], key="gtfs_upload")
if not uploaded:
    st.info("Upload un GTFS .zip pour commencer.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_cached(content: bytes):
    return load_gtfs_from_bytes(content)


tables = load_cached(uploaded.getvalue())

tabs = st.tabs(
    ["1) Explorateur + Qualité + Collisions", "2) Fiches horaires (Vue 2)", "3) Comparaison GTFS vs Excel Région"]
)

# =======================
# TAB 1 — Explorateur + Qualité + Collisions CourseCode10
# =======================
with tabs[0]:
    st.subheader("Structure & exploration")

    files_present = sorted(list(tables.keys()))
    st.write("**Fichiers présents :**", ", ".join(files_present))

    def _count(name):
        return len(tables[name]) if name in tables else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("routes", _count("routes.txt"))
    c2.metric("trips", _count("trips.txt"))
    c3.metric("stop_times", _count("stop_times.txt"))
    c4.metric("stops", _count("stops.txt"))

    if "trips.txt" in tables:
        _t = add_course_code(tables["trips.txt"], max_len=10)
        c5.metric("CourseCode10 distincts", _t["CourseCode10"].replace("", pd.NA).dropna().nunique())
    else:
        c5.metric("CourseCode10 distincts", "—")

    if "calendar.txt" in tables and {"start_date", "end_date"}.issubset(set(tables["calendar.txt"].columns)):
        cal = tables["calendar.txt"].copy()
        try:
            sd = int(pd.to_numeric(cal["start_date"], errors="coerce").dropna().min())
            ed = int(pd.to_numeric(cal["end_date"], errors="coerce").dropna().max())
            st.caption(f"Période calendar.txt : {sd} → {ed}")
        except Exception:
            pass

    st.divider()
    st.subheader("Recherche (CourseCode = trip_headsign-trip_short_name, tronqué à 10)")

    search_mode = st.selectbox("Type de recherche", ["Ligne", "Service", "CourseCode", "Arrêt", "Trip"], key="t1_mode")

    def _download_csv_button(df: pd.DataFrame, label: str, filename: str):
        if df is None or df.empty:
            st.info("Aucune donnée à exporter.")
            return
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

    if search_mode == "Ligne":
        if "routes.txt" not in tables:
            st.error("routes.txt manquant")
        else:
            routes = get_routes(tables)
            chosen = st.selectbox("Choisir une ligne", routes["label"].tolist(), key="t1_route")
            rid = routes.loc[routes["label"] == chosen, "route_id"].iloc[0]

            st.write("**route_id :**", rid)
            st.dataframe(routes[routes["route_id"] == rid], use_container_width=True, hide_index=True)

            trips = tables.get("trips.txt", pd.DataFrame()).copy()
            if trips.empty:
                st.info("trips.txt vide ou absent.")
            else:
                trips["route_id"] = trips["route_id"].astype(str)
                trips["trip_id"] = trips["trip_id"].astype(str)
                if "service_id" in trips.columns:
                    trips["service_id"] = trips["service_id"].astype(str)
                if "direction_id" not in trips.columns:
                    trips["direction_id"] = None

                trips = add_course_code(trips, max_len=10)
                sub = trips[trips["route_id"] == str(rid)]

                st.write("**CourseCode10 — occurrences (top 200)**")
                if sub.empty:
                    st.info("Aucun trip pour cette ligne.")
                else:
                    vc = sub["CourseCode10"].replace("", pd.NA).dropna().value_counts().head(200)
                    df_vc = vc.rename_axis("CourseCode10").reset_index(name="count")
                    df_vc["dupliqué"] = df_vc["count"] > 1
                    st.dataframe(df_vc, use_container_width=True, hide_index=True)

                    st.write("**Collisions (CourseCode10 dupliqués) — détail**")
                    dup_codes = df_vc[df_vc["dupliqué"]]["CourseCode10"].tolist()
                    collisions = sub[sub["CourseCode10"].isin(dup_codes)].copy()

                    if collisions.empty:
                        st.success("Aucun CourseCode10 dupliqué sur cette ligne.")
                    else:
                        cols = [c for c in [
                            "route_id", "trip_id", "service_id", "direction_id",
                            "CourseCode10", "CourseCode_tronqué", "CourseCode_full",
                            "trip_headsign", "trip_short_name"
                        ] if c in collisions.columns]
                        collisions = collisions.sort_values(["CourseCode10", "CourseCode_tronqué", "trip_id"], kind="mergesort")
                        st.dataframe(collisions[cols], use_container_width=True, hide_index=True)

                        _download_csv_button(
                            collisions[cols],
                            label="⬇️ Export CSV — collisions (ligne)",
                            filename=f"collisions_route_{rid}_CourseCode10.csv"
                        )

    elif search_mode == "Service":
        trips = tables.get("trips.txt", pd.DataFrame()).copy()
        if trips.empty or "service_id" not in trips.columns:
            st.error("trips.txt/service_id manquant")
        else:
            trips["service_id"] = trips["service_id"].astype(str)
            trips["trip_id"] = trips["trip_id"].astype(str)
            if "route_id" in trips.columns:
                trips["route_id"] = trips["route_id"].astype(str)
            if "direction_id" not in trips.columns:
                trips["direction_id"] = None

            trips = add_course_code(trips, max_len=10)

            service_ids = sorted(trips["service_id"].unique().tolist())
            sid = st.selectbox("service_id", service_ids, key="t1_sid")
            sub = trips[trips["service_id"] == sid].copy()

            st.write(f"**CourseCode10 par service_id={sid} (top 200)**")
            vc = sub["CourseCode10"].replace("", pd.NA).dropna().value_counts().head(200)
            df_vc = vc.rename_axis("CourseCode10").reset_index(name="count")
            df_vc["dupliqué"] = df_vc["count"] > 1
            st.dataframe(df_vc, use_container_width=True, hide_index=True)

            st.write("**Collisions (CourseCode10 dupliqués) — détail**")
            dup_codes = df_vc[df_vc["dupliqué"]]["CourseCode10"].tolist()
            collisions = sub[sub["CourseCode10"].isin(dup_codes)].copy()

            if collisions.empty:
                st.success("Aucune collision sur ce service_id.")
            else:
                cols = [c for c in [
                    "route_id", "trip_id", "service_id", "direction_id",
                    "CourseCode10", "CourseCode_tronqué", "CourseCode_full",
                    "trip_headsign", "trip_short_name"
                ] if c in collisions.columns]
                collisions = collisions.sort_values(["CourseCode10", "CourseCode_tronqué", "route_id", "trip_id"], kind="mergesort")
                st.dataframe(collisions[cols], use_container_width=True, hide_index=True)

                _download_csv_button(
                    collisions[cols],
                    label="⬇️ Export CSV — collisions (service_id)",
                    filename=f"collisions_service_{sid}_CourseCode10.csv"
                )

            st.write("**Trips (200 max) — avec CourseCode10**")
            cols2 = [c for c in [
                "route_id", "trip_id", "direction_id",
                "trip_headsign", "trip_short_name",
                "CourseCode10", "CourseCode_tronqué", "CourseCode_full",
                "service_id"
            ] if c in sub.columns]
            st.dataframe(sub[cols2].head(200), use_container_width=True, hide_index=True)

    elif search_mode == "CourseCode":
        trips = tables.get("trips.txt", pd.DataFrame()).copy()
        if trips.empty:
            st.error("trips.txt manquant")
        else:
            trips["trip_id"] = trips["trip_id"].astype(str)
            for c in ["route_id", "service_id"]:
                if c in trips.columns:
                    trips[c] = trips[c].astype(str)
            if "direction_id" not in trips.columns:
                trips["direction_id"] = None

            trips = add_course_code(trips, max_len=10)

            q = st.text_input("Filtrer CourseCode (contient, sur le FULL)", value="", key="t1_ccq")
            sub = trips
            if q.strip():
                sub = sub[sub["CourseCode_full"].astype(str).str.contains(q, case=False, na=False)]

            st.write("**CourseCode10 (top 200)**")
            vc = sub["CourseCode10"].replace("", pd.NA).dropna().value_counts().head(200)
            df_vc = vc.rename_axis("CourseCode10").reset_index(name="count")
            df_vc["dupliqué"] = df_vc["count"] > 1
            st.dataframe(df_vc, use_container_width=True, hide_index=True)

            st.write("**Collisions (CourseCode10 dupliqués) — détail**")
            dup_codes = df_vc[df_vc["dupliqué"]]["CourseCode10"].tolist()
            collisions = sub[sub["CourseCode10"].isin(dup_codes)].copy()

            if collisions.empty:
                st.success("Aucune collision (dans le sous-ensemble filtré).")
            else:
                cols = [c for c in [
                    "route_id", "trip_id", "service_id", "direction_id",
                    "CourseCode10", "CourseCode_tronqué", "CourseCode_full",
                    "trip_headsign", "trip_short_name"
                ] if c in collisions.columns]
                collisions = collisions.sort_values(["CourseCode10", "CourseCode_tronqué", "route_id", "trip_id"], kind="mergesort")
                st.dataframe(collisions[cols], use_container_width=True, hide_index=True)

                _download_csv_button(
                    collisions[cols],
                    label="⬇️ Export CSV — collisions (filtre CourseCode)",
                    filename="collisions_CourseCode10_filtre.csv"
                )

    elif search_mode == "Arrêt":
        if "stops.txt" not in tables:
            st.error("stops.txt manquant")
        else:
            q = st.text_input("Recherche stop_name (contient)", value="", key="t1_stopq")
            stops = tables["stops.txt"].copy()
            if "stop_name" not in stops.columns:
                st.error("stops.txt sans stop_name")
            else:
                if q.strip():
                    sub = stops[stops["stop_name"].astype(str).str.contains(q, case=False, na=False)].head(200)
                else:
                    sub = stops.head(200)
                st.dataframe(sub, use_container_width=True, hide_index=True)

    else:  # Trip
        trips = tables.get("trips.txt", pd.DataFrame()).copy()
        if trips.empty:
            st.error("trips.txt manquant")
        else:
            q = st.text_input("trip_id (exact ou contient)", value="", key="t1_tripq")
            trips["trip_id"] = trips["trip_id"].astype(str)
            trips = add_course_code(trips, max_len=10)

            if q.strip():
                sub = trips[trips["trip_id"].str.contains(q, case=False, na=False)].head(200)
            else:
                sub = trips.head(200)

            cols = [c for c in [
                "route_id", "trip_id", "direction_id",
                "trip_headsign", "trip_short_name",
                "CourseCode10", "CourseCode_tronqué", "CourseCode_full",
                "service_id"
            ] if c in sub.columns]
            st.dataframe(sub[cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Qualité (contrôles rapides)")

    issues, details = run_quality_checks(tables)

    issues_df = pd.DataFrame([{
        "level": i.level,
        "where": i.where,
        "message": i.message,
        "count": i.count
    } for i in issues]).sort_values(["level", "where"], kind="mergesort")

    st.dataframe(issues_df, use_container_width=True, hide_index=True)

    if details:
        st.write("Détails (échantillons)")
        detail_key = st.selectbox("Voir détail", ["(aucun)"] + sorted(details.keys()), key="t1_detail")
        if detail_key != "(aucun)":
            st.dataframe(details[detail_key], use_container_width=True, hide_index=True)
# =========================
# =========================
# TAB 2 — VUE 2 (lisible, colonnes = courses)
# =========================
with tabs[1]:
    st.subheader("Fiches horaires — Vue 2 (colonnes = courses)")

    routes = get_routes(tables)
    chosen = st.selectbox("Ligne", routes["label"].tolist(), key="vue2_route")
    route_id = routes.loc[routes["label"] == chosen, "route_id"].iloc[0]

    ref_day = st.date_input("Semaine de référence", value=date.today(), key="vue2_week")
    week = week_dates(ref_day)

    # direction
    trips_all = tables["trips.txt"].copy()
    if "direction_id" in trips_all.columns:
        dirs = sorted(
            trips_all[trips_all["route_id"].astype(str) == str(route_id)]
            ["direction_id"].dropna().astype(str).unique().tolist()
        )
        direction = st.selectbox("Direction", ["(toutes)"] + dirs, key="vue2_dir")
    else:
        direction = "(toutes)"

    # régime
    regime = "VACANCES" if is_vacances(ref_day) else "SCOLAIRE"
    st.caption(f"Régime détecté : **{vacances_label(ref_day)}**")

    # service_id mapping
    all_service_ids = sorted(trips_all["service_id"].astype(str).unique().tolist())
    scolaire_ids = st.multiselect("service_id SCOLAIRE", all_service_ids, key="vue2_sco")
    vacances_ids = st.multiselect("service_id VACANCES", all_service_ids, key="vue2_vac")

    allowed_service_ids = set(scolaire_ids if regime == "SCOLAIRE" else vacances_ids)
    if not allowed_service_ids:
        allowed_service_ids = None

    st.divider()
    st.markdown("### Vue 2 — lecture humaine (comme une fiche papier)")

    sheets: dict[str, pd.DataFrame] = {}

    # ✅ table CourseCode10 prête une fois (évite recalcul dans la boucle)
    trips_cc = add_course_code(tables["trips.txt"], max_len=10)[["trip_id", "CourseCode10"]].copy()
    trips_cc["trip_id"] = trips_cc["trip_id"].astype(str)
    trips_cc["CourseCode10"] = trips_cc["CourseCode10"].fillna("").astype(str)

    for d in week:
        df = trip_dep_arr_for_route_day(
            tables,
            d,
            route_id,
            direction,
            allowed_service_ids
        )

        if df.empty:
            continue

        # ✅ sécuriser type trip_id pour merge fiable
        df["trip_id"] = df["trip_id"].astype(str)

        # ✅ merge CourseCode10
        df = df.merge(trips_cc, on="trip_id", how="left")

        # ✅ garantir colonne + strings
        if "CourseCode10" not in df.columns:
            df["CourseCode10"] = ""
        df["CourseCode10"] = df["CourseCode10"].fillna("").astype(str)

        df = df.sort_values("Départ").reset_index(drop=True)

        # format colonne = une course (avec anti-collision)
        cols = {}
        seen = {}

        for _, r in df.iterrows():
            code = str(r.get("CourseCode10", "")).strip()
            label = code if code else "—"

            # ✅ si label déjà utilisé (CourseCode10 identique), suffixer
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:
                label = f"{label}#{seen[label]}"

            cols[label] = [r.get("Départ", ""), r.get("Arrivée", "")]

        vue2 = pd.DataFrame(cols, index=["Départ", "Arrivée"])

        day_label = f"{d.strftime('%a %d-%m')} ({vacances_label(d)})"
        sheets[day_label] = vue2

        st.markdown(f"#### {day_label}")
        st.dataframe(vue2, use_container_width=True)

    # -------------------------
    # Export Excel multi-onglets
    # -------------------------
    def export_excel_multi_sheets(sheets_dict: dict[str, pd.DataFrame]) -> bytes:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for name, df_sheet in sheets_dict.items():
                df_sheet.to_excel(writer, sheet_name=str(name)[:31])
        return out.getvalue()

    if sheets:
        xlsx = export_excel_multi_sheets(sheets)
        st.download_button(
            "⬇️ Export Excel (Vue 2 — 1 onglet / jour)",
            data=xlsx,
            file_name=f"fiche_vue2_route_{route_id}_week_{week[0].isoformat()}_{regime}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Aucune donnée à exporter pour cette semaine.")

# =========================
# SCRIPT COMPLET — PART 5/5
# (TAB 3 : Comparaison STRICTE GTFS ↔ Excel Région)
# =========================

with tabs[2]:
    st.subheader("Comparaison GTFS ↔ Fiche horaire Région (Excel)")
    st.caption("Comparaison STRICTE Départ–Arrivée (HH:MM). Aucune tolérance. "
               "Les lignes 'SCO&VAC' sont prises en compte à la fois en SCOLAIRE et en VACANCES.")

    # -------------------------
    # Sélection ligne / sens
    # -------------------------
    routes = get_routes(tables)
    chosen_cmp = st.selectbox("Ligne", routes["label"].tolist(), key="t3_route")
    route_id_cmp = routes.loc[routes["label"] == chosen_cmp, "route_id"].iloc[0]

    # direction_id GTFS (si présent)
    trips_cmp = tables["trips.txt"].copy()
    if "direction_id" in trips_cmp.columns:
        dirs = sorted(
            trips_cmp[trips_cmp["route_id"].astype(str) == str(route_id_cmp)]
            ["direction_id"].dropna().astype(str).unique().tolist()
        )
        direction_cmp = st.selectbox("Direction GTFS (optionnel)", ["(toutes)"] + dirs, key="t3_dir")
    else:
        direction_cmp = "(toutes)"
        st.caption("direction_id absent → pas de filtre direction")

    # sens Excel (A/R)
    sens_excel = st.selectbox("Sens Excel (A/R)", ["A", "R"], key="t3_sens")

    st.divider()

    # -------------------------
    # Date / régime
    # -------------------------
    cmp_day = st.date_input("Date à comparer", value=date.today(), key="t3_day")

    detected_regime = "VACANCES" if is_vacances(cmp_day) else "SCOLAIRE"
    st.caption(f"Régime détecté : **{vacances_label(cmp_day)}**")

    cmp_mode = st.radio("Mode", ["📅 Date précise (auto)", "🏷️ Forcer un régime"], key="t3_mode")
    if cmp_mode.startswith("🏷️"):
        regime_cmp = st.selectbox("Régime forcé", ["SCOLAIRE", "VACANCES"], key="t3_regime_forced")
    else:
        regime_cmp = detected_regime

    st.divider()

    # -------------------------
    # Mapping service_id → régime (GTFS)
    # -------------------------
    st.markdown("### Mapping service_id → Régime (GTFS)")
    all_service_ids = sorted(tables["trips.txt"]["service_id"].astype(str).dropna().unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        scolaire_ids = st.multiselect("service_id SCOLAIRE", all_service_ids, key="t3_sco")
    with c2:
        vacances_ids = st.multiselect("service_id VACANCES", all_service_ids, key="t3_vac")

    allowed = set(scolaire_ids if regime_cmp == "SCOLAIRE" else vacances_ids)
    if not allowed:
        st.warning("Aucun service_id sélectionné → tous les services actifs du jour seront utilisés.")
        allowed = None

    st.divider()

    # -------------------------
    # Upload Excel Région
    # -------------------------
    excel_file = st.file_uploader(
        "Fiche horaire Région (Excel retravaillé)",
        type=["xlsx"],
        key="t3_excel"
    )
    if not excel_file:
        st.info("Charge un Excel Région pour continuer.")
        st.stop()

    excel_raw = pd.read_excel(excel_file)
    excel_raw.columns = [normalize_colname(c) for c in excel_raw.columns]

    REQUIRED = {
        "ligne", "sens", "periode", "horaires",
        "lundi", "mardi", "mercredi",
        "jeudi", "vendredi", "samedi", "dimanche"
    }
    missing = REQUIRED - set(excel_raw.columns)
    if missing:
        st.error(f"Colonnes manquantes dans l’Excel Région : {sorted(missing)}")
        st.stop()

    # -------------------------
    # Expand périodes SCO / VAC / SCO&VAC
    # -------------------------
    def period_to_regimes(p: str) -> list[str]:
        p = str(p).lower()
        has_sco = "sco" in p or "scol" in p
        has_vac = "vac" in p
        if has_sco and has_vac:
            return ["SCOLAIRE", "VACANCES"]
        if has_sco:
            return ["SCOLAIRE"]
        if has_vac:
            return ["VACANCES"]
        return []

    rows = []
    for _, r in excel_raw.iterrows():
        regs = period_to_regimes(r["periode"])
        if not regs:
            continue
        for reg in regs:
            rows.append({
                "ligne": str(r["ligne"]).strip(),
                "sens": str(r["sens"]).strip().upper(),
                "regime": reg,
                "horaires": str(r["horaires"]).strip(),
                "lundi": r["lundi"],
                "mardi": r["mardi"],
                "mercredi": r["mercredi"],
                "jeudi": r["jeudi"],
                "vendredi": r["vendredi"],
                "samedi": r["samedi"],
                "dimanche": r["dimanche"],
            })
    excel = pd.DataFrame(rows)
    if excel.empty:
        st.error("Aucune ligne exploitable (période SCO/VAC/SCO&VAC non reconnue).")
        st.stop()

    # -------------------------
    # Filtrage jour (Excel)
    # -------------------------
    weekday_map = {
        0: "lundi", 1: "mardi", 2: "mercredi",
        3: "jeudi", 4: "vendredi",
        5: "samedi", 6: "dimanche",
    }
    day_col = weekday_map[cmp_day.weekday()]

    def is_active_flag(v) -> bool:
        # Excel : 1 = oui, vide = non
        s = str(v).strip()
        return s == "1" or s.lower() == "true" or s.lower() == "oui"

    excel_filt = excel[
        (excel["ligne"] == str(route_id_cmp)) &
        (excel["sens"] == sens_excel) &
        (excel["regime"] == regime_cmp)
    ].copy()

    if day_col not in excel_filt.columns:
        st.error(f"Colonne jour manquante : {day_col}")
        st.stop()

    excel_filt = excel_filt[excel_filt[day_col].apply(is_active_flag)].copy()

    if excel_filt.empty:
        st.warning("Aucune course trouvée dans l’Excel Région pour ce jour / sens / régime.")
        st.stop()

    # -------------------------
    # Paires Région (départ-arrivée)
    # -------------------------
    def parse_pairs_cell(cell: str) -> list[tuple[str, str]]:
        """
        Cellule Excel: "HH:MM-HH:MM, HH:MM-HH:MM, ..."
        Retourne liste [(dep, arr), ...] en HH:MM normalisé.
        """
        out = []
        s = str(cell).strip()
        if not s:
            return out
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            if "-" not in p:
                continue
            dep_raw, arr_raw = p.split("-", 1)
            dep = _normalize_hhmm_token(dep_raw)
            arr = _normalize_hhmm_token(arr_raw)
            if dep and arr:
                out.append((dep, arr))
        return out

    region_pairs = set()
    for _, r in excel_filt.iterrows():
        region_pairs.update(parse_pairs_cell(r["horaires"]))

    if not region_pairs:
        st.error("Aucun couple (Départ-Arrivée) extrait depuis l’Excel Région (colonne 'horaires').")
        st.stop()

    # -------------------------
    # Paires GTFS (départ-arrivée) + CourseCode10 (info)
    # -------------------------
    gtfs_df = trip_dep_arr_for_route_day(
        tables,
        cmp_day,
        str(route_id_cmp),
        direction_cmp,
        allowed,
    ).copy()

    # normaliser HH:MM depuis HH:MM:SS
    gtfs_pairs = set()
    for _, r in gtfs_df.iterrows():
        dep = _normalize_hhmm_token(str(r.get("Départ", ""))[:5])
        arr = _normalize_hhmm_token(str(r.get("Arrivée", ""))[:5])
        if dep and arr:
            gtfs_pairs.add((dep, arr))

    # -------------------------
    # Comparaison stricte
    # -------------------------
    missing_in_region = sorted(list(gtfs_pairs - region_pairs))
    extra_in_region = sorted(list(region_pairs - gtfs_pairs))

    st.divider()
    st.markdown("### Résultat")

    st.write(f"GTFS : **{len(gtfs_pairs)}** couples | Région : **{len(region_pairs)}** couples")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### ❌ Manquants dans la fiche Région")
        st.write(", ".join([f"{d}→{a}" for d, a in missing_in_region]) if missing_in_region else "—")
    with cB:
        st.markdown("#### ❌ En trop dans la fiche Région")
        st.write(", ".join([f"{d}→{a}" for d, a in extra_in_region]) if extra_in_region else "—")

    diff_df = pd.DataFrame(
        [{"type": "missing_in_region", "departure": d, "arrival": a} for d, a in missing_in_region] +
        [{"type": "extra_in_region", "departure": d, "arrival": a} for d, a in extra_in_region]
    )

    st.divider()
    st.subheader("Écarts (exportables)")
    st.dataframe(diff_df, use_container_width=True, hide_index=True)

    if not diff_df.empty:
        st.download_button(
            "⬇️ Export CSV — écarts GTFS vs Région",
            data=diff_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ecarts_gtfs_vs_region_{route_id_cmp}_{cmp_day.isoformat()}_{regime_cmp}.csv",
            mime="text/csv",
        )

    st.divider()
    if st.checkbox("Afficher détail GTFS (inclut CourseCode10)", value=False, key="t3_show_gtfs"):
        st.dataframe(gtfs_df, use_container_width=True, hide_index=True)

    st.divider()
    if st.checkbox("Afficher lignes Excel filtrées (Région)", value=False, key="t3_show_excel"):
        st.dataframe(excel_filt, use_container_width=True, hide_index=True)
