import os
import io
import zipfile
import sqlite3
from datetime import datetime
import re

import pandas as pd
import requests
import streamlit as st
from rapidfuzz import process, fuzz

# =========================
# Config
# =========================
st.set_page_config(page_title="DPD Smart Lookup (Health Canada)", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "dpd.db")
META_PATH = os.path.join(DATA_DIR, "dpd_meta.txt")

# NOTE:
# Health Canada provides official DPD data extracts (recommended over scraping).
# This URL may change; if it does, open the DPD data extract page and update it.
DPD_ALLFILES_ZIP_URL = "https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database/dpd-data-extracts.html"

# The actual download link for "allfiles.zip" is hosted under health-products.canada.ca.
# We resolve it dynamically by letting the user paste it (safest), BUT we also include
# a common known endpoint below as a convenience fallback.
COMMON_ALLFILES_ZIP_FALLBACK = "https://health-products.canada.ca/dpd-bdpp/download/allfiles.zip"


# =========================
# DB helpers
# =========================
def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def write_meta(msg: str):
    with open(META_PATH, "w", encoding="utf-8") as f:
        f.write(msg)

def read_meta() -> str:
    if not os.path.exists(META_PATH):
        return ""
    with open(META_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

def to_sql(df: pd.DataFrame, table: str):
    with conn() as c:
        df.to_sql(table, c, if_exists="replace", index=False)

def exec_sql(sql: str):
    with conn() as c:
        c.executescript(sql)
        c.commit()

def has_tables() -> bool:
    if not os.path.exists(DB_PATH):
        return False
    with conn() as c:
        cur = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('drug','ingred','package','form','route','status','schedule','comp')"
        )
        rows = cur.fetchall()
    return len(rows) >= 3


# =========================
# Download + ingest
# =========================
def download_zip(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def read_csv_from_zip(z: zipfile.ZipFile, filename: str) -> pd.DataFrame:
    with z.open(filename) as f:
        raw = f.read()
    # DPD extract uses UTF-8, quoted CSV :contentReference[oaicite:3]{index=3}
    bio = io.BytesIO(raw)
    df = pd.read_csv(bio, dtype=str, encoding="utf-8")
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def build_dpd_db(allfiles_zip_bytes: bytes):
    z = zipfile.ZipFile(io.BytesIO(allfiles_zip_bytes))

    expected = ["drug.txt", "ingred.txt", "package.txt", "form.txt", "route.txt", "status.txt", "schedule.txt", "comp.txt"]
    available = set(z.namelist())

    missing = [f for f in expected if f not in available]
    if missing:
        raise RuntimeError(f"ZIP missing files: {missing}. Found: {sorted(list(available))[:20]} ...")

    drug = read_csv_from_zip(z, "drug.txt")
    ingred = read_csv_from_zip(z, "ingred.txt")
    package = read_csv_from_zip(z, "package.txt")
    form = read_csv_from_zip(z, "form.txt")
    route = read_csv_from_zip(z, "route.txt")
    status = read_csv_from_zip(z, "status.txt")
    schedule = read_csv_from_zip(z, "schedule.txt")
    comp = read_csv_from_zip(z, "comp.txt")

    # Store raw tables
    to_sql(drug, "drug")
    to_sql(ingred, "ingred")
    to_sql(package, "package")
    to_sql(form, "form")
    to_sql(route, "route")
    to_sql(status, "status")
    to_sql(schedule, "schedule")
    to_sql(comp, "comp")

    # Helpful indexes for speed
    exec_sql("""
    CREATE INDEX IF NOT EXISTS idx_drug_din ON drug(DRUG_IDENTIFICATION_NUMBER);
    CREATE INDEX IF NOT EXISTS idx_drug_brand ON drug(BRAND_NAME);
    CREATE INDEX IF NOT EXISTS idx_ingred_ing ON ingred(INGREDIENT);
    CREATE INDEX IF NOT EXISTS idx_pkg_code ON package(DRUG_CODE);
    CREATE INDEX IF NOT EXISTS idx_form_code ON form(DRUG_CODE);
    CREATE INDEX IF NOT EXISTS idx_route_code ON route(DRUG_CODE);
    CREATE INDEX IF NOT EXISTS idx_status_code ON status(DRUG_CODE);
    """)

    write_meta(f"Built: {datetime.now().isoformat(timespec='seconds')} | Source: allfiles.zip (marketed products)")

def sql_df(q: str, params=None) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query(q, c, params=params or {})


# =========================
# Search + “agent”
# =========================
def search_by_din(din: str) -> pd.DataFrame:
    din = din.strip()
    return sql_df(
        """
        SELECT DRUG_CODE, DRUG_IDENTIFICATION_NUMBER, BRAND_NAME, DESCRIPTOR, CLASS, PRODUCT_CATEGORIZATION, LAST_UPDATE_DATE
        FROM drug
        WHERE DRUG_IDENTIFICATION_NUMBER = :din
        ORDER BY BRAND_NAME
        """,
        {"din": din},
    )

def search_by_text(term: str, limit: int = 50) -> pd.DataFrame:
    t = term.strip()
    # Simple LIKE search across brand + ingredient (fast enough with indexes + limit)
    return sql_df(
        f"""
        WITH hits AS (
            SELECT DRUG_CODE, DRUG_IDENTIFICATION_NUMBER, BRAND_NAME, DESCRIPTOR
            FROM drug
            WHERE BRAND_NAME LIKE :q OR DESCRIPTOR LIKE :q
            LIMIT {limit}
        )
        SELECT * FROM hits
        """,
        {"q": f"%{t}%"},
    )

def ingredient_hits(term: str, limit: int = 50) -> pd.DataFrame:
    t = term.strip()
    return sql_df(
        f"""
        SELECT DISTINCT d.DRUG_CODE, d.DRUG_IDENTIFICATION_NUMBER, d.BRAND_NAME, i.INGREDIENT, i.STRENGTH, i.STRENGTH_UNIT
        FROM ingred i
        JOIN drug d ON d.DRUG_CODE = i.DRUG_CODE
        WHERE i.INGREDIENT LIKE :q
        LIMIT {limit}
        """,
        {"q": f"%{t}%"},
    )

def get_drug_detail(drug_code: str) -> dict:
    d = sql_df("SELECT * FROM drug WHERE DRUG_CODE = :c", {"c": drug_code})
    i = sql_df("SELECT INGREDIENT, STRENGTH, STRENGTH_UNIT, NOTES FROM ingred WHERE DRUG_CODE = :c", {"c": drug_code})
    p = sql_df("SELECT PACKAGE_SIZE, PACKAGE_SIZE_UNIT, PACKAGE_TYPE, PRODUCT_INFORMATION FROM package WHERE DRUG_CODE = :c", {"c": drug_code})
    f = sql_df("SELECT PHARMACEUTICAL_FORM FROM form WHERE DRUG_CODE = :c", {"c": drug_code})
    r = sql_df("SELECT ROUTE_OF_ADMINISTRATION FROM route WHERE DRUG_CODE = :c", {"c": drug_code})
    s = sql_df("SELECT STATUS, HISTORY_DATE FROM status WHERE DRUG_CODE = :c ORDER BY HISTORY_DATE DESC", {"c": drug_code})
    sch = sql_df("SELECT SCHEDULE FROM schedule WHERE DRUG_CODE = :c", {"c": drug_code})

    return {
        "drug": d,
        "ingredients": i,
        "packaging": p,
        "form": f,
        "route": r,
        "status": s,
        "schedule": sch,
    }

def build_fuzzy_choices() -> tuple[list[str], list[str]]:
    # Build cached lists for fuzzy matching
    drugs = sql_df("SELECT BRAND_NAME FROM drug WHERE BRAND_NAME IS NOT NULL AND BRAND_NAME <> ''")
    ingreds = sql_df("SELECT DISTINCT INGREDIENT FROM ingred WHERE INGREDIENT IS NOT NULL AND INGREDIENT <> ''")
    brand_list = sorted(set(drugs["BRAND_NAME"].astype(str).tolist()))
    ing_list = sorted(set(ingreds["INGREDIENT"].astype(str).tolist()))
    return brand_list, ing_list

@st.cache_data(show_spinner=False)
def cached_choices():
    return build_fuzzy_choices()

def agent_route(query: str) -> dict:
    """
    “Agent” behavior:
    - If query contains 8-digit DIN => DIN search
    - Else fuzzy match brand and ingredient to propose best intent
    """
    q = query.strip()
    din_match = re.search(r"\b(\d{8})\b", q)
    if din_match:
        return {"intent": "din", "din": din_match.group(1)}

    brands, ingreds = cached_choices()

    # fuzzy best guess brand vs ingredient
    b = process.extractOne(q, brands, scorer=fuzz.WRatio) if brands else None
    i = process.extractOne(q, ingreds, scorer=fuzz.WRatio) if ingreds else None

    # pick higher confidence
    best = None
    if b and i:
        best = ("brand", b) if b[1] >= i[1] else ("ingredient", i)
    elif b:
        best = ("brand", b)
    elif i:
        best = ("ingredient", i)

    if not best:
        return {"intent": "none"}

    kind, (match, score, _) = best
    return {"intent": kind, "match": match, "score": score}


# =========================
# UI
# =========================
st.title("DPD Smart Lookup + AI Agent (Health Canada)")
st.caption("Uses official DPD data extract (recommended). No scraping required.")

with st.sidebar:
    st.markdown("### Build / Refresh database")
    st.write("Status:", "✅ Ready" if has_tables() else "❌ Not built")
    meta = read_meta()
    if meta:
        st.code(meta)

    st.markdown("**Download source**")
    st.write("DPD extract page (reference):")
    st.caption(DPD_ALLFILES_ZIP_URL)

    dl_url = st.text_input(
        "allfiles.zip direct URL (optional)",
        value="",
        placeholder=COMMON_ALLFILES_ZIP_FALLBACK,
        help="If blank, we use a common fallback URL. If Health Canada changes it, paste the updated direct link here.",
    )

    if st.button("Download + Build DB", type="primary", use_container_width=True):
        try:
            with st.spinner("Downloading ZIP..."):
                url = dl_url.strip() or COMMON_ALLFILES_ZIP_FALLBACK
                zbytes = download_zip(url)
            with st.spinner("Building SQLite DB..."):
                build_dpd_db(zbytes)
            st.success("DB built.")
            st.rerun()
        except Exception as e:
            st.error(f"Build failed: {e}")

    st.divider()
    st.markdown("### Export")
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            st.download_button("Download dpd.db", f.read(), file_name="dpd.db", mime="application/octet-stream")


if not has_tables():
    st.warning("DB not built yet. Use the sidebar button to download + build from the official DPD extract.")
    st.stop()

tab_search, tab_agent = st.tabs(["🔎 Search", "🤖 AI Agent"])

with tab_search:
    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("Search DIN / brand / ingredient", value="")
    with c2:
        mode = st.selectbox("Mode", ["Auto", "DIN", "Brand/Descriptor", "Ingredient"])

    results = pd.DataFrame()
    if q.strip():
        if mode == "DIN":
            results = search_by_din(q)
        elif mode == "Ingredient":
            results = ingredient_hits(q)
        elif mode == "Brand/Descriptor":
            results = search_by_text(q)
        else:
            route = agent_route(q)
            if route["intent"] == "din":
                results = search_by_din(route["din"])
            elif route["intent"] == "ingredient":
                results = ingredient_hits(route["match"])
            elif route["intent"] == "brand":
                results = search_by_text(route["match"])
            else:
                results = search_by_text(q)

    st.subheader("Results")
    st.dataframe(results, use_container_width=True, hide_index=True)

    if not results.empty and "DRUG_CODE" in results.columns:
        sel = st.selectbox("Select DRUG_CODE to view details", results["DRUG_CODE"].astype(str).tolist())
        detail = get_drug_detail(sel)

        st.subheader("Drug")
        st.dataframe(detail["drug"], use_container_width=True, hide_index=True)

        cA, cB = st.columns(2)
        with cA:
            st.subheader("Ingredients")
            st.dataframe(detail["ingredients"], use_container_width=True, hide_index=True)
            st.subheader("Form / Route")
            st.dataframe(detail["form"], use_container_width=True, hide_index=True)
            st.dataframe(detail["route"], use_container_width=True, hide_index=True)
        with cB:
            st.subheader("Packaging")
            st.dataframe(detail["packaging"], use_container_width=True, hide_index=True)
            st.subheader("Status / Schedule")
            st.dataframe(detail["status"], use_container_width=True, hide_index=True)
            st.dataframe(detail["schedule"], use_container_width=True, hide_index=True)

        # quick export of chosen drug_code bundle
        out = detail["drug"].copy()
        out.to_csv(index=False)
        st.download_button(
            "Export selected drug (CSV)",
            data=detail["drug"].to_csv(index=False).encode("utf-8"),
            file_name=f"dpd_drug_{sel}.csv",
            mime="text/csv",
        )

with tab_agent:
    st.subheader("AI Agent (query router + fuzzy matching)")
    st.caption("Type messy queries — agent will decide whether to search DIN, brand, or ingredient.")

    user_q = st.text_input("Ask the agent", placeholder="e.g. DIN 02212345 or metformin 500 tab or advil liqui-gels")
    if user_q.strip():
        route = agent_route(user_q)
        st.write("Agent decision:", route)

        if route["intent"] == "din":
            res = search_by_din(route["din"])
        elif route["intent"] == "brand":
            res = search_by_text(route["match"])
        elif route["intent"] == "ingredient":
            res = ingredient_hits(route["match"])
        else:
            res = search_by_text(user_q)

        st.dataframe(res, use_container_width=True, hide_index=True)

        if not res.empty and "DRUG_CODE" in res.columns:
            top = str(res.iloc[0]["DRUG_CODE"])
            detail = get_drug_detail(top)

            # “agent summary”
            drow = detail["drug"].iloc[0].to_dict() if not detail["drug"].empty else {}
            brand = drow.get("BRAND_NAME", "")
            din = drow.get("DRUG_IDENTIFICATION_NUMBER", "")
            descriptor = drow.get("DESCRIPTOR", "")

            ings = []
            if not detail["ingredients"].empty:
                for _, r in detail["ingredients"].iterrows():
                    ing = str(r.get("INGREDIENT", "")).strip()
                    stg = str(r.get("STRENGTH", "")).strip()
                    unit = str(r.get("STRENGTH_UNIT", "")).strip()
                    if ing:
                        ings.append(f"{ing} {stg} {unit}".strip())

            st.markdown("### Agent summary (auto-generated)")
            st.write(f"**Top match:** {brand} (DIN {din}) {descriptor}".strip())
            if ings:
                st.write("**Ingredients:**", "; ".join(ings[:8]))

            st.info("If you want a true LLM assistant later, we can add it on top of this DB (RAG).")
