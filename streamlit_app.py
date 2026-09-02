"""Streamlit UI for the Gemini PDF file-search persona pipeline."""

import json
import re
import tempfile
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st
from google import genai
from google.genai import types

from shaw_contract_persona import persona


MODEL = "gemini-3.1-flash-lite"
PERSONA_IDX = 1
PERSONA_NAME = "sus"
DEFAULT_DEMOGRAPHICS = Path(__file__).with_name("twin_imp_columns.csv")
POLL_INTERVAL = 3
POLL_TIMEOUT = 1200

st.set_page_config(page_title="Gemini PDF Search", page_icon="🔎", layout="wide")


def get_api_key():
    """Read the Gemini key exclusively from Streamlit secrets."""
    try:
        key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None
    return str(key).strip() or None


@st.cache_data(show_spinner=False)
def load_default_demographics():
    if not DEFAULT_DEMOGRAPHICS.exists():
        return None
    return pd.read_csv(DEFAULT_DEMOGRAPHICS)


def read_demographics(uploaded_file):
    if uploaded_file is None:
        frame = load_default_demographics()
        source = DEFAULT_DEMOGRAPHICS.name
    elif uploaded_file.name.lower().endswith(".xlsx"):
        frame = pd.read_excel(uploaded_file)
        source = uploaded_file.name
    else:
        frame = pd.read_csv(uploaded_file)
        source = uploaded_file.name

    if frame is None:
        raise FileNotFoundError(
            f"Default demographics file is missing: {DEFAULT_DEMOGRAPHICS}"
        )
    required = {"Location", "Sex Assigned at Birth"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Demographics file is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(required)).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Demographics file has no complete demographic rows.")
    return frame, source


def read_queries(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8-sig")
    queries = []
    for line in text.splitlines():
        query = line.strip().rstrip(",").strip().strip('"').strip("'")
        if query:
            queries.append(query)
    return queries


def safe_name(filename):
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9_-]+", "_", stem).strip("_") or "document"


def create_system_prompt(demographics):
    row = demographics.sample(n=1).iloc[0]
    location = row["Location"]
    gender = row["Sex Assigned at Birth"]
    prompt = f"""{persona[PERSONA_IDX]}

You are located in: {location}
Gender: {gender}

You have access to a file search tool connected to a single reference document about flooring brands.
Answer ONLY using information retrieved from that document via file search. Do not use any outside
knowledge, prior training data, or general familiarity with flooring brands. If the document does not
mention a brand or attribute relevant to the query, do not include it; do not guess.

Structure your response as:
1. A bullet list of specific BRAND NAMES and COMPANY NAMES found in the document that offer relevant products.
2. For each brand, briefly note attributes using only wording and facts drawn from the document.
3. Cite page numbers or section titles from the document for each brand.

If the document contains no relevant brands for this query, say so explicitly.
If brands are present, order them from most relevant to least relevant.

At the very end of the response, include exactly one machine-readable line:
BRAND_RANKINGS_JSON: [{{"rank": 1, "brand": "Brand Name"}}]
Include every brand/company named in the answer, with sequential integer ranks. If none are found, use:
BRAND_RANKINGS_JSON: []"""
    return prompt, location, gender


def index_pdf(client, pdf_path, original_name, status):
    category = safe_name(original_name)
    status.write(f"Creating a search index for **{original_name}**…")
    print(f"Creating Gemini File Search store for {original_name}", flush=True)
    store = client.file_search_stores.create(
        config={"display_name": f"streamlit_{category}_{int(time.time())}"}
    )
    status.write(f"Uploading **{original_name}** to Gemini Files…")
    uploaded_file = client.files.upload(
        file=str(pdf_path),
        config={"display_name": original_name},
    )
    print(f"Uploaded Gemini file {uploaded_file.name}", flush=True)
    status.write(f"Importing and indexing **{original_name}**…")
    operation = client.file_search_stores.import_file(
        file_search_store_name=store.name,
        file_name=uploaded_file.name,
    )

    started = time.monotonic()
    while not operation.done:
        elapsed = time.monotonic() - started
        if elapsed > POLL_TIMEOUT:
            raise TimeoutError(f"Indexing {original_name} exceeded {POLL_TIMEOUT}s")
        status.write(f"Indexing **{original_name}** — {elapsed:.0f}s elapsed…")
        time.sleep(POLL_INTERVAL)
        operation = client.operations.get(operation)

    if getattr(operation, "error", None):
        raise RuntimeError(f"Indexing {original_name} failed: {operation.error}")
    status.write(f"Indexed **{original_name}** successfully.")
    return store.name


def extract_citations(response):
    citations = []
    try:
        grounding = response.candidates[0].grounding_metadata
        for chunk in grounding.grounding_chunks or []:
            context = chunk.retrieved_context
            if context:
                citations.append({"title": context.title, "text": context.text})
    except (AttributeError, IndexError, TypeError):
        pass
    return citations


def extract_brand_rankings(response_text):
    """Parse the machine-readable brand ranking appended by the model."""
    marker = "BRAND_RANKINGS_JSON:"
    marker_position = response_text.rfind(marker)
    if marker_position < 0:
        return [], response_text.strip()

    answer = response_text[:marker_position].rstrip()
    payload = response_text[marker_position + len(marker):].strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.I)
    try:
        raw_rankings = json.loads(payload)
    except json.JSONDecodeError:
        return [], answer

    rankings = []
    seen = set()
    if isinstance(raw_rankings, list):
        for item in raw_rankings:
            if not isinstance(item, dict):
                continue
            brand = str(item.get("brand", "")).strip()
            if not brand or brand.casefold() in seen:
                continue
            try:
                rank = int(item.get("rank"))
            except (TypeError, ValueError):
                rank = len(rankings) + 1
            seen.add(brand.casefold())
            rankings.append({"rank": rank, "brand": brand})
    rankings.sort(key=lambda item: item["rank"])
    return rankings, answer


def run_query(client, query, store_name, demographics):
    system_prompt, location, gender = create_system_prompt(demographics)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[types.Tool(file_search=types.FileSearch(
                    file_search_store_names=[store_name]
                ))],
            ),
        )
        rankings, clean_response_text = extract_brand_rankings(response.text or "")
        return {
            "query": query,
            "persona_name": PERSONA_NAME,
            "system_prompt": system_prompt,
            "persona_location": location,
            "persona_gender": gender,
            "response_text": clean_response_text,
            "brand_rankings": json.dumps(rankings, ensure_ascii=False),
            "citations": json.dumps(extract_citations(response), ensure_ascii=False),
            "error": None,
        }
    except Exception as error:
        return {
            "query": query,
            "persona_name": PERSONA_NAME,
            "system_prompt": system_prompt,
            "persona_location": location,
            "persona_gender": gender,
            "response_text": None,
            "brand_rankings": None,
            "citations": None,
            "error": str(error),
        }


def to_excel_bytes(frame):
    buffer = BytesIO()
    occurrences, percentages = build_brand_analytics(frame)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="results")
        if not occurrences.empty:
            occurrences.to_excel(writer, index=False, sheet_name="brand_ranks")
            percentages.to_excel(writer, index=False, sheet_name="brand_percentages")
    return buffer.getvalue()


def run_pipeline(client, pdf_files, queries, demographics):
    total = len(pdf_files) * len(queries)
    completed = 0
    results = []
    progress = st.progress(0, text="Starting…")
    status = st.empty()

    with tempfile.TemporaryDirectory(prefix="shaw_rag_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        for pdf_number, uploaded_pdf in enumerate(pdf_files, start=1):
            pdf_path = temporary_path / f"{pdf_number}_{Path(uploaded_pdf.name).name}"
            pdf_path.write_bytes(uploaded_pdf.getvalue())
            category = safe_name(uploaded_pdf.name).removesuffix("_attribute_corpus")
            try:
                store_name = index_pdf(client, pdf_path, uploaded_pdf.name, status)
            except Exception as error:
                for query in queries:
                    results.append({
                        "query": query,
                        "persona_name": PERSONA_NAME,
                        "system_prompt": None,
                        "persona_location": None,
                        "persona_gender": None,
                        "response_text": None,
                        "brand_rankings": None,
                        "citations": None,
                        "error": f"PDF indexing failed: {error}",
                        "category": category,
                        "source_file": uploaded_pdf.name,
                    })
                    completed += 1
                    progress.progress(completed / total, text=f"{completed}/{total} searches")
                continue

            for query in queries:
                status.write(f"Searching **{uploaded_pdf.name}**: {query}")
                result = run_query(client, query, store_name, demographics)
                result["category"] = category
                result["source_file"] = uploaded_pdf.name
                results.append(result)
                completed += 1
                progress.progress(completed / total, text=f"{completed}/{total} searches")

    status.success("Search complete.")
    return pd.DataFrame(results)


def build_brand_analytics(frame):
    """Return occurrence-level ranks and per-brand appearance percentages."""
    occurrences = []
    successful = frame[
        frame["error"].isna() & frame["response_text"].fillna("").str.strip().ne("")
    ]
    for result_index, row in successful.iterrows():
        try:
            rankings = json.loads(row.get("brand_rankings") or "[]")
        except (json.JSONDecodeError, TypeError):
            rankings = []
        seen = set()
        for item in rankings:
            brand = str(item.get("brand", "")).strip()
            key = brand.casefold()
            if not brand or key in seen:
                continue
            seen.add(key)
            occurrences.append({
                "result_id": result_index,
                "brand": brand,
                "rank": int(item.get("rank", len(seen))),
                "category": row.get("category"),
                "query": row.get("query"),
            })

    occurrence_frame = pd.DataFrame(occurrences)
    if occurrence_frame.empty:
        return occurrence_frame, pd.DataFrame()

    total_results = len(successful)
    percentage_frame = (
        occurrence_frame.groupby("brand", as_index=False)["result_id"]
        .nunique()
        .rename(columns={"result_id": "results_appeared"})
    )
    percentage_frame["total_successful_results"] = total_results
    percentage_frame["appearance_percentage"] = (
        percentage_frame["results_appeared"] / total_results * 100
    )
    percentage_frame = percentage_frame.sort_values(
        ["appearance_percentage", "brand"], ascending=[False, True]
    )
    return occurrence_frame, percentage_frame


def show_brand_analytics(frame):
    occurrences, percentages = build_brand_analytics(frame)
    st.subheader("Brand analytics")
    if occurrences.empty:
        st.warning(
            "No machine-readable brand rankings were found. Run the search again "
            "to generate analytics with the updated prompt."
        )
        return

    st.markdown("#### Rank distribution")
    rank_counts = (
        occurrences.groupby(["brand", "rank"], as_index=False)
        .size()
        .rename(columns={"size": "frequency"})
    )
    rank_chart = (
        alt.Chart(rank_counts)
        .mark_bar()
        .encode(
            x=alt.X("rank:O", title="Rank"),
            y=alt.Y("frequency:Q", title="Number of appearances"),
            color=alt.Color("brand:N", title="Brand"),
            tooltip=["brand:N", "rank:O", "frequency:Q"],
        )
        .properties(height=420)
    )
    st.altair_chart(rank_chart, use_container_width=True)

    st.markdown("#### Percentage appearance across successful results")
    percentage_chart = (
        alt.Chart(percentages)
        .mark_bar()
        .encode(
            x=alt.X(
                "appearance_percentage:Q",
                title="Appearance in successful results (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("brand:N", title="Brand", sort="-x"),
            tooltip=[
                "brand:N",
                alt.Tooltip("appearance_percentage:Q", format=".1f"),
                "results_appeared:Q",
                "total_successful_results:Q",
            ],
        )
        .properties(height=max(300, 24 * len(percentages)))
    )
    st.altair_chart(percentage_chart, use_container_width=True)
    display_percentages = percentages.copy()
    display_percentages["appearance_percentage"] = display_percentages[
        "appearance_percentage"
    ].round(2)
    st.dataframe(display_percentages, use_container_width=True, hide_index=True)


st.title("Gemini PDF File Search")
st.caption(f"Fixed model: `{MODEL}` · one non-empty line in the TXT file = one query")

api_key = get_api_key()
if not api_key:
    st.error(
        "Add `GOOGLE_API_KEY` to `.streamlit/secrets.toml` before running the app."
    )

left, right = st.columns(2)
with left:
    pdf_files = st.file_uploader(
        "Reference PDFs", type=["pdf"], accept_multiple_files=True
    )
    query_file = st.file_uploader("Search queries", type=["txt"])
with right:
    demographics_file = st.file_uploader(
        "Demographics override (optional)", type=["csv", "xlsx"]
    )
    if demographics_file is None:
        st.info(f"Using default demographics: `{DEFAULT_DEMOGRAPHICS.name}`")

queries = []
demographics = None
try:
    demographics, demographics_source = read_demographics(demographics_file)
    st.caption(f"Demographics: {demographics_source} · {len(demographics)} usable rows")
except Exception as error:
    st.error(str(error))

if query_file is not None:
    try:
        queries = read_queries(query_file)
        st.caption(f"Loaded {len(queries)} queries from `{query_file.name}`.")
        if queries:
            with st.expander("Preview queries"):
                st.code("\n".join(queries), language="text")
    except UnicodeDecodeError:
        st.error("The query file must be UTF-8 text.")

ready = bool(api_key and pdf_files and queries and demographics is not None)
if st.button("Run file search", type="primary", disabled=not ready):
    st.session_state.pop("results", None)
    st.info("Request received. Starting Gemini File Search…")
    print(
        f"Run requested: {len(pdf_files)} PDF(s), {len(queries)} query/queries",
        flush=True,
    )
    try:
        client = genai.Client(api_key=api_key)
        frame = run_pipeline(client, pdf_files, queries, demographics)
        st.session_state["results"] = frame
    except Exception as error:
        print(f"Unhandled search error: {error!r}", flush=True)
        st.error("The search stopped because of an error:")
        st.exception(error)

if "results" in st.session_state:
    result_frame = st.session_state["results"]
    st.subheader("Results")
    st.dataframe(result_frame, use_container_width=True, hide_index=True)
    show_brand_analytics(result_frame)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "Download Excel results",
        data=to_excel_bytes(result_frame),
        file_name=f"gemini_pdf_search_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.divider()
st.caption(
    "Uploaded PDFs are sent to Gemini File Search and remain subject to your Google API data-retention settings."
)
