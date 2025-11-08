import re, json, time
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AI Visibility Runner", page_icon="🔎", layout="centered")

TEMPLATE_PATH = Path("templates_runtime.csv")
RUNS_PATH = Path("runs.csv")

# Load templates
df = pd.read_csv(TEMPLATE_PATH)

# Auto-generate a run id (hidden from UI)
run_id = str(int(time.time()))

st.title("AI Search Visibility Runner")

# Tabs: main runner + history
tab_run, tab_history = st.tabs(["Run prompt", "Run history"])

# -----------------------
# TAB 1: RUN PROMPT
# -----------------------
with tab_run:
    with st.sidebar:
        st.header("Run inputs")
        st.caption(
            "Fill only fields that match this test. "
            "The tool picks templates that use your inputs."
        )

        intent_pick = st.selectbox(
            "Intent (optional filter)",
            ["(any)"] + sorted(df["intent_type"].unique().tolist())
        )

        # Core inputs
        inputs = {
            "brand": st.text_input("Your brand"),
            "competitor": st.text_input("Main competitor (optional)"),
            "industry": st.text_input("Industry"),
            "product_or_service": st.text_input("Product or Service"),
            "target_audience": st.text_input("Target audience"),
            "unique_feature": st.text_input("Unique feature"),
            "location": st.text_input("Location"),
            "pain_point": st.text_input("Pain point"),
            "budget": st.text_input("Budget"),
            "timeframe": st.text_input("Timeframe"),
        }

        # Map brand fields used in some templates
        inputs["brand_a"] = inputs["brand"]
        inputs["brand_b"] = inputs["competitor"]

    st.subheader("1) Pick a template")

    pool = df if intent_pick == "(any)" else df[df["intent_type"] == intent_pick]

    def has_required(row, vals):
        req = set(str(row["variables_required"]).split("|")) if pd.notna(row["variables_required"]) else set()
        return all(vals.get(k) for k in req)

    def score_optional(row, vals):
        opt = set(str(row["variables_optional"]).split("|")) if pd.notna(row["variables_optional"]) else set()
        return sum(1 for k in opt if vals.get(k))

    # Find candidates that have all required variables
    cands = pool[pool.apply(lambda r: has_required(r, inputs), axis=1)].copy()

    if cands.empty:
        st.info("No perfect matches yet. Fill more inputs or clear the intent filter.")
        # Show a small sample to help the user see options
        cands = pool.head(5).copy()
        cands["score"] = 0
    else:
        cands["score"] = cands.apply(lambda r: score_optional(r, inputs), axis=1)

    if cands.empty:
        st.warning("No templates found. Add more templates or adjust filters.")
        st.stop()

    cands = cands.sort_values(["score"], ascending=False)

    # Show only user-friendly columns
    st.dataframe(
        cands[["template_id", "intent_type", "prompt_template", "notes"]].reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )

    st.caption("Pick a template ID from the table. Higher rows match more of your inputs.")

    # Friendlier template picker: dropdown from candidates
    template_ids = cands["template_id"].tolist()
    chosen_id = st.selectbox("Template ID to use", template_ids)

    chosen = df[df["template_id"] == chosen_id].head(1)

    st.subheader("2) Fill prompt")

    def fill_template(tpl: str, vals: dict) -> str:
        text = tpl
        for k, v in vals.items():
            text = text.replace("{" + k + "}", v or "")
        # Clean extra spaces
        text = re.sub(r"\s{2,}", " ", text).strip()
        # Drop hanging prepositions before punctuation/end
        text = re.sub(r"\s+(in|for|with)\s+(?=[?.!]|$)", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    if not chosen.empty:
        template = chosen.iloc[0]["prompt_template"]
        filled = fill_template(template, inputs)
        st.markdown("**Template**")
        st.code(template, language="text")
        st.markdown("**Filled prompt**")
        st.text_area("Prompt", value=filled, height=140)
    else:
        st.warning("No template selected.")
        st.stop()

    st.subheader("3) Log the run (no APIs yet)")
    st.caption(
        "This only logs the prompt. "
        "You still run the search on ChatGPT, Perplexity, Gemini, or Google by hand."
    )

    def log_run(platform, prompt, meta):
        RUNS_PATH.touch(exist_ok=True)
        cols = [
            "timestamp",
            "platform",
            "run_id",
            "template_id",
            "intent_type",
            "filled_prompt",
            "inputs_json",
            "top_brands_json",
            "top_domains_json",
            "rank_notes",
            "raw_capture_path",
        ]
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform,
            "run_id": meta["run_id"],
            "template_id": meta["template_id"],
            "intent_type": meta["intent_type"],
            "filled_prompt": prompt,
            "inputs_json": json.dumps(inputs),
            "top_brands_json": json.dumps([]),
            "top_domains_json": json.dumps([]),
            "rank_notes": "",
            "raw_capture_path": "",
        }
        try:
            if RUNS_PATH.stat().st_size == 0:
                pd.DataFrame([row])[cols].to_csv(RUNS_PATH, index=False)
            else:
                pd.DataFrame([row])[cols].to_csv(
                    RUNS_PATH, index=False, mode="a", header=False
                )
        except FileNotFoundError:
            pd.DataFrame([row])[cols].to_csv(RUNS_PATH, index=False)

    platforms = st.multiselect(
        "Pick platforms to log",
        ["ChatGPT", "Perplexity", "Gemini", "AI Overviews"],
        default=["ChatGPT"],
    )

    meta = {
        "run_id": run_id,
        "template_id": chosen.iloc[0]["template_id"],
        "intent_type": chosen.iloc[0]["intent_type"],
    }

    if st.button("Save log rows"):
        for p in platforms:
            log_run(p, filled, meta)
        st.success("Saved to runs.csv. Download below.")
        try:
            with open(RUNS_PATH, "rb") as f:
                st.download_button(
                    "Download runs.csv",
                    data=f.read(),
                    file_name="runs.csv",
                    mime="text/csv",
                )
        except FileNotFoundError:
            st.info("No runs file yet. Run at least one log first.")

# -----------------------
# TAB 2: RUN HISTORY
# -----------------------
with tab_history:
    st.subheader("Previous runs")

    if not RUNS_PATH.exists() or RUNS_PATH.stat().st_size == 0:
        st.info("No runs yet. Log at least one run in the **Run prompt** tab.")
    else:
        runs_df = pd.read_csv(RUNS_PATH)

        # Basic filters
        platforms = ["All"] + sorted(runs_df["platform"].dropna().unique().tolist())
        intents = ["All"] + sorted(runs_df["intent_type"].dropna().unique().tolist())

        col1, col2 = st.columns(2)
        with col1:
            platform_filter = st.selectbox("Filter by platform", platforms)
        with col2:
            intent_filter = st.selectbox("Filter by intent", intents)

        filtered = runs_df.copy()
        if platform_filter != "All":
            filtered = filtered[filtered["platform"] == platform_filter]
        if intent_filter != "All":
            filtered = filtered[filtered["intent_type"] == intent_filter]

        # Show newest first
        if "timestamp" in filtered.columns:
            filtered = filtered.sort_values("timestamp", ascending=False)

        st.caption("Showing recent runs. Use filters to narrow the view.")
        st.dataframe(
            filtered[
                [
                    "timestamp",
                    "platform",
                    "template_id",
                    "intent_type",
                    "filled_prompt",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.caption("Need full data? Download the CSV below.")
        with open(RUNS_PATH, "rb") as f:
            st.download_button(
                "Download full runs.csv",
                data=f.read(),
                file_name="runs.csv",
                mime="text/csv",
            )
