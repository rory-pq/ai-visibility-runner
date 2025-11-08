import re, json, time
import pandas as pd
import streamlit as st
import requests
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

# ---------- API CALL HELPERS ----------

def call_chatgpt(prompt: str) -> str:
    """Call OpenAI Chat Completions API. Returns text or error message."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "ChatGPT API not configured. Add OPENAI_API_KEY in Streamlit secrets."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=40)
        resp.raise_for_status()
        out = resp.json()
        return out["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ChatGPT API error: {e}"

def call_perplexity(prompt: str) -> str:
    """Call Perplexity API. Returns text or error message."""
    api_key = st.secrets.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "Perplexity API not configured. Add PERPLEXITY_API_KEY in Streamlit secrets."

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "llama-3.1-sonar-small-online",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=40)
        resp.raise_for_status()
        out = resp.json()
        return out["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Perplexity API error: {e}"

def call_gemini(prompt: str) -> str:
    """Call Gemini API. Returns text or error message."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API not configured. Add GEMINI_API_KEY in Streamlit secrets."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-1.5-flash:generateContent"
        f"?key={api_key}"
    )
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        resp = requests.post(url, json=data, timeout=40)
        resp.raise_for_status()
        out = resp.json()
        # Simple text join
        parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts)
        return text or "Gemini response was empty."
    except Exception as e:
        return f"Gemini API error: {e}"

# ---------- TAB 1: RUN PROMPT ----------

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

        st.markdown("---")
        api_mode = st.checkbox(
            "Run via APIs (ChatGPT, Perplexity, Gemini)",
            value=False,
            help="Off = manual copy/paste. On = call APIs with this prompt."
        )

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

    st.subheader("3) Run and log")

    if api_mode:
        st.caption(
            "API mode is ON. The app calls ChatGPT, Perplexity, and Gemini "
            "for each platform you select."
        )
    else:
        st.caption(
            "API mode is OFF. The app only logs the prompt. "
            "You run searches in AI tools by hand."
        )

    def log_run(platform, prompt, meta, response_text: str):
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
        # Put a short snippet of the response in rank_notes
        snippet = (response_text or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."

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
            "rank_notes": snippet,
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
        "Pick platforms to run/log",
        ["ChatGPT", "Perplexity", "Gemini", "AI Overviews"],
        default=["ChatGPT"],
    )

    meta = {
        "run_id": run_id,
        "template_id": chosen.iloc[0]["template_id"],
        "intent_type": chosen.iloc[0]["intent_type"],
    }

    if st.button("Run now"):
        results = {}
        # Call APIs only for non-Google tools
        for p in platforms:
            response_text = ""
            if api_mode and p != "AI Overviews":
                if p == "ChatGPT":
                    response_text = call_chatgpt(filled)
                elif p == "Perplexity":
                    response_text = call_perplexity(filled)
                elif p == "Gemini":
                    response_text = call_gemini(filled)
            else:
                # Manual mode or AI Overviews
                response_text = ""

            log_run(p, filled, meta, response_text)
            results[p] = response_text

        st.success("Runs logged. See quick previews below and in the Run history tab.")

        # Show previews for API calls
        for p, text in results.items():
            if not text:
                continue
            st.markdown(f"**{p} response preview**")
            st.text_area(
                f"{p} response",
                value=text,
                height=200,
            )

        # Download button for runs
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

# ---------- TAB 2: RUN HISTORY ----------

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

        st.caption("Showing recent runs. Response snippet comes from the model output or notes.")
        display_df = filtered[
            [
                "timestamp",
                "platform",
                "template_id",
                "intent_type",
                "filled_prompt",
                "rank_notes",
            ]
        ].rename(columns={"rank_notes": "response_snippet"})

        st.dataframe(
            display_df,
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
