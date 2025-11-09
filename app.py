import re, json, time
import pandas as pd
import streamlit as st
import requests
from urllib.parse import urlparse
from pathlib import Path

st.set_page_config(page_title="AI Visibility Runner", page_icon="🔎", layout="centered")

TEMPLATE_PATH = Path("templates_runtime.csv")
RUNS_PATH = Path("runs.csv")

# Load templates
df = pd.read_csv(TEMPLATE_PATH)

# Auto-generate a run id (hidden from UI)
run_id = str(int(time.time()))

st.title("AI Visibility Runner")

# ---------- API CALL HELPERS ----------

def call_chatgpt(prompt: str) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "ChatGPT API not configured. Add OPENAI_API_KEY in Streamlit secrets."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
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
        "messages": [{"role": "user", "content": prompt}],
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
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API not configured. Add GEMINI_API_KEY in Streamlit secrets."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-1.5-flash:generateContent"
        f"?key={api_key}"
    )
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(url, json=data, timeout=40)
        resp.raise_for_status()
        out = resp.json()
        parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts)
        return text or "Gemini response was empty."
    except Exception as e:
        return f"Gemini API error: {e}"

# ---------- BRAND + DOMAIN HELPERS ----------

def detect_brands_in_text(text: str, inputs: dict) -> list:
    """Use only Your brand and Main competitor for brand hits."""
    if not text:
        return []
    lower_text = text.lower()

    brands = []
    my_brand = (inputs.get("brand") or "").strip()
    if my_brand:
        brands.append({"brand": my_brand, "aliases": []})

    comp = (inputs.get("competitor") or "").strip()
    if comp and not any(b["brand"].lower() == comp.lower() for b in brands):
        brands.append({"brand": comp, "aliases": []})

    hits = []
    for b in brands:
        name = b["brand"]
        aliases = b.get("aliases", [])
        patterns = [name] + aliases
        total = 0
        for pat in patterns:
            pat = pat.strip()
            if not pat:
                continue
            total += len(
                re.findall(r"\b" + re.escape(pat.lower()) + r"\b", lower_text)
            )
        if total > 0:
            hits.append({"brand": name, "hits": int(total)})

    return hits

def extract_domains_from_text(text: str) -> list:
    """Find domains from URLs in plain text."""
    if not text:
        return []
    urls = re.findall(r"https?://[^\s)>\]\"'}]+", text)
    counts = {}
    for u in urls:
        try:
            netloc = urlparse(u).netloc.lower()
        except Exception:
            continue
        if not netloc:
            continue
        if netloc.startswith("www."):
            netloc = netloc[4:]
        counts[netloc] = counts.get(netloc, 0) + 1
    return [{"domain": d, "hits": c} for d, c in counts.items()]

# ---------- TEMPLATE FILL HELPER ----------

def fill_template(tpl: str, vals: dict) -> str:
    text = str(tpl)
    placeholders = re.findall(r"{([^}]+)}", text)
    for key in placeholders:
        value = vals.get(key) or ""
        text = text.replace("{" + key + "}", value)

    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+(in|for|with)\s+(?=[?.!]|$)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

# Tabs: main runner + history
tab_run, tab_history = st.tabs(["Run prompt", "Run history"])

# ---------- TAB 1: RUN PROMPT ----------

with tab_run:
    with st.sidebar:
        st.header("Run inputs")
        st.caption(
            "Fill fields that match this test. "
            "Skip fields that do not matter."
        )

        intent_pick = st.selectbox(
            "Intent filter (optional)",
            ["(any)"] + sorted(df["intent_type"].unique().tolist())
        )

        client_name = st.text_input(
            "Client / project name",
            help="Short label for this brand or engagement."
        )

        inputs = {
            "client": client_name,
            "brand": st.text_input(
                "Your brand",
                help="Client brand or your agency name."
            ),
            "competitor": st.text_input(
                "Main competitor",
                help="Key rival for this run. Leave blank if not needed."
            ),
            "industry": st.text_input(
                "Industry or category",
                help="Example: B2B SaaS, pet food, real estate, travel."
            ),
            "product_or_service": st.text_input(
                "Product or service",
                help="Example: SEO agency, CRM software, flexible packaging."
            ),
            "target_audience": st.text_input(
                "Audience (who or what this is for)",
                help="Use people or things. Example: marketers, food, B2B buyers, pets."
            ),
            "unique_feature": st.text_input(
                "Key feature or angle",
                help="Short phrase. Example: AI reporting, eco materials, same-day shipping."
            ),
            "location": st.text_input(
                "Location",
                help="City, region, or country. Example: New York, US, Europe."
            ),
            "pain_point": st.text_input(
                "Main problem or use case",
                help="Example: reduce ad spend, keep food fresh, cut churn."
            ),
            "budget": st.text_input(
                "Budget limit",
                help="Add only if used. Example: under $500/month, under $5K."
            ),
            "timeframe": st.text_input(
                "Timeframe",
                help="Add only if used. Example: this quarter, 2025, 30 days."
            ),
        }

        inputs["brand_a"] = inputs["brand"]
        inputs["brand_b"] = inputs["competitor"]

        st.markdown("---")
        api_mode = st.checkbox(
            "Run via APIs (ChatGPT, Perplexity, Gemini)",
            value=False,
            help="Off: copy prompts by hand. On: call model APIs from this app."
        )

    st.subheader("1) Pick a template")

    pool = df if intent_pick == "(any)" else df[df["intent_type"] == intent_pick]

    def has_required(row, vals):
        req = set(str(row["variables_required"]).split("|")) if pd.notna(row["variables_required"]) else set()
        return all(vals.get(k) for k in req)

    def score_optional(row, vals):
        opt = set(str(row["variables_optional"]).split("|")) if pd.notna(row["variables_optional"]) else set()
        return sum(1 for k in opt if vals.get(k))

    cands = pool[pool.apply(lambda r: has_required(r, inputs), axis=1)].copy()

    if cands.empty:
        st.info("No perfect matches yet. Add more inputs or clear the intent filter.")
        cands = pool.head(5).copy()
        cands["score"] = 0
    else:
        cands["score"] = cands.apply(lambda r: score_optional(r, inputs), axis=1)

    if cands.empty:
        st.warning("No templates found. Add more templates or adjust filters.")
        st.stop()

    cands = cands.sort_values(["score"], ascending=False)

    st.dataframe(
        cands[["template_id", "intent_type", "prompt_template", "notes"]].reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )

    st.caption("Pick a template ID from the table. Higher rows match more of your inputs.")

    template_ids = cands["template_id"].tolist()
    chosen_id = st.selectbox("Template ID to use", template_ids)

    chosen = df[df["template_id"] == chosen_id].head(1)

st.subheader("2) Fill prompt")

if not chosen.empty:
    template = chosen.iloc[0]["prompt_template"]
    filled = fill_template(template, inputs)

    # Optional debug: show unfilled vars
    missing_vars = re.findall(r"{([^}]+)}", filled)
    if missing_vars:
        st.warning(
            "These variables are still in the prompt and have no value: "
            + ", ".join(sorted(set(missing_vars)))
        )

    st.markdown("**Template**")
    st.code(template, language="text")
    st.markdown("**Filled prompt**")
    st.code(filled, language="text")
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

    pasted_sources = st.text_area(
        "Optional: paste source URLs or citations here before you hit Run now.",
        help="The app will extract domains from this text for this run.",
        height=100,
    )

    def log_run(platform, prompt, meta, response_text: str, extra_sources: str):
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

        brand_hits = detect_brands_in_text(response_text, inputs)
        combined_for_domains = ((response_text or "") + "\n" + (extra_sources or "")).strip()
        domain_hits = extract_domains_from_text(combined_for_domains)

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
            "top_brands_json": json.dumps(brand_hits),
            "top_domains_json": json.dumps(domain_hits),
            "rank_notes": snippet,
            "raw_capture_path": (extra_sources or ""),
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
                response_text = ""

            log_run(p, filled, meta, response_text, pasted_sources)
            results[p] = response_text

        st.success("Runs logged. See previews below and in the Run history tab.")

        for p, text in results.items():
            if not text:
                continue
            st.markdown(f"**{p} response preview**")
            st.text_area(
                f"{p} response",
                value=text,
                height=200,
                disabled=True,
            )

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

        def extract_client(val: str) -> str:
            try:
                data = json.loads(val or "{}")
                return (data.get("client") or "").strip()
            except Exception:
                return ""

        runs_df["client"] = runs_df.get("inputs_json", "").apply(extract_client)

        platforms_all = ["All"] + sorted(runs_df["platform"].dropna().unique().tolist())
        intents_all = ["All"] + sorted(runs_df["intent_type"].dropna().unique().tolist())
        clients_all = ["All"] + sorted(
            [c for c in runs_df["client"].dropna().unique().tolist() if c]
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            platform_filter = st.selectbox("Filter by platform", platforms_all)
        with col2:
            intent_filter = st.selectbox("Filter by intent", intents_all)
        with col3:
            client_filter = st.selectbox("Filter by client", clients_all)

        filtered = runs_df.copy()
        if platform_filter != "All":
            filtered = filtered[filtered["platform"] == platform_filter]
        if intent_filter != "All":
            filtered = filtered[filtered["intent_type"] == intent_filter]
        if client_filter != "All":
            filtered = filtered[filtered["client"] == client_filter]

        if "timestamp" in filtered.columns:
            filtered = filtered.sort_values("timestamp", ascending=False)

        st.caption("Run log with response snippet.")
        display_df = filtered[
            [
                "timestamp",
                "client",
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

        st.markdown("### Brand counts by platform")

        brand_rows = []
        for _, row in runs_df.iterrows():
            platform = row.get("platform", "")
            intent = row.get("intent_type", "")
            try:
                brands_json = json.loads(row.get("top_brands_json", "[]") or "[]")
            except Exception:
                brands_json = []
            for item in brands_json:
                brand_name = item.get("brand")
                hits = item.get("hits", 0)
                if brand_name and hits:
                    brand_rows.append(
                        {
                            "brand": brand_name,
                            "platform": platform,
                            "intent_type": intent,
                            "hits": int(hits),
                        }
                    )

        if brand_rows:
            brand_df = pd.DataFrame(brand_rows)

            latest_inputs = {}
            try:
                latest_row = runs_df.sort_values("timestamp").iloc[-1]
                latest_inputs = json.loads(latest_row.get("inputs_json", "{}") or "{}")
            except Exception:
                latest_inputs = {}

            focus_my_brand = (latest_inputs.get("brand") or "").strip()
            focus_comp = (latest_inputs.get("competitor") or "").strip()

            focus_toggle = st.checkbox(
                "Show only your brand and main competitor "
                "(from the most recent run)",
                value=False,
                help=(
                    f"Current focus: '{focus_my_brand or 'n/a'}' "
                    f"vs '{focus_comp or 'n/a'}'."
                ),
            )

            intent_chart_options = ["All"] + sorted(
                brand_df["intent_type"].dropna().unique().tolist()
            )
            intent_chart_filter = st.selectbox(
                "Chart intent filter", intent_chart_options, index=0
            )

            brand_chart_df = brand_df.copy()
            if intent_chart_filter != "All":
                brand_chart_df = brand_chart_df[
                    brand_chart_df["intent_type"] == intent_chart_filter
                ]

            if focus_toggle and (focus_my_brand or focus_comp):
                keep = set()
                if focus_my_brand:
                    keep.add(focus_my_brand)
                if focus_comp:
                    keep.add(focus_comp)
                brand_chart_df = brand_chart_df[
                    brand_chart_df["brand"].isin(list(keep))
                ]

            brand_summary = (
                brand_chart_df.groupby(["brand", "platform"], as_index=False)["hits"].sum()
            )

            st.dataframe(
                brand_summary.sort_values(["hits"], ascending=False).reset_index(
                    drop=True
                ),
                hide_index=True,
                use_container_width=True,
            )

            if not brand_chart_df.empty:
                chart_pivot = (
                    brand_chart_df.groupby(["platform", "brand"], as_index=False)["hits"]
                    .sum()
                    .pivot(index="platform", columns="brand", values="hits")
                    .fillna(0)
                )
                st.markdown("**Brand share by platform (hits)**")
                st.bar_chart(chart_pivot)
        else:
            st.info(
                "No brand hits detected yet. "
                "Set 'Your brand' and 'Main competitor' in runs, then test again."
            )

        st.markdown("### Domain counts by platform")
        domain_rows = []
        for _, row in runs_df.iterrows():
            platform = row.get("platform", "")
            try:
                dom_json = json.loads(row.get("top_domains_json", "[]") or "[]")
            except Exception:
                dom_json = []
            for item in dom_json:
                dom = item.get("domain")
                hits = item.get("hits", 0)
                if dom and hits:
                    domain_rows.append(
                        {"domain": dom, "platform": platform, "hits": int(hits)}
                    )

        if domain_rows:
            dom_df = pd.DataFrame(domain_rows)
            dom_summary = (
                dom_df.groupby(["domain", "platform"], as_index=False)["hits"].sum()
            )
            st.dataframe(
                dom_summary.sort_values(["hits"], ascending=False).reset_index(
                    drop=True
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info(
                "No domains detected yet. Perplexity and other tools must return URLs first "
                "or paste sources into runs."
            )

        st.caption("Need full data? Download the CSV below.")
        with open(RUNS_PATH, "rb") as f:
            st.download_button(
                "Download full runs.csv",
                data=f.read(),
                file_name="runs.csv",
                mime="text/csv",
            )
