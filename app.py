
import re, json, time
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AI Visibility Runner", page_icon="🔎", layout="centered")

TEMPLATE_PATH = Path("templates_runtime.csv")
RUNS_PATH = Path("runs.csv")

# Load templates
df = pd.read_csv(TEMPLATE_PATH)

st.title("AI Search Visibility Runner")

with st.sidebar:
    st.header("Run inputs")
    run_id = st.text_input("Run ID", value=str(int(time.time())))
    intent_pick = st.selectbox("Intent (optional)", ["(any)"] + sorted(df["intent_type"].unique().tolist()))
    inputs = {
        "brand": st.text_input("Your brand"),
        "competitor": st.text_input("Competitor"),
        "brand_a": st.text_input("Brand A"),
        "brand_b": st.text_input("Brand B"),
        "industry": st.text_input("Industry"),
        "product_or_service": st.text_input("Product or Service"),
        "target_audience": st.text_input("Target audience"),
        "unique_feature": st.text_input("Unique feature"),
        "location": st.text_input("Location"),
        "pain_point": st.text_input("Pain point"),
        "budget": st.text_input("Budget"),
        "timeframe": st.text_input("Timeframe"),
    }

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
    st.info("Fill inputs for required fields. Or clear the intent filter.")
    cands = pool.head(5).copy()
    cands["score"] = 0
else:
    cands["score"] = cands.apply(lambda r: score_optional(r, inputs), axis=1)
cands = cands.sort_values(["score"], ascending=False)

st.dataframe(cands[["template_id","intent_type","prompt_template","variables_required","variables_optional","score"]].reset_index(drop=True))

chosen_id = st.text_input("Template ID", value=cands.iloc[0]["template_id"] if not cands.empty else "")
chosen = df[df["template_id"] == chosen_id].head(1)

st.subheader("2) Fill prompt")
def fill_template(tpl: str, vals: dict) -> str:
    text = tpl
    for k, v in vals.items():
        text = text.replace("{"+k+"}", v or "")
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+(in|for|with)\s+(?=[?.!]|$)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

if not chosen.empty:
    template = chosen.iloc[0]["prompt_template"]
    filled = fill_template(template, inputs)
    st.code(template, language="text")
    st.markdown("**Filled prompt**")
    st.text_area("Prompt", value=filled, height=120)
else:
    st.stop()

st.subheader("3) Log the run (no APIs yet)")
def log_run(platform, prompt, meta):
    RUNS_PATH.touch(exist_ok=True)
    cols = ["timestamp","platform","run_id","template_id","intent_type","filled_prompt","inputs_json","top_brands_json","top_domains_json","rank_notes","raw_capture_path"]
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
        "raw_capture_path": ""
    }
    try:
        if RUNS_PATH.stat().st_size == 0:
            pd.DataFrame([row])[cols].to_csv(RUNS_PATH, index=False)
        else:
            pd.DataFrame([row])[cols].to_csv(RUNS_PATH, index=False, mode="a", header=False)
    except FileNotFoundError:
        pd.DataFrame([row])[cols].to_csv(RUNS_PATH, index=False)

platforms = st.multiselect("Pick platforms to log", ["ChatGPT","Perplexity","Gemini","AI Overviews"], default=["ChatGPT"])
meta = {"run_id": run_id, "template_id": chosen.iloc[0]["template_id"], "intent_type": chosen.iloc[0]["intent_type"]}
if st.button("Save log rows"):
    for p in platforms:
        log_run(p, filled, meta)
    st.success("Saved to runs.csv. Download below.")
    st.download_button("Download runs.csv", data=open(RUNS_PATH, "rb").read(), file_name="runs.csv")
