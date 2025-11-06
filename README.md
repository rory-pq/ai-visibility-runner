# AI Visibility Runner

Test brand visibility in AI answers. Pick a prompt template. Fill inputs. Run checks. Log results.

## Quick Start
1. Open the app on Streamlit Cloud.
2. Fill the sidebar inputs.
3. Pick a template ID.
4. Copy the filled prompt.
5. Click **Save log rows** to export `runs.csv`.

## Files
- `app.py` — Streamlit app
- `templates_runtime.csv` — prompt templates (runtime)
- `requirements.txt` — Python deps

## Add Templates
1. Open `templates_runtime.csv`.
2. Add rows. Keep these headers:
   - `template_id,intent_type,prompt_template,variables_required,variables_optional,selection_rules,notes`
3. Commit. Redeploy.

## Inputs (tokens)
Use these names in templates:
`brand, competitor, brand_a, brand_b, industry, product_or_service, target_audience, unique_feature, location, pain_point, budget, timeframe`

## Run Local
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
