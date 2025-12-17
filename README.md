# xai-credit-research
Masters thesis research on explainable credit decisioning with RAG, ontology mapping, and ASP-based compliance checks with counterfactual analysis

## Repository layout
- `notebooks/`: primary research notebooks (active: `xai-credit-research-v2.ipynb`)
- `data/`: UCI German Credit dataset files
- `models/` (generated): trained pipelines, metrics, and plots
- `regulatory/`: regulatory text snippets (GDPR, Basel III, MiFID II)
- `asp/`: ASP rules (clingo) for compliance validation
- `src/`: helper modules (RAG, ASP validation, etc.)

## Setup (local)
### 1) Create and activate a virtual environment
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies
```sh
pip install -r requirements.txt
```

### 3) Set OpenAI API key (for LLM calls)
Temporary (current shell session):
```sh
export OPENAI_API_KEY="your_api_key_here"
```

Persist for the venv:
```sh
echo 'export OPENAI_API_KEY="your_api_key_here"' >> venv/bin/activate
source venv/bin/activate
```

### 4) Run the notebook
```sh
jupyter notebook
```
Open `notebooks/xai-credit-research-v2.ipynb` and run cells top-to-bottom.

## Notes
- The model pipeline expects the German Credit CSV at `data/statlog-german-credit-data/german_credit_data.csv`.
- If you use local LLMs instead of OpenAI, update the provider/model in the RAG cells and ensure `transformers` is installed.
