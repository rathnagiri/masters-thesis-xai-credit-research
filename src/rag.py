"""
Lightweight RAG utilities for regulatory-aware credit explanations.

Features:
- Loads a small regulatory corpus from regulatory/*.txt and builds a FAISS index.
- Retrieves top-k snippets for a query using sentence-transformers embeddings.
- Builds prompt scaffolds that blend model outputs, SHAP features, ontology labels, and retrieved text.
- LLM client wrapper that can use either a local transformers model or OpenAI chat models.

Dependencies (optional and lazily imported):
- sentence-transformers, faiss-cpu for indexing/retrieval
- transformers for local text-generation (set provider=\"local\")
- openai for hosted LLMs (set provider=\"openai\")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys


# ---------- Ontology loading + explanation-only binning ----------

def load_ontology_mapping(
    path: str = "ontology/mapping-extended.json",
    fallback_path: str = "ontology/mapping.json",
) -> Dict[str, dict]:
    """
    Load ontology mapping; falls back to the legacy mapping.json.
    Backward compatible: if a field is just a string, treat it as {'ontology_term': <string>}.
    """
    # Try multiple candidates for robustness (cwd, repo root, repo_root/notebooks).
    candidates: List[Path] = []
    repo_root = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
    for rel in (path, fallback_path):
        candidates.append(Path(rel))  # cwd-relative
        candidates.append(repo_root / rel)
        candidates.append(repo_root / "notebooks" / rel)

    chosen: Optional[Path] = None
    for c in candidates:
        if c.exists():
            chosen = c
            break
    if chosen is None:
        raise FileNotFoundError(f"Ontology mapping not found. Tried: {candidates}")

    mapping_raw = json.loads(chosen.read_text())
    mapping: Dict[str, dict] = {}
    for k, v in mapping_raw.items():
        if isinstance(v, str):
            mapping[k] = {"ontology_term": v, "notes": "Legacy entry from mapping.json"}
        else:
            mapping[k] = v
    return mapping


def _bin_numeric_value(value: float, bins: List[dict], quantiles: Optional[Dict[str, float]] = None) -> str:
    """
    Explanation-only binning:
    - bins: list of {'max': <num|\"Q1\"|...>, 'label': <str>} with a final catch-all having no 'max'.
    - quantiles: optional dict with keys 'Q1','Q2','Q3' derived from training data.
    """
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return "unknown"
    for spec in bins:
        if "max" not in spec:
            return spec.get("label", "unknown")
        max_val = spec["max"]
        if isinstance(max_val, str) and max_val.upper().startswith("Q"):
            if quantiles is None or max_val.upper() not in quantiles:
                continue
            thresh = quantiles[max_val.upper()]
        else:
            thresh = float(max_val)
        if value <= thresh:
            return spec.get("label", "bin")
    return bins[-1].get("label", "bin")


def compute_quantiles_from_training(
    train_csv_path: str = "notebooks/models/X_train_raw.csv",
    fields: Sequence[str] = ("credit_amount",),
) -> Dict[str, Dict[str, float]]:
    """
    Compute Q1/Q2/Q3 for specified numeric fields using the saved training split.
    Only used for explanation-time binning; does not alter the model pipeline.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return {}
    p = Path(train_csv_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    qmap: Dict[str, Dict[str, float]] = {}
    for field in fields:
        if field in df.columns:
            qs = df[field].quantile([0.25, 0.5, 0.75])
            qmap[field] = {
                "Q1": float(qs.loc[0.25]),
                "Q2": float(qs.loc[0.5]),
                "Q3": float(qs.loc[0.75]),
            }
    return qmap


def map_row_to_ontology(
    row: Dict[str, object],
    ontology: Optional[Dict[str, dict]] = None,
    quantiles: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, str]:
    """
    Map a raw row (dict-like) to ontology-labeled values for explanation/RAG/ASP.
    - Uses value_map if provided.
    - Uses bins for numeric fields (explanation-only), with optional quantiles for Q1/Q2/Q3 placeholders.
    """
    print("Mapping row to ontology labels...")
    if ontology is None:
        print("No ontology provided, loading default mapping...")
        ontology = load_ontology_mapping()
    out: Dict[str, str] = {}
    for col, meta in ontology.items():
        if col not in row:
            continue
        val = row[col]
        term = meta.get("ontology_term", f"fibo:{col}")
        label = None
        # categorical mapping
        if "value_map" in meta:
            label = meta["value_map"].get(str(val), str(val))
        # numeric bins
        elif "bins" in meta:
            field_q = None
            if quantiles and col in quantiles:
                field_q = quantiles[col]
            try:
                num_val = float(val)
                label = _bin_numeric_value(num_val, meta["bins"], field_q)
            except Exception:
                label = str(val)
        else:
            label = str(val)
        out[term] = label
    return out


# ---------- Regulatory corpus helpers ----------

def load_regulatory_texts(reg_dir: str = "regulatory") -> List[Dict[str, str]]:
    """
    Load text files from the regulatory/ directory.
    Returns a list of dicts with keys: path, title, text.
    """
    docs: List[Dict[str, str]] = []
    for p in sorted(Path(reg_dir).glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        fileTitle = p.stem.replace("_", " ").title()
        print(f"loading document file: {fileTitle} ")
        docs.append(
            {
                "path": str(p),
                "title": fileTitle,
                "text": text,
            }
        )
        
    return docs


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "sentence-transformers or faiss-cpu not installed. "
            "Install them to enable regulatory retrieval."
        ) from exc
    return SentenceTransformer, faiss


def build_faiss_index(
    reg_dir: str = "regulatory",
    model_name: str = "all-MiniLM-L6-v2",
    index_path: str = "indexes/regulatory.index",
) -> str:
    """
    Build a FAISS index over regulatory snippets.
    Writes index to index_path and metadata to index_path.meta.json.
    """
    docs = load_regulatory_texts(reg_dir)
    if not docs:
        raise ValueError(f"No regulatory snippets found in {reg_dir}. Add .txt files first.")

    SentenceTransformer, faiss = _require_sentence_transformers()
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")

    dim = embeddings.shape[1]
    # IndexFlatL2 = simple, exact nearest-neighbor search
    # Distance metric = L2 (Euclidean distance)
    # Very fast for small/medium datasets.
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    index_path = str(index_path)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    meta_path = Path(index_path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(f"Saved FAISS index to {index_path} and metadata to {meta_path}")
    return index_path


@dataclass
class RetrievedSnippet:
    text: str # the snippet text
    score: float # similarity score (lower is more similar for L2) e.g., FAISS distance
    source: str # source file path
    title: Optional[str] = None # optional title


def retrieve_regulatory_snippets(
    query: str,
    model_name: str = "all-MiniLM-L6-v2",
    index_path: str = "indexes/regulatory.index",
    k: int = 3,
) -> List[RetrievedSnippet]:
    """
    Retrieve top-k snippets for a query from a FAISS index + metadata.
    """
    SentenceTransformer, faiss = _require_sentence_transformers()
    meta_path = Path(index_path).with_suffix(".meta.json")
    if not Path(index_path).exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Index or metadata missing. Build with build_faiss_index() at {index_path}."
        )

    print(f"loading meta file at path: {meta_path.absolute()} ")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    texts = [m["text"] for m in meta]
    titles = [m.get("title") for m in meta]
    sources = [m.get("path", "") for m in meta]

    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    index = faiss.read_index(index_path)
    scores, idxs = index.search(q_emb, min(k, len(texts)))
    results: List[RetrievedSnippet] = []
    
    for score, idx in zip(scores[0], idxs[0]):
        results.append(
            RetrievedSnippet(
                text=texts[int(idx)],
                score=float(score),
                source=sources[int(idx)],
                title=titles[int(idx)],
            )
        )
        
    return results


# ---------- Prompt scaffolding ----------

def _format_top_features(top_features: Sequence[Tuple[str, float]], limit: int = 5) -> str:
    """
    Convert a list of (feature_name, contribution_value) pairs into a readable
    multi-line bullet list.

    Parameters
    ----------
    top_features : Sequence[Tuple[str, float]]
        A sequence of tuples where each tuple contains:
        - feature name (str)
        - numeric contribution value (float)
    limit : int, default=5
        Maximum number of top features to include in the output.

    Returns
    -------
    str
        A formatted string with one feature per line, or "None" if empty.
    """
    lines = []
    
    for name, val in list(top_features)[:limit]:
        lines.append(f"- {name}: contribution {val:.4f}")
        
    return "\n".join(lines) if lines else "None"


def _format_ontology_labels(ontology_labels: Optional[Dict[str, str]]) -> str:
    """
    Format ontology label key/value mappings into a multi-line bullet list.

    Parameters
    ----------
    ontology_labels : Dict[str, str] or None
        Dictionary mapping ontology terms to their corresponding values.

    Returns
    -------
    str
        Bullet list of ontology labels, or "None" if empty.
    """
    if not ontology_labels:
        return "None"
    return "\n".join([f"- {k}: {v}" for k, v in ontology_labels.items()])


def _format_snippets(snippets: Sequence[RetrievedSnippet]) -> str:
    """
    Format retrieved regulatory/document snippets into a readable list,
    including title/source, score, and text content.

    Parameters
    ----------
    snippets : Sequence[RetrievedSnippet]
        A list of RetrievedSnippet objects holding:
        - text
        - score
        - source
        - optional title

    Returns
    -------
    str
        Human-readable multi-line string of snippet summaries,
        or "None" if no snippets were provided.
    """
    if not snippets:
        return "None"
    blocks = []
    for snip in snippets:
        title = snip.title or snip.source
        blocks.append(f"* {title} (score={snip.score:.2f}): {snip.text}")
    return "\n".join(blocks)


def build_explanation_prompt(
    decision: str,
    probability: Optional[float],
    top_features: Sequence[Tuple[str, float]],
    ontology_labels: Optional[Dict[str, str]],
    retrieved_snippets: Sequence[RetrievedSnippet],
    mode: str = "rag",
) -> str:
    """
    Assemble a prompt for explanation generation.
    - mode == "rag": include regulatory snippets, ontology labels, and model factors with compliance-aware instructions.
    - mode != "rag": keep it plain (decision + probability only), no regulatory, ontology, or factor references.
    """
    prob_str = f"{probability:.3f}" if probability is not None else "N/A"
    
    print(f"Building explanation prompt in mode: {mode} ")

    if mode == "rag":
        features_block = _format_top_features(top_features)
        ontology_block = _format_ontology_labels(ontology_labels)
        snippets_block = _format_snippets(retrieved_snippets)

        instructions = (
            "You are a compliance-aware credit model explainer. "
            "Write a short, plain-language explanation for the decision, citing key factors. "
            "Use plain language, reference risk/solvency when relevant, and align with GDPR Art.22, "
            "Basel III risk awareness, and MiFID II transparency. "
            "Ground your reasoning in the retrieved regulatory snippets and ontology labels. "
            "Do not fabricate inputs; only use the provided context."
        )

        prompt = f"""
            {instructions}

            Decision: {decision}
            Probability (approve/reject score): {prob_str}

            Top model factors (SHAP/LIME):
            {features_block}

            Ontology labels:
            {ontology_block}

            Retrieved regulatory snippets:
            {snippets_block}

            Deliver:
            - A short explanation (3-5 sentences).
            - Call out which factors increased or decreased approval likelihood.
            - Note any compliance considerations if visible from the snippets.
            """
    else:
        instructions = (
            "You are a concise credit model explainer. "
            "Write a short, plain-language explanation for the decision using only the provided decision and probability. "
            "Do not reference regulatory texts, ontology labels, or model factor details."
        )
        prompt = f"""
            {instructions}

            Decision: {decision}
            Probability (approve/reject score): {prob_str}

            Deliver:
            - A brief rationale (2-4 sentences) consistent with the decision and probability.
            - Avoid mentioning regulatory clauses, ontology terms, or feature attributions.
            """
    final_prompt = prompt.strip()
    print(f"Prompt built successfully.\n {final_prompt} ")
    return final_prompt


# ---------- LLM client ----------


class LLMClient:
    """
    A simple unified interface for generating text using either:
    - a local HuggingFace Transformers model, or
    - an OpenAI-hosted chat model.

    This abstraction allows switching between local and cloud inference
    without changing the rest of your application code.
    """

    def __init__(
        self,
        provider: str = "local",
        local_model: str = "sshleifer/tiny-gpt2",
        openai_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_tokens: int = 400,
    ) -> None:
        """
        Initialize the LLM client.

        Parameters
        ----------
        provider : str
            Which backend to use: "local" (Transformers) or "openai".
        local_model : str
            HuggingFace model name for local inference.
        openai_model : str
            OpenAI model name to use when provider="openai".
        api_key : str, optional
            API key for OpenAI. Falls back to OPENAI_API_KEY environment variable.
        max_tokens : int
            Maximum number of tokens to generate in responses.

        Raises
        ------
        ValueError
            If provider is not "local" or "openai".
        RuntimeError
            If required dependencies are missing.
        """
        self.provider = provider
        self.local_model = local_model
        self.openai_model = openai_model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

        if provider == "openai":
            self._init_openai()
        elif provider == "local":
            self._init_local()
        else:
            raise ValueError("provider must be 'local' or 'openai'")

    def _init_openai(self) -> None:
        """
        Initialize the OpenAI API client.

        Loads the OpenAI SDK, validates the API key,
        and creates an authenticated OpenAI client instance.

        Raises
        ------
        RuntimeError
            If the OpenAI package is not installed or API key is missing.
        """
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("openai package not installed.") from exc

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")

        self._client = OpenAI(api_key=self.api_key)

    def _init_local(self) -> None:
        """
        Initialize a local HuggingFace Transformers text-generation pipeline.

        Loads the specified model from the 'local_model' attribute
        and prepares it for inference.

        Raises
        ------
        RuntimeError
            If transformers library is not installed.
        """
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError("transformers not installed. Install it for local inference.") from exc

        self._client = pipeline("text-generation", model=self.local_model)

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate text using the selected LLM backend.

        Parameters
        ----------
        prompt : str
            The user prompt to send to the model.
        temperature : float
            Sampling temperature (0–1). Higher = more randomness.

        Returns
        -------
        str
            The generated output text.

        Raises
        ------
        RuntimeError
            If the initialization failed earlier.
        """
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a concise, compliance-aware explainer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content  # type: ignore
        else:
            outputs = self._client(
                prompt,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=self.max_tokens,
                num_return_sequences=1,
            )
            return outputs[0]["generated_text"]  # type: ignore



def explain_with_rag(
    decision: str,
    probability: Optional[float],
    top_features: Sequence[Tuple[str, float]],
    ontology_labels: Optional[Dict[str, str]],
    query: str,
    raw_row: Optional[Dict[str, object]] = None,
    provider: str = "local",
    retrieved_snippets: Optional[Sequence[RetrievedSnippet]] = None,
    model_name: str = "all-MiniLM-L6-v2",
    index_path: str = "indexes/regulatory.index",
    local_model: str = "sshleifer/tiny-gpt2",
    openai_model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    High-level helper: retrieve regulatory context (if not provided), build prompt, and call the chosen LLM.
    """
    # Build ontology labels from raw row if not supplied (explanation-only binning)
    if ontology_labels is None and raw_row is not None:
        mapping = load_ontology_mapping()
        # compute quantiles for fields that use Q1/Q2/Q3
        needs_quantiles = any(
            isinstance(meta, dict) and "bins" in meta and any(isinstance(b.get("max"), str) and b.get("max", "").upper().startswith("Q") for b in meta.get("bins", []))
            for meta in mapping.values()
        )
        quantiles = compute_quantiles_from_training() if needs_quantiles else None
        ontology_labels = map_row_to_ontology(raw_row, mapping, quantiles)

    snippets = list(retrieved_snippets) if retrieved_snippets is not None else retrieve_regulatory_snippets(
        query=query,
        model_name=model_name,
        index_path=index_path,
        k=3,
    )
    prompt = build_explanation_prompt(
        decision=decision,
        probability=probability,
        top_features=top_features,
        ontology_labels=ontology_labels,
        retrieved_snippets=snippets,
        mode="rag" if snippets else "llm_only",
    )
    client = LLMClient(
        provider=provider,
        local_model=local_model,
        openai_model=openai_model,
    )
    
    return client.generate(prompt, temperature=temperature)
