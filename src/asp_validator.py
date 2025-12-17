# validate that an explanation mentions required factors using ASP/Clingo. 
# Here’s an example of producing a simple ASP program and calling clingo to check pass/fail.
# src/asp_validator.py
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def _sanitize(term: str) -> str:
    """Convert a term to an ASP-safe atom (lowercase, underscores)."""
    return term.strip().replace(" ", "_").lower()


def build_asp_program(
    explanation_terms: Sequence[str],
    required_terms: Sequence[str],
    risk_flags: Sequence[str],
    rule_path: str = "asp/rules/compliance.lp",
    decision: Optional[str] = None,
    require_transparency: bool = True,
) -> str:
    """
    Build an ASP program that asserts facts from the explanation and uses a rules file.
    - explanation_terms: factors mentioned in the LLM explanation (e.g., income, debt_ratio).
    - required_terms: must be present to satisfy completeness.
    - risk_flags: if present, must be acknowledged (Basel risk awareness).
    - decision: optional (approve/reject) fact for downstream rules.
    - require_transparency: set a fact to enforce transparency requirement.
    """
    asp_lines: List[str] = []
    asp_lines.append(f'#include "{rule_path}".')

    for t in explanation_terms:
        asp_lines.append(f"mentioned({_sanitize(t)}).")
    for t in required_terms:
        asp_lines.append(f"required({_sanitize(t)}).")
    for t in risk_flags:
        asp_lines.append(f"risk_flag({_sanitize(t)}).")
    if decision:
        asp_lines.append(f"decision({_sanitize(decision)}).")
    if require_transparency:
        asp_lines.append("require_transparency.")

    return "\n".join(asp_lines)


def run_clingo(asp_program: str, clingo_bin: str = "clingo") -> Dict[str, Iterable[str]]:
    """
    Run clingo on the provided ASP program string and return violations/pass.
    """
    with tempfile.NamedTemporaryFile(suffix=".lp", mode="w", delete=False) as f:
        f.write(asp_program)
        fname = f.name
    proc = subprocess.run([clingo_bin, fname], capture_output=True, text=True)
    out = proc.stdout
    # parse simple answers: look for 'pass' and 'violation(...)'
    violations: List[str] = []
    passed = False
    for line in out.splitlines():
        if line.startswith("Answer"):
            continue
        if "pass" in line:
            passed = True
        if "violation(" in line:
            # extract atoms like violation(missing_required_income)
            parts = [p.strip() for p in line.split() if p.startswith("violation(")]
            violations.extend(parts)
    return {"pass": passed, "violations": violations, "stdout": out, "stderr": proc.stderr}


# Example usage (you’d first parse the LLM explanation into explanation_terms):
# asp_program = build_asp_program(
#     explanation_terms=["income", "debt_ratio", "credit_history"],
#     required_terms=["income", "debt_ratio", "credit_history"],
#     risk_flags=["high_debt"],
# )
# result = run_clingo(asp_program)
# print(result)
