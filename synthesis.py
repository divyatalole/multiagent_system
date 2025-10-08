"""
Synthesis Module (MCDM-style Weighted Scoring)
==============================================

Consumes per-agent JSON analyses and produces a unified summary with
weighted scores and a final recommendation. This module intentionally
keeps the math simple and transparent.
"""

from typing import Dict, Any


def _avg_doc_relevance(agent: Dict[str, Any]) -> float:
    previews = agent.get("document_previews") or []
    if not previews:
        return 0.0
    vals = [float(p.get("relevance", 0.0)) for p in previews]
    return sum(vals) / max(1, len(vals))


def _success_probability(agent: Dict[str, Any]) -> float:
    qm = agent.get("quantitative_model") or {}
    sp = qm.get("success_probability")
    if sp is None:
        return 0.0
    return float(sp) / 100.0  # convert to 0..1


def compute_synthesis(agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute overall metrics from agent outputs.

    Inputs (per agent):
      - document_previews[].relevance (0..1)
      - investor.quantitative_model.success_probability (0..100)

    Outputs:
      - overall_score (0..10)
      - risk_level (Low/Medium/High)
      - market_potential (Low/Medium/High)
      - feasibility (Low/Medium/High)
      - recommendation (text)
    """
    investor = agent_results.get("investor", {})
    researcher = agent_results.get("researcher", {})
    user = agent_results.get("user", {})

    inv_rel = _avg_doc_relevance(investor)
    res_rel = _avg_doc_relevance(researcher)
    usr_rel = _avg_doc_relevance(user)

    success_prob = _success_probability(investor)  # 0..1

    # Weighted aggregation (sum to 1)
    # Heuristic: investor 0.4 (includes model), researcher 0.35, user 0.25
    relevance_score = 10.0 * (
        0.4 * inv_rel + 0.35 * res_rel + 0.25 * usr_rel
    )

    # Boost by investor success probability (up to +2 points)
    overall_score = min(10.0, relevance_score + 2.0 * success_prob)

    def band(x: float) -> str:
        if x >= 7.5:
            return "High"
        if x >= 5.0:
            return "Medium"
        return "Low"

    market_potential = band(10.0 * (0.6 * res_rel + 0.4 * inv_rel))
    feasibility = band(10.0 * (0.6 * usr_rel + 0.4 * res_rel))

    # Risk inversely related to investor signals
    risk_raw = 10.0 * (1.0 - (0.7 * inv_rel + 0.3 * success_prob))
    risk_level = "Low" if risk_raw <= 3.5 else ("Medium" if risk_raw <= 6.5 else "High")

    if overall_score >= 8.0:
        recommendation = (
            "Strong recommendation to proceed. Consider preparing a fundraising deck and an MVP roadmap."
        )
    elif overall_score >= 6.0:
        recommendation = (
            "Proceed with targeted validation. Focus on differentiators, early customer pilots, and risk mitigation."
        )
    else:
        recommendation = (
            "Not ready yet. Reassess problem-solution fit, collect more evidence, and iterate on positioning."
        )

    return {
        "overall_score": round(overall_score, 1),
        "risk_level": risk_level,
        "market_potential": market_potential,
        "feasibility": feasibility,
        "recommendation": recommendation,
        "signals": {
            "avg_relevance": {
                "investor": round(inv_rel, 3),
                "researcher": round(res_rel, 3),
                "user": round(usr_rel, 3),
            },
            "success_probability": round(success_prob * 100.0, 1),
        },
    }


