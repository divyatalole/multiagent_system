"""
LangGraph Orchestrator for Multi-Agent Workflow
================================================

- Fans out the initial topic to Investor, Researcher, and User agents in parallel
- Maintains a lightweight conversation state
- Routes follow-up questions to the most relevant agent(s)

This file does NOT change existing agent logic. It composes the existing
RAGMultiAgentSystem and exposes a simple orchestrated interface.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field

# LangGraph
try:
    from langgraph.graph import StateGraph, END
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "LangGraph is required for the orchestrator. Please install with `pip install langgraph`."
    ) from e

from multi_agent_system_simple import RAGMultiAgentSystem
from synthesis import compute_synthesis


@dataclass
class OrchestratorState:
    topic: str
    system_ready: bool = False
    agent_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    followup_question: Optional[str] = None
    followup_target: Optional[str] = None  # 'investor' | 'researcher' | 'user' | 'all'
    followup_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MultiAgentOrchestrator:
    def __init__(self, system: Optional[RAGMultiAgentSystem] = None):
        self.system = system or RAGMultiAgentSystem()
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(OrchestratorState)

        def init_node(state: OrchestratorState):
            state.system_ready = True
            return state

        def fanout_node(state: OrchestratorState):
            # Run all agents in parallel (synchronously for now; agents are independent)
            results = {}
            for agent_id, agent in self.system.agents.items():
                results[agent_id] = agent.analyze_topic(state.topic)
            state.agent_results = results
            return state

        def synthesize_node(state: OrchestratorState):
            if state.agent_results:
                # Compute unified synthesis metrics
                state.agent_results["_synthesis"] = compute_synthesis(state.agent_results)
            return state

        def route_followup_node(state: OrchestratorState):
            if not state.followup_question:
                return state
            target = state.followup_target or self._choose_target(state)
            if target == 'all':
                out = {}
                for agent_id, agent in self.system.agents.items():
                    out[agent_id] = agent.analyze_topic(state.followup_question)
                state.followup_results = out
            else:
                agent = self.system.agents.get(target)
                if agent is not None:
                    state.followup_results = {target: agent.analyze_topic(state.followup_question)}
            return state

        g.add_node("init", init_node)
        g.add_node("fanout", fanout_node)
        g.add_node("route_followup", route_followup_node)
        g.add_node("synthesize", synthesize_node)

        g.set_entry_point("init")
        g.add_edge("init", "fanout")
        g.add_edge("fanout", "synthesize")
        g.add_edge("synthesize", "route_followup")
        g.add_edge("route_followup", END)

        return g.compile()

    def _choose_target(self, state: OrchestratorState) -> str:
        """Simple heuristic: match keywords to pick the most relevant agent."""
        q = (state.followup_question or "").lower()
        investor_kw = ["roi", "revenue", "market", "competition", "valuation", "risk"]
        researcher_kw = ["benchmark", "dataset", "accuracy", "method", "feasibility", "technology"]
        user_kw = ["ux", "adoption", "onboarding", "retention", "pricing", "design", "user"]

        def score(kws):
            return sum(q.count(k) for k in kws)

        scores = {
            'investor': score(investor_kw),
            'researcher': score(researcher_kw),
            'user': score(user_kw),
        }
        # If no clear signal, broadcast to all
        if max(scores.values()) == 0:
            return 'all'
        return max(scores, key=scores.get)

    # Public API
    def run(self, topic: str) -> Dict[str, Any]:
        state = OrchestratorState(topic=topic)
        out: OrchestratorState = self.graph.invoke(state)
        return {
            'topic': topic,
            'agent_results': out.agent_results,
        }

    def follow_up(self, prev_state: Dict[str, Any], question: str, target: Optional[str] = None) -> Dict[str, Any]:
        state = OrchestratorState(
            topic=prev_state.get('topic', ''),
            agent_results=prev_state.get('agent_results', {}),
            followup_question=question,
            followup_target=target,
        )
        out: OrchestratorState = self.graph.invoke(state)
        return {
            'topic': out.topic,
            'agent_results': out.agent_results,
            'followup_question': out.followup_question,
            'followup_results': out.followup_results,
        }


if __name__ == "__main__":  # simple smoke test
    orchestrator = MultiAgentOrchestrator()
    initial = orchestrator.run("AI startup funding")
    print("Agents done:", list(initial['agent_results'].keys()))
    nxt = orchestrator.follow_up(initial, "What is the expected ROI?", target=None)
    print("Follow-up targets:", list(nxt['followup_results'].keys()))


