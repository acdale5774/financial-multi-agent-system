"""The multi-agent system: routed specialists over a shared evidence ledger.

    question -> dispatch (one structured-output call, deterministic fallback)
             -> ONE specialist agent (LangChain create_agent tool loop):
                  quant   - SQL over the structured store (+ charts)
                  docs    - filtered semantic search over filings/transcripts
                  hybrid  - both toolsets for genuinely cross-domain questions
             -> deterministic citation validation (one repair turn, then
                strip-and-warn) -> MultiAgentAnswer

Topology rationale (full discussion in agents/README.md): the dispatcher is a
classifier, not a supervisor — it runs once, cannot fail a request (any error
falls back to the hybrid superset), and keeps each request to at most two LLM
roles, which is what makes a live gpt-5.5 demo predictable. Specialists get
isolated toolsets and prompts because quant and document work want different
disciplines (citation-keyed SQL vs coverage-scoped search); the hybrid agent
is the escape hatch for questions that genuinely need both in one context.

Every request gets a fresh EvidenceLedger and fresh tool closures over it, so
concurrent requests cannot leak evidence into each other. All citation ids in
the final prose are validated against the ledger before the answer leaves
this module — see agents/citations.py.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from agents import citations as citation_rules
from agents.lc_tools import make_toolsets
from agents.ledger import EvidenceLedger
from agents.prompts import DOC_PROMPT, HYBRID_PROMPT, QUANT_PROMPT, ROUTER_PROMPT
from agents.schemas import MultiAgentAnswer, ValidationReport

_SPECIALIST_PROMPTS = {"quant": QUANT_PROMPT, "docs": DOC_PROMPT, "hybrid": HYBRID_PROMPT}

UNSUPPORTED_ANSWER = (
    "This system answers questions about the companies in its financial "
    "database and document corpus: quantitative questions over SimFin "
    "financial statements (lookups, comparisons, rankings, trends, charts) "
    "and qualitative questions over SEC filings and earnings-call "
    "transcripts (summaries, risks, management commentary). Please ask a "
    "question about that data."
)


class Route(BaseModel):
    """The dispatcher's verdict, echoed verbatim in the response."""

    route: Literal["quant", "docs", "hybrid", "unsupported"]
    reason: str


def dispatch(question: str, *, chat_model: Any) -> Route:
    """Classify the question; any failure falls back to the hybrid superset."""
    try:
        result = chat_model.with_structured_output(Route).invoke(
            [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": question},
            ]
        )
        if isinstance(result, Route):
            return result
        return Route.model_validate(result)
    except Exception as exc:  # noqa: BLE001 - routing must never fail a request
        return Route(
            route="hybrid",
            reason=f"dispatcher unavailable ({type(exc).__name__}); "
            "defaulted to the full toolset",
        )


def answer_multi(
    question: str,
    *,
    model: str | None = None,
    database_url: str | None = None,
    chat_model: Any = None,
    router: Any = None,
    max_iterations: int | None = None,
) -> MultiAgentAnswer:
    """Answer a question through the routed multi-agent system.

    Args:
        question: The user's question.
        model: OpenAI model id; defaults to settings.agent_model.
        database_url: Override Postgres connection string.
        chat_model: Injectable LangChain chat model (tests use a fake).
        router: Injectable callable question -> Route (tests bypass the LLM).
        max_iterations: Cap on specialist model turns.

    Returns:
        MultiAgentAnswer with cited prose, the citation records, charts, and
        the full SQL/search trace.
    """
    from core.config import settings

    model = model or settings.agent_model
    max_iterations = max_iterations or settings.agent_max_iterations
    if chat_model is None:
        from langchain_openai import ChatOpenAI  # lazy: tests inject fakes

        chat_model = ChatOpenAI(model=model, api_key=settings.openai_api_key or None)

    route = router(question) if router is not None else dispatch(question, chat_model=chat_model)

    ledger = EvidenceLedger()
    if route.route == "unsupported":
        return _assemble(
            question, route, UNSUPPORTED_ANSWER, ledger, model, ValidationReport()
        )

    toolsets = make_toolsets(ledger, database_url)
    answer_text, repaired = _run_specialist(
        chat_model=chat_model,
        tools=toolsets[route.route],
        system_prompt=_SPECIALIST_PROMPTS[route.route],
        question=question,
        ledger=ledger,
        recursion_limit=2 * max_iterations + 2,  # each turn = model + tool node
    )

    answer_text, report = citation_rules.apply_validation(
        answer_text, ledger, repaired=repaired
    )
    return _assemble(question, route, answer_text, ledger, model, report)


def _run_specialist(
    *,
    chat_model: Any,
    tools: list[Any],
    system_prompt: str,
    question: str,
    ledger: EvidenceLedger,
    recursion_limit: int,
) -> tuple[str, bool]:
    """Run one create_agent loop; one repair turn if markers don't validate.

    Never raises: the iteration cap (GraphRecursionError) degrades to a
    partial answer over whatever evidence was collected.
    """
    from langchain.agents import create_agent
    from langgraph.errors import GraphRecursionError

    graph = create_agent(chat_model, tools, system_prompt=system_prompt)
    config = {"recursion_limit": recursion_limit}

    try:
        state = graph.invoke({"messages": [{"role": "user", "content": question}]}, config)
    except GraphRecursionError:
        return (
            "The analysis hit its iteration cap before producing a final "
            "answer. The queries and evidence collected so far are attached "
            "below.",
            False,
        )

    text = _final_text(state["messages"])
    _, invalid = citation_rules.split_valid_invalid(text, ledger)
    if not invalid:
        return text, False

    # One bounded repair turn: same conversation plus a corrective message.
    repair = citation_rules.build_repair_message(invalid, ledger)
    try:
        state = graph.invoke(
            {"messages": [*state["messages"], {"role": "user", "content": repair}]}, config
        )
        return _final_text(state["messages"]), True
    except GraphRecursionError:
        return text, True  # keep the original; validation strips what's left


def _final_text(messages: list[Any]) -> str:
    """Content of the last AI message (str or content-block list)."""
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" or message.__class__.__name__ == "AIMessage":
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):  # content blocks
                return "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
    return ""


def _assemble(
    question: str,
    route: Route,
    answer_text: str,
    ledger: EvidenceLedger,
    model: str,
    report: ValidationReport,
) -> MultiAgentAnswer:
    cited_ids = citation_rules.extract_marker_ids(answer_text)
    return MultiAgentAnswer(
        question=question,
        route=route.route,
        route_reason=route.reason,
        answer=answer_text,
        citations=ledger.cited_subset(cited_ids),
        charts=list(ledger.charts.values()),
        tables=list(ledger.tables.values()),
        validation=report,
        sql_queries=ledger.sql_queries,
        searches=ledger.searches,
        model=model,
    )
