# Frontend (`/fe`)

User-facing application for the Financial Multi-Agent Intelligence System.

> Status: **placeholder**. The frontend is not implemented yet; the stack is
> intentionally left open.

## Purpose

Provide a chat-style interface where a user asks a financial question and sees:

- the natural-language **answer**,
- the **citations** backing it (SQL result references and document passages), and
- any **charts** the backend returned (rendered from declarative chart specs — see
  [`../be/tools/chart_tool.py`](../be/tools/chart_tool.py)).

## Backend contract

The frontend talks to the FastAPI backend in [`../be`](../be). The primary
interaction will be a question-answering endpoint (planned `POST /ask`) that
returns an answer, citations, and chart specifications.

> TODO: Choose the framework (e.g. a React/Next.js single-page app), a charting
> library that consumes the backend's chart spec (e.g. Vega-Lite), and document
> the request/response contract once the backend `POST /ask` endpoint is defined.
