"""Tools package.

Small, independently testable functions the agent can call:

- `sql_tool`             — safe, read-only SQL over structured financials.
- `document_search_tool` — semantic search over document chunks (with citations).
- `chart_tool`           — turn structured results into a chart specification.

Each tool does one thing and returns plain data; the agent decides when to use them.
"""
