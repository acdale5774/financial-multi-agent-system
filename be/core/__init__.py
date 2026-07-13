"""Cross-cutting foundations (configuration) shared by all backend layers.

Kept deliberately tiny — only things every layer may need. Lower layers
(`db`, `ingestion`, `tools`) may import from `core`; `core` imports from nobody.
"""
