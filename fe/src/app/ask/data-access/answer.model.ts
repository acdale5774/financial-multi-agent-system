/**
 * Mirrors the backend response models (be/agents/schemas.py::MultiAgentAnswer).
 */

export interface SimFinCitation {
  id: string;
  source: 'simfin';
  granularity: 'datapoint' | 'query';
  ticker: string | null;
  company: string | null;
  fiscal_year: number | null;
  fiscal_period: string | null;
  metric: string | null;
  value: number | null;
  currency: string | null;
  sql_index: number;
}

export interface DocumentCitation {
  id: string;
  source: 'document';
  document_id: string;
  title: string | null;
  doc_type: string | null;
  company_name: string | null;
  section: string | null;
  chunk_index: number;
  document_date: string | null;
  snippet: string;
  score: number;
}

export type Citation = SimFinCitation | DocumentCitation;

export interface ChartRecord {
  id: string;
  source: 'chart';
  url: string;
  file: string;
  spec: Record<string, unknown>;
  citation_ids: string[];
}

export interface TableCell {
  text: string | null;
  value: number | null;
  citation_id: string | null;
}

export interface TableRecord {
  id: string;
  source: 'table';
  title: string;
  columns: string[];
  rows: TableCell[][];
  markdown: string;
}

export interface SqlQueryRecord {
  query: string;
  row_count: number;
  rows: Record<string, unknown>[];
}

export interface SearchRecord {
  query: string;
  filters: Record<string, unknown>;
  result_count: number;
}

export interface ValidationReport {
  markers_found: number;
  invalid_markers_stripped: string[];
  repaired: boolean;
  uncited_numeric_sentences: number;
  value_mismatch_sentences: number;
  warnings: string[];
}

export interface MultiAgentAnswer {
  question: string;
  route: string;
  route_reason: string;
  answer: string;
  citations: Citation[];
  charts: ChartRecord[];
  tables: TableRecord[];
  validation: ValidationReport;
  sql_queries: SqlQueryRecord[];
  searches: SearchRecord[];
  model: string;
}

/** 48211000000 -> "48.2B" — matches the backend's display convention. */
export function formatValue(value: number | null): string {
  if (value === null) return '—';
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${(value / 1e12).toFixed(1)}T`;
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `${(value / 1e3).toFixed(1)}K`;
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
}
