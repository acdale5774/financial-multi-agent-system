import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { Citation, formatValue } from '../data-access/answer.model';

/** Sidebar listing every cited record so any claim can be verified. */
@Component({
  selector: 'app-citation-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <h2>Citations</h2>
    @if (citations().length === 0) {
      <p class="muted">No citations in this answer.</p>
    }
    @for (c of citations(); track c.id) {
      <button
        type="button"
        class="citation"
        [class.selected]="c.id === selectedId()"
        (click)="toggled.emit(c.id)"
      >
        <span class="chip" [class.doc]="c.source === 'document'">{{ c.id }}</span>
        @if (c.source === 'simfin') {
          @if (c.granularity === 'datapoint') {
            <span class="title">
              {{ c.ticker || c.company }} · {{ c.metric }} · {{ c.fiscal_period }}
              {{ c.fiscal_year }}
            </span>
            <span class="detail">
              {{ fmt(c.value) }} {{ c.currency || '' }} — SQL query #{{ c.sql_index + 1 }}
            </span>
          } @else {
            <span class="title">Whole-result citation</span>
            <span class="detail">SQL query #{{ c.sql_index + 1 }} (see agent reasoning)</span>
          }
        } @else {
          <span class="title">{{ c.title || c.document_id }}</span>
          <span class="detail">
            {{ c.doc_type }}@if (c.section) { · §{{ c.section }}}
            @if (c.document_date) { · {{ c.document_date }}} · relevance {{ c.score }}
          </span>
          @if (c.id === selectedId()) {
            <blockquote>{{ c.snippet }}…</blockquote>
          }
        }
      </button>
    }
  `,
  styles: `
    h2 {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin: 0 0 0.6rem;
    }
    .citation {
      display: block;
      width: 100%;
      text-align: left;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.6rem 0.75rem;
      margin-bottom: 0.5rem;
      cursor: pointer;
    }
    .citation.selected {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px var(--accent);
    }
    .chip {
      display: inline-block;
      padding: 0 0.45rem;
      border-radius: 999px;
      background: var(--accent-bg);
      color: var(--accent);
      font-size: 0.72rem;
      font-weight: 700;
      margin-right: 0.4rem;
    }
    .chip.doc {
      background: var(--doc-bg);
      color: var(--doc);
    }
    .title {
      font-weight: 600;
      font-size: 0.86rem;
    }
    .detail {
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      margin-top: 0.15rem;
    }
    blockquote {
      margin: 0.5rem 0 0;
      padding: 0.4rem 0.6rem;
      border-left: 3px solid var(--doc);
      background: var(--doc-bg);
      font-size: 0.8rem;
      color: var(--text);
      white-space: pre-wrap;
    }
    .muted {
      color: var(--muted);
      font-size: 0.85rem;
    }
  `,
})
export class CitationPanel {
  readonly citations = input.required<Citation[]>();
  readonly selectedId = input<string | null>(null);
  readonly toggled = output<string>();

  protected readonly fmt = formatValue;
}
