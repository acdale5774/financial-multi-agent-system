import { JsonPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import { MultiAgentAnswer } from '../data-access/answer.model';

/** Agent transparency: routing, executed SQL, searches, validation verdict. */
@Component({
  selector: 'app-reasoning-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <details class="panel" open>
      <summary>
        Agent reasoning
        <span class="route">route: {{ answer().route }}</span>
        <span class="model">{{ answer().model }}</span>
      </summary>

      <p class="reason">{{ answer().route_reason }}</p>

      @if (answer().sql_queries.length > 0) {
        <h3>SQL queries ({{ answer().sql_queries.length }})</h3>
        @for (q of answer().sql_queries; track $index) {
          <details class="query">
            <summary>query #{{ $index + 1 }} — {{ q.row_count }} rows</summary>
            <pre>{{ q.query }}</pre>
          </details>
        }
      }

      @if (answer().searches.length > 0) {
        <h3>Document searches ({{ answer().searches.length }})</h3>
        <ul>
          @for (s of answer().searches; track $index) {
            <li>
              “{{ s.query }}” → {{ s.result_count }} chunks
              @if ((s.filters | json) !== '{}') {
                <code>{{ s.filters | json }}</code>
              }
            </li>
          }
        </ul>
      }

      <h3>Citation validation</h3>
      <ul class="validation">
        <li>{{ answer().validation.markers_found }} markers checked against the evidence ledger</li>
        @if (answer().validation.repaired) {
          <li>one repair turn was needed</li>
        }
        @if (answer().validation.invalid_markers_stripped.length > 0) {
          <li class="warn">
            stripped invalid markers: {{ answer().validation.invalid_markers_stripped.join(', ') }}
          </li>
        }
        @if (answer().validation.uncited_numeric_sentences > 0) {
          <li class="warn">
            {{ answer().validation.uncited_numeric_sentences }} numeric sentence(s) without a citation
          </li>
        }
        @if (answer().validation.value_mismatch_sentences > 0) {
          <li class="warn">
            {{ answer().validation.value_mismatch_sentences }} sentence(s) whose figures were not found
            in the cited evidence
          </li>
        }
        @for (w of answer().validation.warnings; track $index) {
          <li class="warn">{{ w }}</li>
        }
        @if (cleanRun()) {
          <li class="ok">no violations, no warnings</li>
        }
      </ul>
    </details>
  `,
  styles: `
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.75rem 1rem;
    }
    summary {
      cursor: pointer;
      font-weight: 600;
    }
    .route,
    .model {
      font-size: 0.75rem;
      font-weight: 500;
      color: var(--muted);
      margin-left: 0.6rem;
      padding: 0.1rem 0.5rem;
      border: 1px solid var(--border);
      border-radius: 999px;
    }
    .reason {
      color: var(--muted);
      font-size: 0.85rem;
    }
    h3 {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin: 0.9rem 0 0.3rem;
    }
    .query summary {
      font-weight: 500;
      font-size: 0.85rem;
    }
    pre {
      background: var(--code-bg);
      border-radius: 8px;
      padding: 0.6rem 0.8rem;
      overflow-x: auto;
      font-size: 0.78rem;
    }
    ul {
      margin: 0.2rem 0;
      padding-left: 1.2rem;
      font-size: 0.85rem;
    }
    code {
      font-size: 0.75rem;
      background: var(--code-bg);
      padding: 0.05rem 0.3rem;
      border-radius: 4px;
    }
    .warn {
      color: var(--warn);
    }
    .ok {
      color: var(--ok);
    }
  `,
  imports: [JsonPipe],
})
export class ReasoningPanel {
  readonly answer = input.required<MultiAgentAnswer>();

  protected cleanRun(): boolean {
    const v = this.answer().validation;
    return (
      v.invalid_markers_stripped.length === 0 &&
      v.uncited_numeric_sentences === 0 &&
      v.value_mismatch_sentences === 0 &&
      v.warnings.length === 0
    );
  }
}
