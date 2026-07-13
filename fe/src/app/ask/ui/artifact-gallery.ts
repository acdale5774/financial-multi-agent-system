import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { ChartRecord, TableRecord, formatValue } from '../data-access/answer.model';

/**
 * Structured visuals (requirement 4): charts rendered by the backend and
 * citation-hydrated tables, each value cell wearing its evidence chip.
 */
@Component({
  selector: 'app-artifact-gallery',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @for (chart of charts(); track chart.id) {
      <figure class="artifact">
        <img [src]="chart.url" [alt]="chart.id" />
        <figcaption>
          <span class="chip">{{ chart.id }}</span>
          plotted from
          @for (id of chart.citation_ids; track $index) {
            <button type="button" class="cite" (click)="citationClicked.emit(id)">{{ id }}</button>
          }
        </figcaption>
      </figure>
    }
    @for (table of tables(); track table.id) {
      <figure class="artifact">
        <figcaption>
          <span class="chip">{{ table.id }}</span> {{ table.title }}
        </figcaption>
        <table>
          <thead>
            <tr>
              @for (col of table.columns; track $index) {
                <th>{{ col }}</th>
              }
            </tr>
          </thead>
          <tbody>
            @for (row of table.rows; track $index) {
              <tr>
                @for (cell of row; track $index) {
                  <td>
                    @if (cell.citation_id) {
                      {{ fmt(cell.value) }}
                      <button
                        type="button"
                        class="cite"
                        (click)="citationClicked.emit(cell.citation_id)"
                      >
                        {{ cell.citation_id }}
                      </button>
                    } @else {
                      {{ cell.text }}
                    }
                  </td>
                }
              </tr>
            }
          </tbody>
        </table>
      </figure>
    }
  `,
  styles: `
    .artifact {
      margin: 1rem 0 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.9rem 1rem;
      overflow-x: auto;
    }
    img {
      max-width: 100%;
      border-radius: 8px;
    }
    figcaption {
      font-size: 0.8rem;
      color: var(--muted);
      margin: 0.4rem 0;
    }
    .chip {
      display: inline-block;
      padding: 0 0.45rem;
      border-radius: 999px;
      background: var(--chart-bg);
      color: var(--chart);
      font-size: 0.72rem;
      font-weight: 700;
    }
    table {
      border-collapse: collapse;
      width: 100%;
    }
    th,
    td {
      border: 1px solid var(--border);
      padding: 0.4rem 0.7rem;
      text-align: right;
      font-size: 0.88rem;
    }
    th:first-child,
    td:first-child {
      text-align: left;
    }
    .cite {
      border: 1px solid var(--accent-soft);
      border-radius: 999px;
      background: var(--accent-bg);
      color: var(--accent);
      font-size: 0.7rem;
      font-weight: 600;
      padding: 0 0.35rem;
      cursor: pointer;
    }
  `,
})
export class ArtifactGallery {
  readonly charts = input.required<ChartRecord[]>();
  readonly tables = input.required<TableRecord[]>();
  readonly citationClicked = output<string>();

  protected readonly fmt = formatValue;
}
