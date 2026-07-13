import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  output,
} from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { marked } from 'marked';

const MARKER = /\[((?:[SDCT]\d+)(?:\s*,\s*(?:[SDCT]\d+))*)\](?!\()/g;

/**
 * Renders the answer markdown; [S1]-style markers become clickable chips
 * (one delegated click handler — the HTML itself stays derived state).
 */
@Component({
  selector: 'app-answer-card',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<article class="answer" [innerHTML]="html()" (click)="onClick($event)"></article>`,
  styles: `
    .answer {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      line-height: 1.65;
      overflow-x: auto;
    }
    .answer ::ng-deep table {
      border-collapse: collapse;
      margin: 0.75rem 0;
    }
    .answer ::ng-deep th,
    .answer ::ng-deep td {
      border: 1px solid var(--border);
      padding: 0.4rem 0.7rem;
      text-align: right;
    }
    .answer ::ng-deep th:first-child,
    .answer ::ng-deep td:first-child {
      text-align: left;
    }
    .answer ::ng-deep img {
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin: 0.5rem 0;
    }
    .answer ::ng-deep .cite {
      display: inline-block;
      margin: 0 0.15rem;
      padding: 0 0.4rem;
      border: 1px solid var(--accent-soft);
      border-radius: 999px;
      background: var(--accent-bg);
      color: var(--accent);
      font-size: 0.72rem;
      font-weight: 600;
      cursor: pointer;
      vertical-align: 0.1rem;
    }
  `,
})
export class AnswerCard {
  private readonly sanitizer = inject(DomSanitizer);

  readonly markdown = input.required<string>();
  readonly citationClicked = output<string>();

  protected readonly html = computed<SafeHtml>(() => {
    const withChips = this.markdown().replace(MARKER, (_whole, group: string) =>
      group
        .split(',')
        .map((id) => id.trim())
        .map((id) => `<button type="button" class="cite" data-id="${id}">${id}</button>`)
        .join(''),
    );
    const rendered = marked.parse(withChips, { async: false }) as string;
    return this.sanitizer.bypassSecurityTrustHtml(rendered);
  });

  protected onClick(event: Event): void {
    const chip = (event.target as HTMLElement).closest('.cite');
    const id = chip?.getAttribute('data-id');
    if (id) this.citationClicked.emit(id);
  }
}
