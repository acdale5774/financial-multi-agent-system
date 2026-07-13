import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { AskService } from './data-access/ask.service';
import { AnswerCard } from './ui/answer-card';
import { ArtifactGallery } from './ui/artifact-gallery';
import { CitationPanel } from './ui/citation-panel';
import { QuestionInput } from './ui/question-input';
import { ReasoningPanel } from './ui/reasoning-panel';

/**
 * Smart container: wires the AskService's selectors into dumb components and
 * forwards their outputs into the service's sources. No logic lives here.
 */
@Component({
  selector: 'app-ask-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [QuestionInput, AnswerCard, ArtifactGallery, CitationPanel, ReasoningPanel],
  template: `
    <app-question-input [busy]="ask.status() === 'loading'" (asked)="ask.ask$.next($event)" />

    @switch (ask.status()) {
      @case ('loading') {
        <div class="status">
          <span class="spinner"></span>
          Answering “{{ ask.question() }}” — {{ ask.elapsedSeconds() }}s
          <span class="hint">(routing, querying, citing…)</span>
        </div>
      }
      @case ('error') {
        <div class="status error">{{ ask.error() }}</div>
      }
      @case ('success') {
        @if (ask.answer(); as a) {
          <div class="result">
            <section class="main">
              <app-answer-card
                [markdown]="a.answer"
                (citationClicked)="ask.citationToggled$.next($event)"
              />
              <app-artifact-gallery
                [charts]="a.charts"
                [tables]="a.tables"
                (citationClicked)="ask.citationToggled$.next($event)"
              />
              <app-reasoning-panel [answer]="a" />
            </section>
            <aside>
              <app-citation-panel
                [citations]="a.citations"
                [selectedId]="ask.selectedCitationId()"
                (toggled)="ask.citationToggled$.next($event)"
              />
            </aside>
          </div>
        }
      }
      @case ('idle') {
        <p class="status hint">
          Ask about SimFin financials (4,599 companies) or the SEC filing / earnings-call corpus
          — every claim in the answer is cited back to its data point or document section.
        </p>
      }
    }
  `,
  styles: `
    .status {
      margin-top: 1.25rem;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .status.error {
      color: var(--warn);
      background: var(--warn-bg);
      border: 1px solid var(--warn);
      border-radius: 8px;
      padding: 0.6rem 0.9rem;
    }
    .hint {
      font-size: 0.85rem;
    }
    .spinner {
      width: 0.9rem;
      height: 0.9rem;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to {
        transform: rotate(360deg);
      }
    }
    .result {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 1rem;
      margin-top: 1.25rem;
      align-items: start;
    }
    .main {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      min-width: 0;
    }
    @media (max-width: 900px) {
      .result {
        grid-template-columns: 1fr;
      }
    }
  `,
})
export class AskPage {
  protected readonly ask = inject(AskService);
}
