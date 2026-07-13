import { ChangeDetectionStrategy, Component } from '@angular/core';

import { AskPage } from './ask/ask-page';

@Component({
  selector: 'app-root',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [AskPage],
  template: `
    <header>
      <h1>Financial Multi-Agent Intelligence</h1>
      <p>Cited answers over SimFin financials, SEC filings, and earnings calls</p>
    </header>
    <main>
      <app-ask-page />
    </main>
  `,
  styles: `
    header {
      padding: 1.5rem 0 1rem;
    }
    h1 {
      font-size: 1.35rem;
      margin: 0;
    }
    header p {
      margin: 0.2rem 0 0;
      color: var(--muted);
      font-size: 0.9rem;
    }
  `,
})
export class App {}
