import { ChangeDetectionStrategy, Component, input, output, signal } from '@angular/core';

const EXAMPLES = [
  'Collate Revenue, Net Income and EPS for Ford across the last 4 quarters in a table, and plot the EPS over time.',
  'Which companies grew revenue fastest year over year in the last fiscal year?',
  'Plot the revenue for Ford over the last 3 quarters.',
  'What are the common trends or risks seen across recent earnings disclosures for the Passenger Airlines sector?',
  'List all companies in the database with sector and most recent annual revenue.',
];

@Component({
  selector: 'app-question-input',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <form class="ask-form" (submit)="submit($event)">
      <input
        type="text"
        name="question"
        placeholder="Ask a financial question…"
        [value]="draft()"
        (input)="draft.set($any($event.target).value)"
        [disabled]="busy()"
      />
      <button type="submit" [disabled]="busy() || !draft().trim()">
        {{ busy() ? 'Working…' : 'Ask' }}
      </button>
    </form>
    <div class="examples">
      @for (example of examples; track example) {
        <button type="button" class="example" [disabled]="busy()" (click)="use(example)">
          {{ example }}
        </button>
      }
    </div>
  `,
  styles: `
    .ask-form {
      display: flex;
      gap: 0.5rem;
    }
    input {
      flex: 1;
      padding: 0.65rem 0.9rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 1rem;
      background: var(--surface);
    }
    button[type='submit'] {
      padding: 0.65rem 1.4rem;
      border: none;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      cursor: pointer;
    }
    button[type='submit']:disabled {
      opacity: 0.5;
      cursor: default;
    }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-top: 0.6rem;
    }
    .example {
      font-size: 0.78rem;
      padding: 0.3rem 0.6rem;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--surface);
      color: var(--muted);
      cursor: pointer;
      text-align: left;
    }
    .example:hover:not(:disabled) {
      border-color: var(--accent);
      color: var(--accent);
    }
  `,
})
export class QuestionInput {
  readonly busy = input.required<boolean>();
  readonly asked = output<string>();

  protected readonly examples = EXAMPLES;
  protected readonly draft = signal('');

  protected submit(event: Event): void {
    event.preventDefault();
    const question = this.draft().trim();
    if (question) this.asked.emit(question);
  }

  protected use(example: string): void {
    this.draft.set(example);
    this.asked.emit(example);
  }
}
