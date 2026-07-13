/**
 * Declarative state for the ask feature ("Joshua Morony style"):
 *
 *   sources (Subjects fed by the UI)
 *     -> reducers (RxJS flows declared ONCE in the constructor,
 *        each ending in a single state.update)
 *       -> selectors (computed signals the templates read)
 *
 * Components never mutate state imperatively — they push events into the
 * sources and render the selectors. All async orchestration (switchMap on the
 * question stream, error mapping, elapsed-time ticking) lives here,
 * declared up front.
 */
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, map, of, startWith, switchMap, timer } from 'rxjs';

import { Citation, MultiAgentAnswer } from './answer.model';

export type AskStatus = 'idle' | 'loading' | 'success' | 'error';

interface AskState {
  status: AskStatus;
  question: string | null;
  answer: MultiAgentAnswer | null;
  error: string | null;
  selectedCitationId: string | null;
  elapsedSeconds: number;
}

const INITIAL_STATE: AskState = {
  status: 'idle',
  question: null,
  answer: null,
  error: null,
  selectedCitationId: null,
  elapsedSeconds: 0,
};

type AskEvent =
  | { kind: 'started'; question: string }
  | { kind: 'succeeded'; answer: MultiAgentAnswer }
  | { kind: 'failed'; error: string };

@Injectable({ providedIn: 'root' })
export class AskService {
  private readonly http = inject(HttpClient);

  // --- sources -------------------------------------------------------------
  readonly ask$ = new Subject<string>();
  readonly citationToggled$ = new Subject<string>();

  // --- state ---------------------------------------------------------------
  private readonly state = signal<AskState>(INITIAL_STATE);

  // --- selectors -----------------------------------------------------------
  readonly status = computed(() => this.state().status);
  readonly question = computed(() => this.state().question);
  readonly answer = computed(() => this.state().answer);
  readonly error = computed(() => this.state().error);
  readonly elapsedSeconds = computed(() => this.state().elapsedSeconds);
  readonly selectedCitationId = computed(() => this.state().selectedCitationId);
  readonly selectedCitation = computed<Citation | null>(() => {
    const id = this.state().selectedCitationId;
    return this.state().answer?.citations.find((c) => c.id === id) ?? null;
  });

  constructor() {
    // --- reducers ----------------------------------------------------------
    this.ask$
      .pipe(
        switchMap((question) =>
          this.http.post<MultiAgentAnswer>('/agent/ask', { question }).pipe(
            map((answer): AskEvent => ({ kind: 'succeeded', answer })),
            catchError((err: HttpErrorResponse) =>
              of<AskEvent>({ kind: 'failed', error: describeError(err) }),
            ),
            startWith<AskEvent>({ kind: 'started', question }),
          ),
        ),
        takeUntilDestroyed(),
      )
      .subscribe((event) => this.state.update((s) => reduce(s, event)));

    // A ticking clock while a question is in flight (switchMap cancels the
    // previous tick stream whenever a new question starts).
    this.ask$
      .pipe(
        switchMap(() => timer(0, 1000)),
        takeUntilDestroyed(),
      )
      .subscribe((tick) =>
        this.state.update((s) =>
          s.status === 'loading' ? { ...s, elapsedSeconds: tick } : s,
        ),
      );

    this.citationToggled$
      .pipe(takeUntilDestroyed())
      .subscribe((id) =>
        this.state.update((s) => ({
          ...s,
          selectedCitationId: s.selectedCitationId === id ? null : id,
        })),
      );
  }
}

function reduce(state: AskState, event: AskEvent): AskState {
  switch (event.kind) {
    case 'started':
      return {
        ...INITIAL_STATE,
        status: 'loading',
        question: event.question,
      };
    case 'succeeded':
      return { ...state, status: 'success', answer: event.answer, error: null };
    case 'failed':
      return { ...state, status: 'error', error: event.error };
  }
}

function describeError(err: HttpErrorResponse): string {
  const detail = (err.error as { detail?: string } | null)?.detail;
  if (detail) return detail;
  if (err.status === 0) {
    return 'Backend unreachable — is `uvicorn app.main:app --port 8000` running?';
  }
  return `${err.status} ${err.statusText}`;
}
