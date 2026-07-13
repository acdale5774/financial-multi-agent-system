# Frontend (`/fe`)

Angular UI for the Financial Multi-Agent Intelligence System (case-study
requirement 5). One page: ask a question, watch the multi-agent backend
answer it with per-claim citations, charts, and structured tables.

What it surfaces (the requirement's bullets, mapped):

- **Natural-language input** — question box + the five example questions as
  one-click chips.
- **Agent reasoning** — which route the dispatcher chose (and why), every
  executed SQL query with its row count, every document search with its
  filters, and the citation-validation verdict (repairs, stripped markers,
  lint warnings).
- **Citations** — `[S1]`-style markers in the answer render as clickable
  chips; the sidebar lists every cited record (SimFin: ticker · metric ·
  fiscal period · value · source query; documents: title · type · section ·
  snippet). Clicking a chip highlights/expands its record.
- **Charts and tables** — chart PNGs from the backend's `/charts/` plus the
  citation-hydrated tables, each value cell wearing the evidence chip it was
  hydrated from (requirement 4's "every plotted value traces back").

## Architecture — declarative Angular ("Joshua Morony style")

Angular 21, standalone components, zoneless, signals throughout. The feature
follows the declarative pattern:

```
ask/
├── data-access/
│   ├── ask.service.ts      the entire state of the feature:
│   │                         sources   ask$ / citationToggled$ (Subjects the UI pushes into)
│   │                         reducers  RxJS flows declared once in the constructor
│   │                                   (switchMap on questions, error mapping, elapsed timer),
│   │                                   each ending in ONE state.update
│   │                         selectors computed() signals the templates read
│   └── answer.model.ts     types mirroring be/agents/schemas.py
├── ui/                     dumb components: input() signals in, output() events out,
│   │                       OnPush, zero service access, zero state mutation
│   ├── question-input.ts   answer-card.ts (markdown + citation chips)
│   ├── citation-panel.ts   reasoning-panel.ts   artifact-gallery.ts
└── ask-page.ts             smart container: selectors -> inputs, outputs -> sources
```

Rules the code keeps to: components never call `subscribe` (the service's
constructor owns every subscription via `takeUntilDestroyed`); no imperative
state pokes from event handlers — events go into a source Subject and come
back out as derived state; everything the template renders is a signal.

## Running it

```bash
# 1. backend up (see ../be/README.md): Postgres + OPENAI_API_KEY in be/.env
cd ../be && source .venv/bin/activate && uvicorn app.main:app --port 8000

# 2. frontend dev server (proxies /agent, /ask, /charts, /health to :8000)
cd ../fe && npm start        # -> http://localhost:4200
```

Answers take 20–90 s (a reasoning model is routing, querying, and citing);
the UI shows elapsed time while it works.

Notes: answer markdown comes from our own backend and is rendered with
`marked` into `innerHTML` — fine for a local demo; a production deployment
would add sanitization (e.g. DOMPurify) and serve the built app from
`dist/` behind the same origin as the API.
