"""Re-embed existing document chunks in place (no re-chunking).

Used after switching the embedding provider/model (see core/config.py and
db/schema.sql — re-applying the schema retypes the vector column and NULLs
the old vectors, since embeddings are not comparable across models):

    python -m ingestion.reembed            # embed every chunk missing a vector
    python -m ingestion.reembed --all      # clear all vectors first, then embed

Chunk rows (content, metadata, ids) are untouched — only `embedding` is
updated — so citations, the lexical index, and the benchmark's ground-truth
labels are unaffected. Each batch is committed as it completes and only
`embedding IS NULL` rows are selected, so an interrupted run resumes where
it left off.
"""

from __future__ import annotations

import time


def reembed(*, batch_size: int = 256, database_url: str | None = None) -> int:
    """Embed every chunk whose vector is NULL; returns the number embedded.

    Transient API failures (rate limits, timeouts) retry with exponential
    backoff; a batch is committed only after its vectors are written, so no
    partial batches persist.
    """
    import openai
    from pgvector.psycopg import register_vector

    from db.connection import get_connection
    from ingestion.embeddings import get_embedder

    embedder = get_embedder()
    done = 0
    started = time.monotonic()

    with get_connection(database_url) as conn:
        register_vector(conn)
        total = conn.execute(
            "SELECT count(*) AS n FROM document_chunks WHERE embedding IS NULL"
        ).fetchone()["n"]
        print(
            f"{total} chunks to embed with {type(embedder).__name__} "
            f"({embedder.dimension}-d), batch size {batch_size}"
        )

        while True:
            # Committed batches drop out of the predicate, so a plain LIMIT
            # walks the remaining work without cursor bookkeeping.
            rows = conn.execute(
                """
                SELECT id, content FROM document_chunks
                WHERE embedding IS NULL
                ORDER BY id
                LIMIT %(n)s
                """,
                {"n": batch_size},
            ).fetchall()
            if not rows:
                break

            for attempt in range(6):
                try:
                    vectors = embedder.embed([r["content"] for r in rows])
                    break
                except (
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                ) as exc:
                    wait = 2**attempt
                    print(f"  transient API error ({type(exc).__name__}); retrying in {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError("embedding API kept failing after 6 attempts; re-run to resume")

            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE document_chunks SET embedding = %s WHERE id = %s",
                    [(vectors[i], r["id"]) for i, r in enumerate(rows)],
                )
            conn.commit()
            done += len(rows)
            rate = done / max(time.monotonic() - started, 1e-9)
            print(f"  {done}/{total} embedded ({rate:.0f} chunks/s)")

    print(f"Done. {done} chunks embedded in {time.monotonic() - started:.0f}s.")
    return done


def main(argv: list[str] | None = None) -> None:
    import argparse

    from db.connection import get_connection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--all",
        action="store_true",
        help="re-embed every chunk (clears all existing vectors first)",
    )
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)

    if args.all:
        with get_connection(args.database_url) as conn:
            conn.execute("UPDATE document_chunks SET embedding = NULL")
            conn.commit()
        print("cleared existing vectors (--all)")

    reembed(batch_size=args.batch_size, database_url=args.database_url)


if __name__ == "__main__":
    main()
