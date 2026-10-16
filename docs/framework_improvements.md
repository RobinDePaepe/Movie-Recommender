# Architecture Roadmap

Status reviewed on 2026-09-16. This document describes optional structural evolution; `feedback-status.md` is the canonical product roadmap.

Current assessment:

- SQLite and Streamlit remain appropriate for this personal, single-user application.
- Pytest coverage and fixture databases are in place; ranking ground-truth coverage remains open.
- Database rebuilds preserve personal records, but versioned migrations and backups around every mutating workflow remain open.
- LLM-assisted curation is implemented and remains additive; deterministic ranking is still authoritative.
- FastAPI, PostgreSQL and pgvector remain conditional future work, not current requirements.

Your current stack is a good MVP stack. I would not replace SQLite or Streamlit yet—your catalogue is only a few thousand titles, and the recommender logic is already sensibly centred on explainable TF‑IDF/content scoring, Letterboxd signals, feedback, and offline evaluation.

The highest-value additions are:

| Priority | Add | Why |
|---|---|---|
| Now | **SQLAlchemy 2 + Alembic** | Give the database a typed data-access layer and versioned schema migrations before the schema grows further. Alembic can generate a migration draft by comparing your models with the current database schema. [Alembic docs](https://alembic.sqlalchemy.org/en/latest/autogenerate.html) |
| Now | **Pydantic models/settings** | Validate imports, recommendation requests, feedback, and environment configuration such as TMDB keys. It also makes a later API easy to introduce. |
| Now | **pytest + fixture database + ranking regression tests** | Test importer idempotency, metadata matching, “already watched” exclusion, and that known favourite films rank above known dislikes. This is especially important because recommendation changes can look plausible while quietly getting worse. |
| Soon | **A clean service layer** | Move import, enrichment, ranking, and explanation code into reusable services that Streamlit calls. Do this before adding an API; it prevents your UI from becoming the application architecture. |
| Soon | **Structured logs and backups** | Record import counts, failed TMDB matches, recommendation configuration, and errors. Make timestamped SQLite backups before sync/enrichment runs. |
| When you want another client or public deployment | **FastAPI** | Add a small API around the service layer for a mobile/web UI, integrations, or a separate frontend. FastAPI supports typed dependencies and simple post-response background work. [FastAPI dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/), [background tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) |
| When multiple users or concurrent writes matter | **PostgreSQL** | Migrate only at this point. It is worthwhile for accounts, shared watchlists, hosted deployment, stronger concurrency, and richer search—not for raw catalogue size. |
| Optional, later | **pgvector + sentence-transformers** | Add semantic similarity for natural-language requests such as “a reflective, quietly unsettling film like *Arrival*.” `pgvector` keeps embeddings alongside relational data and supports exact or indexed nearest-neighbour search. [pgvector](https://github.com/pgvector/pgvector) |

For recommendation quality, I’d focus more on signals than frameworks:

- Keep your current TF‑IDF model as the reliable baseline.
- Add a second, embedding-based score from overview + genres + keywords + cast/directors. Sentence Transformers is well suited to semantic similarity; at your scale, you can compute similarities directly without a specialised vector database. Its own guidance says a manual embedding search remains practical up to roughly one million corpus items. [Semantic-search guidance](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/README.md)
- Combine scores with a transparent hybrid ranker: personal rating affinity + content similarity + list affinity + recency/novelty + availability.
- Store the score breakdown and explanation for every recommendation. Your README’s existing emphasis on “why” explanations is exactly the right product decision.

A practical target architecture would be:

```text
Streamlit UI
    ↓
Application services
(import / enrich / recommend / evaluate)
    ↓
SQLAlchemy repositories
    ↓
SQLite now → PostgreSQL later

Optional FastAPI layer
    └─ web/mobile clients, scheduled imports, integrations
```

A few things I would deliberately avoid for now:

- A separate vector database, Elasticsearch, Redis, Celery, Docker/Kubernetes, or a React rewrite. None solve your current core problem better than clean data, tests, and a well-calibrated hybrid ranker.
- An LLM orchestration framework as the recommender itself. An LLM can make the interface conversational and generate explanations, but should not replace deterministic ranking and evaluation.
- Moving to Postgres just because it is “production-grade.” SQLite remains a strong fit for a personal, single-user app.

Updated suggested build order:

1. Add ranking ground truth and complete the personal-taste signal work.
2. Introduce versioned migrations and typed validation without rewriting the working SQLite layer wholesale.
3. Move import, enrichment and recommendation orchestration into services; add structured logs and automatic backups.
4. Add semantic embeddings alongside TF‑IDF only after they can be compared against the ground-truth set.
5. Add FastAPI only once the Streamlit UI no longer represents the only client.
6. Move to Postgres—with `pgvector` if embeddings prove useful—when you introduce accounts, shared use or a deployed service.
