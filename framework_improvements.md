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

My suggested build order:

1. SQLAlchemy + Alembic, Pydantic settings, and automated tests.
2. Make imports/enrichment repeatable and observable; add backups.
3. Add semantic embeddings alongside TF‑IDF and compare them on your existing evaluation page.
4. Add FastAPI only once the Streamlit UI no longer represents the only client.
5. Move to Postgres—with `pgvector` if embeddings prove useful—when you introduce user accounts, shared use, or a deployed service.