"""Streamlit chat view for the shared, framework-neutral application services."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any

import pandas as pd
import streamlit as st

from movie_recommender.application.orchestrator import ChatOrchestrator, ChatResult, parse_library_query
from movie_recommender.application.services import (
    LibraryQueryService,
    RecommendationService,
    ReviewRetriever,
    TasteAnalysisService,
)
from movie_recommender.data import movie_database
from movie_recommender.integrations import llm_providers


class _ConfiguredAnswerProvider:
    def __init__(self, name: str):
        self.name = name
        self.is_local = name == "ollama"

    def complete(self, prompt: str, *, max_tokens: int = 500) -> str:
        return llm_providers.complete(self.name, prompt, use_search=False, max_tokens=max_tokens)


def _make_orchestrator(db_path: str | Path, provider_name: str | None,
                       allow_hosted_personal_data: bool) -> ChatOrchestrator:
    data_loader = lambda: movie_database.load_data_from_db(db_path)
    metadata_loader = lambda: movie_database.load_metadata_from_db(db_path)
    feedback_loader = lambda: movie_database.load_feedback_from_db(db_path)
    return ChatOrchestrator(
        library=LibraryQueryService(data_loader),
        retriever=ReviewRetriever(),
        recommender=RecommendationService(
            data_loader=data_loader, metadata_loader=metadata_loader, feedback_loader=feedback_loader,
        ),
        analysis=TasteAnalysisService(
            data_loader=data_loader, metadata_loader=metadata_loader, feedback_loader=feedback_loader,
        ),
        answer_provider=_ConfiguredAnswerProvider(provider_name) if provider_name else None,
        allow_hosted_personal_data=allow_hosted_personal_data,
    )


def _render_reference(reference: Any, flagged_ids: set[str]) -> None:
    title = reference.title or reference.movie_id
    if reference.year:
        title += f" ({reference.year})"
    label = {
        "personal_opinion": "Personal review",
        "external_metadata": "TMDb metadata",
        "computed_library_fact": "Library record",
    }.get(reference.source_type, reference.source_type.replace("_", " ").title())
    with st.container(border=True):
        st.markdown(f"**{title}** · {label}")
        details = []
        if reference.watched_date:
            date_label = "Watched" if reference.source_type == "computed_library_fact" else "Review date"
            details.append(f"{date_label}: {reference.watched_date}")
        elif reference.source == "tmdb":
            details.append("Source: TMDb")
        if reference.rating is not None:
            details.append(f"Letterboxd rating: {reference.rating:g}/5")
        if reference.content_type:
            details.append(reference.content_type.replace("_", " "))
        if details:
            st.caption(" · ".join(details))
        if reference.excerpt:
            st.write(reference.excerpt[:420] + ("…" if len(reference.excerpt) > 420 else ""))
        if reference.uri:
            st.link_button("Open Letterboxd record", reference.uri, width="content")
        elif reference.tmdb_id:
            st.link_button("Open TMDb record", f"https://www.themoviedb.org/movie/{reference.tmdb_id}", width="content")
        if reference.source_id in flagged_ids:
            st.info("Flagged as mismatched evidence for this session.")
        elif st.button("Flag mismatched evidence", key=f"flag_evidence_{reference.source_id}"):
            flagged_ids.add(reference.source_id)
            st.session_state["movie_chat_flagged_evidence"] = flagged_ids
            st.info("Flagged for this session. This report is not saved to the database.")


def _render_result(result: ChatResult, flagged_ids: set[str]) -> None:
    st.markdown(result.answer)
    if result.warning:
        st.warning(result.warning, title="Some sources were unavailable")
    if result.evidence_path:
        st.caption("Evidence path: " + " · ".join(result.evidence_path))
    if result.provider_notice:
        st.caption(result.provider_notice)

    if result.references:
        with st.expander(f"Supporting sources ({len(result.references)})", expanded=True):
            for reference in result.references:
                _render_reference(reference, flagged_ids)
        if result.route in ("semantic", "mixed") and not any(
            reference.source_type == "personal_opinion" for reference in result.references
        ):
            st.info("No personal Letterboxd review evidence was found. Metadata can describe a film, but it cannot establish your opinion of it.")
    elif result.route in ("semantic", "mixed"):
        st.info("No supporting review or metadata passages were found. Treat this answer as incomplete.")
    elif not result.library_records and result.recommendations is None and result.analysis_result is None:
        st.info("No matching source records were found. Try a film title, a more specific question, or check that the library and retrieval index are current.")

    if result.recommendations is not None:
        ranked = result.recommendations
        if ranked.empty:
            st.info("The deterministic recommender did not find any candidates for those settings.")
        else:
            columns = [column for column in ("Name", "Year", "score", "why", "taste_matches") if column in ranked.columns]
            st.dataframe(ranked[columns].head(10), hide_index=True, width="stretch")
    if isinstance(result.analysis_result, pd.DataFrame):
        if result.analysis_result.empty:
            st.info("There is not enough library data for this analysis yet.")
        else:
            st.dataframe(result.analysis_result.head(20), hide_index=True, width="stretch")


def render_chat_page(db_path: str | Path, *, use_database: bool) -> None:
    st.subheader("Ask your movie library")
    st.caption("Ask about films you watched, your reviews, themes in your taste, or what to watch next.")
    st.info(
        "Answers are read-only. Each result shows where its evidence came from. "
        "Conversation history stays in this browser session and is cleared when you use Clear chat or end the session.",
        title="About this chat",
    )

    if not use_database:
        st.warning("Build the SQLite library before using chat. The current CSV/JSON fallback does not expose the structured service interface.")
        return

    configured_specs = {name: spec for name, spec in llm_providers.PROVIDERS.items() if spec.ready}
    available_providers = ["Off"] + [spec.label for spec in configured_specs.values()]
    labels_to_names = {spec.label: name for name, spec in configured_specs.items()}
    provider_label = st.selectbox(
        "Answer model", available_providers, index=0, key="movie_chat_provider",
        help="Off uses a local, evidence-only response. Ollama runs locally; other providers send the prompt off-device.",
    )
    provider_name = labels_to_names.get(provider_label)
    is_hosted = bool(provider_name and provider_name != "ollama")
    allow_personal = False
    if is_hosted:
        allow_personal = st.checkbox(
            "Allow this hosted provider to receive my personal ratings, reviews, and recommendation results",
            value=False, key="movie_chat_share_personal",
        )
        if not allow_personal:
            st.caption("With this unchecked, personal library facts and Letterboxd excerpts are withheld from the hosted model. Your question is still sent to it.")
        else:
            st.warning("Personal library evidence may be sent to the selected hosted provider when needed to answer.", title="Personal data sharing enabled")
    elif provider_name == "ollama":
        st.caption("Ollama is configured as a local provider. Its URL may point to another machine if OLLAMA_BASE_URL is customized.")

    with st.expander("Search filters", expanded=False):
        title_filter = st.text_input("Limit evidence to a film title", key="movie_chat_title_filter")
        source_types = st.multiselect(
            "Review and metadata sources", ["letterboxd_diary", "letterboxd_rss", "tmdb"],
            default=[], key="movie_chat_sources",
        )

    if st.button("Clear chat", key="movie_chat_clear"):
        st.session_state["movie_chat_messages"] = []
        st.session_state["movie_chat_flagged_evidence"] = set()
        st.rerun()

    st.session_state.setdefault("movie_chat_messages", [])
    st.session_state.setdefault("movie_chat_flagged_evidence", set())
    flagged_ids = st.session_state["movie_chat_flagged_evidence"]
    for message in st.session_state["movie_chat_messages"]:
        with st.chat_message("user"):
            st.write(message["query"])
        with st.chat_message("assistant"):
            _render_result(message["result"], flagged_ids)

    prompt = st.chat_input("Ask about your reviews or movie library", max_chars=1000, key="movie_chat_input")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking your library and review evidence…"):
                try:
                    from movie_recommender.retrieval.repository import RetrievalFilters
                    library_query = replace(parse_library_query(prompt), title=title_filter) if title_filter else None
                    filters = RetrievalFilters(sources=tuple(source_types) if source_types else ())
                    orchestrator = _make_orchestrator(db_path, provider_name, allow_personal)
                    result = orchestrator.ask(prompt, library_query=library_query, retrieval_filters=filters)
                except Exception as exc:
                    # Keep connection/configuration failures visible and actionable in the chat.
                    from movie_recommender.application.orchestrator import ChatResult
                    result = ChatResult(
                        query=prompt, route="library",
                        answer="I couldn't complete this query. Check the library setup and try again.",
                        warning=str(exc),
                    )
            _render_result(result, flagged_ids)
        st.session_state["movie_chat_messages"].append({"query": prompt, "result": result})
