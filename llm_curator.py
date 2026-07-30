"""LLM-driven "Filmweek" curator.

Ported from a standalone React prototype (`filmweek.jsx`). Unlike `curator.py`,
which scores movies already in the user's local library, this module asks Claude
to compose a themed week of films around a free-text anchor, using web search to
verify recent releases. It can therefore surface films that are not in the user's
Letterboxd history at all.

The Letterboxd taste digest is built directly from the app's DB frames (no CSV
upload needed, unlike the prototype). Poster/metadata enrichment is left to the
caller via `tmdb_client`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

import llm_providers

logger = logging.getLogger(__name__)

# Category -> (display label, accent colour). Order controls slot ordering.
CATEGORIES: Dict[str, Dict[str, str]] = {
    "historical": {"label": "Film-historical anchor", "accent": "#6E8CA0"},
    "thematic": {"label": "Thematic connection", "accent": "#E8B04B"},
    "director": {"label": "Same director", "accent": "#9B7BB0"},
    "recent": {"label": "Recent · last 2 years", "accent": "#D96A5B"},
    "wildcard": {"label": "Wildcard · lateral match", "accent": "#5FA8A0"},
    "rewatch": {"label": "Rewatch connector", "accent": "#8FA97E"},
}
CATEGORY_ORDER = ["historical", "thematic", "director", "recent", "wildcard", "rewatch"]

_uid_counter = 0


def _uid() -> str:
    global _uid_counter
    _uid_counter += 1
    return f"p{_uid_counter}"


def extract_json(text: str) -> Optional[str]:
    """Return the last balanced top-level {...} object in a blob of text.

    Web search can prepend narration or citation text before the real answer,
    so we scan backwards for the outermost balanced braces.
    """
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return text[i : end + 1]
    return None


@dataclass
class TasteDigest:
    count: int = 0
    mean: Optional[float] = None
    top: List[Dict[str, Any]] = field(default_factory=list)  # {name, year, rating}
    seen_keys: set = field(default_factory=set)  # normalised titles already logged

    @property
    def loaded(self) -> bool:
        return self.count > 0


def _norm_title(value: Any) -> str:
    import re
    import unicodedata

    s = "" if value is None else str(value)
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s)


def build_taste_digest(data: Dict[str, pd.DataFrame]) -> TasteDigest:
    """Build a compact taste signal from the app's rated/watched frames."""
    ratings = data.get("ratings", pd.DataFrame())
    watched = data.get("watched", pd.DataFrame())

    seen_keys: set = set()
    for frame in (ratings, watched, data.get("watchlist", pd.DataFrame())):
        if frame is not None and not frame.empty and "Name" in frame.columns:
            seen_keys |= {_norm_title(n) for n in frame["Name"].dropna()}

    if ratings is None or ratings.empty or "Rating" not in ratings.columns:
        # No ratings, but seen-set may still be useful for "already seen" tags.
        count = 0 if watched is None or watched.empty else len(watched)
        return TasteDigest(count=count, mean=None, top=[], seen_keys=seen_keys)

    rated = ratings.copy()
    rated["Rating"] = pd.to_numeric(rated.get("Rating"), errors="coerce")
    rated = rated.dropna(subset=["Rating"])
    if rated.empty:
        return TasteDigest(count=len(ratings), mean=None, top=[], seen_keys=seen_keys)

    mean = float(rated["Rating"].mean())
    sorted_rated = rated.sort_values("Rating", ascending=False)
    high = sorted_rated[sorted_rated["Rating"] >= 4.5]
    source = high if len(high) >= 8 else sorted_rated
    top = [
        {
            "name": row.get("Name"),
            "year": row.get("Year"),
            "rating": float(row.get("Rating")),
        }
        for _, row in source.head(15).iterrows()
    ]
    return TasteDigest(count=len(ratings), mean=mean, top=top, seen_keys=seen_keys)


def _taste_block(digest: Optional[TasteDigest]) -> str:
    if digest is None or not digest.loaded or not digest.top:
        return ""
    top_str = "; ".join(
        f"{t['name']}{f' ({t['year']})' if pd.notna(t.get('year')) else ''} {t['rating']}/5"
        for t in digest.top
    )
    mean_note = (
        f" Average score: {digest.mean:.2f}/5 (a harsh grader if this is low — weight strong picks accordingly)."
        if digest.mean is not None
        else ""
    )
    return (
        f"\nThe user shared a Letterboxd profile.{mean_note} Films logged: {digest.count}."
        f"\nHighest rated: {top_str}."
        "\nUse this as a strong taste signal. For the rewatch pick, prefer one of these"
        " highly-rated, already-seen films that fits the anchor thematically."
    )


def generate_filmweek(
    *,
    anchor: str,
    seen: bool = True,
    era: str = "1980",
    with_director: bool = True,
    taste_note: str = "",
    digest: Optional[TasteDigest] = None,
    provider: str = "anthropic",
    search_enabled: bool = True,
) -> Dict[str, Any]:
    """Build a themed week around `anchor`. Returns {anchor, picks}.

    `search_enabled` reflects whether the chosen provider can web-verify the
    recent pick; when False the prompt drops the verification instruction.
    Raises RuntimeError with a human-readable message on API/parse failure.
    """
    anchor = anchor.strip()
    if not anchor:
        raise RuntimeError("Enter an anchor film first.")

    search_enabled = search_enabled and llm_providers.supports_search(provider)
    recent_verify = (
        " Use web search to verify the film really exists and that the title, year and director are correct."
        if search_enabled
        else " Only suggest a film you are confident actually exists with the correct title, year and director."
    )

    taste_line = f"Extra taste note from the user (weigh this in): {taste_note.strip()}\n" if taste_note.strip() else ""

    prompt = f"""You are a film curator building a weekly viewing schedule around a single anchor film.

Anchor film: "{anchor}"
Already seen by the user: {"yes" if seen else "no"}
Film-historical pick must be: made before {era}
Include a director pick: {"yes, if meaningful" if with_director else "no"}
{taste_line}{_taste_block(digest)}

Assemble picks (SEPARATE from the anchor film itself):
1. historical — a film made before {era} that connects to the anchor both film-historically and thematically
2. thematic — a film with strong thematic/motivic kinship
{"3. director — another strong film by the same director. Omit if the anchor has no clear director or no strong second film." if with_director else ""}
4. recent — a film from 2024-2026 that fits well.{recent_verify}
   IMPORTANT: if the ANCHOR film itself is from 2024-2026, it already fills the recent role. In that case add NO separate recent pick, and instead give one "wildcard": a less obvious but strong lateral match (different tone/genre/country, same underlying obsession or motif).
{"5. rewatch — a well-known, widely-seen film that connects (separate from the anchor), as a rewatch option." if not seen else "The user has already seen the anchor, so it counts as the rewatch itself: add NO separate rewatch pick."}

Per pick: category, title, year (number), director, and a reason of AT MOST 2 sentences that concretely explains how the film connects to the anchor (theme, motif, style, obsession/craft). No generalities, no marketing language.
{"Search at most once (only to verify the recent film) and do not write any explanation about the search. Answer directly afterwards." if search_enabled else ""}
Respond with ONLY a valid JSON object, no surrounding text and no markdown:
{{
  "anchor": {{"title": "...", "year": 0, "director": "...", "note": "1 sentence on why this film carries the week"}},
  "picks": [
    {{"category": "historical", "title": "...", "year": 0, "director": "...", "reason": "..."}}
  ]
}}"""

    # Reasoning models spend part of the budget thinking before emitting JSON,
    # so keep this generous or the JSON gets truncated.
    text = llm_providers.complete(provider, prompt, use_search=search_enabled, max_tokens=4000)
    json_str = extract_json(text)
    if not json_str:
        snippet = (text or "").strip()[:200] or "(empty response)"
        raise RuntimeError(
            "No usable answer from the model — it did not return valid JSON. "
            f"It returned: {snippet}…\nTry again, add a year to the title, or pick a "
            "non-reasoning instruct model (e.g. meta-llama/Llama-3.3-70B-Instruct)."
        )
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        raise RuntimeError("The model's answer was not valid JSON. Try again.")

    picks = parsed.get("picks", []) or []
    picks.sort(key=lambda p: CATEGORY_ORDER.index(p.get("category")) if p.get("category") in CATEGORY_ORDER else 99)
    for p in picks:
        p["_id"] = _uid()
    return {"anchor": parsed.get("anchor", {}) or {}, "picks": picks}


_ROLE_INSTRUCTIONS = {
    "historical": "a film made before {era} that connects to the anchor both film-historically and thematically",
    "thematic": "a film with strong thematic or motivic kinship to the anchor",
    "director": "another strong film by the same director as the anchor{dir}",
    "recent": "a film from 2024-2026 that connects to the anchor",
    "wildcard": "a less obvious but strong lateral match (different tone, genre or country, but the same underlying obsession or motif)",
    "rewatch": "a well-known, widely-seen film that connects to the anchor{rewatch_hint}",
}


def resimulate_pick(
    *,
    category: str,
    anchor_title: str,
    anchor_year: Any = None,
    anchor_director: str = "",
    era: str = "1980",
    taste_note: str = "",
    digest: Optional[TasteDigest] = None,
    exclude_titles: Optional[List[str]] = None,
    provider: str = "anthropic",
    search_enabled: bool = True,
) -> Dict[str, Any]:
    """Fetch a single alternative pick for one category. Returns a pick dict."""
    exclude = list(dict.fromkeys([anchor_title] + (exclude_titles or [])))
    dir_suffix = f" ({anchor_director})" if anchor_director else ""
    rewatch_hint = ", preferably from the highest-rated already-seen films" if digest and digest.loaded else ""
    instr = _ROLE_INSTRUCTIONS.get(category, "a film that connects to the anchor").format(
        era=era, dir=dir_suffix, rewatch_hint=rewatch_hint
    )
    use_search = category == "recent" and search_enabled and llm_providers.supports_search(provider)
    anchor_meta = ", ".join(str(x) for x in [anchor_year, anchor_director] if x)
    taste_line = f"Taste note: {taste_note.strip()}\n" if taste_note.strip() else ""

    prompt = f"""You are a film curator. Anchor film: "{anchor_title}"{f' ({anchor_meta})' if anchor_meta else ''}.
{taste_line}{_taste_block(digest)}
Give EXACTLY ONE film suggestion for this role: {instr}.
Exclude these already-proposed or rejected titles: {'; '.join(exclude) or '(none)'}.
Pick a different, equally strong alternative — not the most obvious one if it has already been proposed.
{"Verify via web search that title, year and director are correct. Search at most once, write no search explanation." if use_search else ""}
Reason: at most 2 sentences, concrete (theme, motif, style, craft/obsession).

Respond with ONLY JSON, no surrounding text:
{{"category":"{category}","title":"...","year":0,"director":"...","reason":"..."}}"""

    text = llm_providers.complete(provider, prompt, use_search=use_search, max_tokens=2000 if use_search else 1500)
    json_str = extract_json(text)
    if not json_str:
        snippet = (text or "").strip()[:160] or "(empty response)"
        raise RuntimeError(f"No usable answer when re-simulating — got: {snippet}… Try again.")
    try:
        pick = json.loads(json_str)
    except json.JSONDecodeError:
        raise RuntimeError("The re-simulation answer was not valid JSON.")
    pick["category"] = category
    pick["_id"] = _uid()
    return pick
