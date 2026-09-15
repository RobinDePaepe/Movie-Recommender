import json

import llm_curator


def test_generate_filmweek_drops_duplicate_and_anchor_titles(monkeypatch):
    reply = {
        "anchor": {"title": "Videodrome", "year": 1983},
        "picks": [
            {"category": "historical", "title": "Don't Look Now", "year": 1973},
            {"category": "thematic", "title": "Don't Look Now", "year": 1973},
            {"category": "director", "title": "Videodrome", "year": 1983},
            {"category": "recent", "title": "The Substance", "year": 2024},
        ],
    }
    monkeypatch.setattr(llm_curator.llm_providers, "complete", lambda *args, **kwargs: json.dumps(reply))

    result = llm_curator.generate_filmweek(
        anchor="Videodrome (1983)", provider="ollama", search_enabled=False
    )

    assert [pick["title"] for pick in result["picks"]] == ["Don't Look Now", "The Substance"]
