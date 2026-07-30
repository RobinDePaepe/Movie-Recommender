"""Pluggable LLM backends for the Filmweek curator.

Every backend exposes the same contract via `complete()`:

    complete(provider, prompt, use_search, max_tokens) -> str  (raw model text)

The prompt, JSON parsing, and taste digest live in `llm_curator.py` and are
provider-agnostic. Only the transport, auth, and web-search wiring differ here.

Model IDs are env-overridable (`FILMWEEK_<PROVIDER>_MODEL`) because provider
model names change often — the defaults below are current as of 2026-07 and may
need bumping over time.

SDK packages are imported lazily so the app runs without every provider's SDK
installed; a missing package raises a clear, actionable error only when that
provider is actually used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    default_model: str
    supports_search: bool
    api_key_env: tuple  # accepted env var names; empty => no key needed (local)
    base_url_env: Optional[str] = None
    default_base_url: Optional[str] = None
    sdk_hint: str = ""

    def model(self) -> str:
        return os.getenv(f"FILMWEEK_{self.key.upper()}_MODEL", self.default_model)

    def api_key(self) -> Optional[str]:
        for name in self.api_key_env:
            val = os.getenv(name)
            if val:
                return val
        return None

    def base_url(self) -> Optional[str]:
        if self.base_url_env:
            return os.getenv(self.base_url_env, self.default_base_url)
        return self.default_base_url

    @property
    def needs_key(self) -> bool:
        return bool(self.api_key_env)

    @property
    def ready(self) -> bool:
        return (not self.needs_key) or bool(self.api_key())


PROVIDERS: Dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        key="anthropic",
        label="Anthropic (Claude)",
        default_model="claude-opus-4-8",
        supports_search=True,
        api_key_env=("ANTHROPIC_API_KEY",),
        sdk_hint="pip install anthropic",
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI",
        default_model="gpt-5.6",
        supports_search=True,
        api_key_env=("OPENAI_API_KEY",),
        sdk_hint="pip install openai",
    ),
    "gemini": ProviderSpec(
        key="gemini",
        label="Google Gemini",
        default_model="gemini-3.6-flash",
        supports_search=True,
        api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        sdk_hint="pip install google-genai",
    ),
    "openrouter": ProviderSpec(
        key="openrouter",
        label="OpenRouter",
        default_model="anthropic/claude-opus-4-8",
        supports_search=True,  # via the ":online" model suffix / web plugin
        api_key_env=("OPENROUTER_API_KEY",),
        base_url_env="OPENROUTER_BASE_URL",
        default_base_url="https://openrouter.ai/api/v1",
        sdk_hint="pip install openai",
    ),
    "huggingface": ProviderSpec(
        key="huggingface",
        label="Hugging Face",
        # Any conversational model on the router. Optional policy/provider
        # suffixes work too, e.g. "...:cheapest" or "...:groq".
        default_model="openai/gpt-oss-120b",
        supports_search=False,  # router chat endpoint has no built-in web search
        api_key_env=("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
        base_url_env="HUGGINGFACE_BASE_URL",
        default_base_url="https://router.huggingface.co/v1",
        sdk_hint="pip install openai",
    ),
    "ollama": ProviderSpec(
        key="ollama",
        label="Ollama (local)",
        default_model="llama3.1",
        supports_search=False,  # no built-in web search
        api_key_env=(),  # local, no key
        base_url_env="OLLAMA_BASE_URL",
        default_base_url="http://localhost:11434/v1",
        sdk_hint="pip install openai (and run Ollama locally)",
    ),
}

DEFAULT_PROVIDER = os.getenv("FILMWEEK_PROVIDER", "anthropic")


def get_spec(provider: str) -> ProviderSpec:
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise RuntimeError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
    return spec


def supports_search(provider: str) -> bool:
    return get_spec(provider).supports_search


def _require(package: str, spec: ProviderSpec):
    try:
        return __import__(package)
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            f"The '{spec.label}' backend needs a package that isn't installed. "
            f"Install it with: {spec.sdk_hint}"
        ) from exc


# ── Backends ─────────────────────────────────────────────────────────────────

def _complete_anthropic(spec: ProviderSpec, prompt: str, use_search: bool, max_tokens: int) -> str:
    anthropic = _require("anthropic", spec)
    client = anthropic.Anthropic(api_key=spec.api_key())
    tools = [{"type": "web_search_20260209", "name": "web_search"}] if use_search else None
    messages = [{"role": "user", "content": prompt}]
    parts: List[str] = []
    for _ in range(4):  # resume server-side tool loops on pause_turn
        kwargs = {"model": spec.model(), "max_tokens": max_tokens, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = client.messages.create(**kwargs)
        parts.append("\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text"))
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        break
    return "\n".join(p for p in parts if p)


def _complete_openai(spec: ProviderSpec, prompt: str, use_search: bool, max_tokens: int) -> str:
    _require("openai", spec)
    from openai import OpenAI

    client = OpenAI(api_key=spec.api_key())
    kwargs = {"model": spec.model(), "input": prompt, "max_output_tokens": max_tokens}
    if use_search:
        kwargs["tools"] = [{"type": "web_search"}]
    resp = client.responses.create(**kwargs)
    return resp.output_text or ""


def _complete_openai_compatible(spec: ProviderSpec, prompt: str, use_search: bool, max_tokens: int) -> str:
    """OpenAI-compatible Chat Completions (OpenRouter, Ollama, generic gateways)."""
    _require("openai", spec)
    from openai import OpenAI

    client = OpenAI(api_key=spec.api_key() or "not-needed", base_url=spec.base_url())
    model = spec.model()
    # OpenRouter enables web search by appending ":online" to the model slug.
    if use_search and spec.key == "openrouter" and not model.endswith(":online"):
        model = f"{model}:online"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _complete_gemini(spec: ProviderSpec, prompt: str, use_search: bool, max_tokens: int) -> str:
    _require("google", spec)  # google-genai installs as the 'google.genai' namespace
    from google import genai

    client = genai.Client(api_key=spec.api_key())
    kwargs = {"model": spec.model(), "input": prompt}
    if use_search:
        kwargs["tools"] = [{"type": "google_search"}]
    interaction = client.interactions.create(**kwargs)
    return interaction.output_text or ""


_BACKENDS: Dict[str, Callable[[ProviderSpec, str, bool, int], str]] = {
    "anthropic": _complete_anthropic,
    "openai": _complete_openai,
    "gemini": _complete_gemini,
    "openrouter": _complete_openai_compatible,
    "huggingface": _complete_openai_compatible,
    "ollama": _complete_openai_compatible,
}


def complete(provider: str, prompt: str, *, use_search: bool, max_tokens: int) -> str:
    """Run one completion. `use_search` is silently ignored if unsupported."""
    spec = get_spec(provider)
    if use_search and not spec.supports_search:
        use_search = False
    if spec.needs_key and not spec.api_key():
        raise RuntimeError(
            f"No API key found for {spec.label}. Set one of: {', '.join(spec.api_key_env)}."
        )
    return _BACKENDS[spec.key](spec, prompt, use_search, max_tokens)
