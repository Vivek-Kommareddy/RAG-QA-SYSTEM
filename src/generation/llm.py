"""LLM wrappers for different providers.

Supports:
- **OpenAI** – uses the ``openai >= 1.0`` SDK (``chat.completions.create``).
- **Anthropic** – uses the current ``anthropic`` SDK (``messages.create``).
- **Ollama** – sends HTTP requests to a locally running Ollama server.

Use :func:`get_llm` to obtain the correct implementation based on the
application config.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..config import get_settings

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for language model wrappers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response given a complete prompt string.

        Args:
            prompt: The full prompt (system + user content) to send to the model.

        Returns:
            The model's text response.
        """


class OpenAILLM(BaseLLM):
    """LLM implementation using the OpenAI chat completions API (v1+ SDK)."""

    def __init__(self, model_name: str, api_key: Optional[str]) -> None:
        import openai  # type: ignore

        self.model_name = model_name
        self._client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Send *prompt* to OpenAI and return the text of the first choice."""
        logger.debug("Generating with OpenAI model '%s'", self.model_name)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()  # type: ignore[union-attr]


class AnthropicLLM(BaseLLM):
    """LLM implementation using the Anthropic Messages API."""

    def __init__(self, model_name: str, api_key: str | None) -> None:
        import anthropic  # type: ignore

        self.model_name = model_name
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Send *prompt* to Anthropic and return the response text."""
        logger.debug("Generating with Anthropic model '%s'", self.model_name)
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()  # type: ignore[attr-defined]


class OllamaLLM(BaseLLM):
    """LLM implementation using a locally running Ollama server.

    The default endpoint is ``http://localhost:11434``.  Requires the Ollama
    daemon to be running with the desired model already pulled.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434") -> None:
        import httpx  # type: ignore

        self.model_name = model_name
        self._client = httpx.Client(timeout=120.0)
        self._url = f"{base_url}/api/generate"

    def generate(self, prompt: str) -> str:
        """Send *prompt* to Ollama and return the generated text."""
        logger.debug("Generating with Ollama model '%s'", self.model_name)
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        response = self._client.post(self._url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()


def get_llm() -> BaseLLM:
    """Return an LLM instance based on the current application config.

    Raises:
        ValueError: If ``LLM_PROVIDER`` is not one of ``openai``, ``anthropic``,
            or ``ollama``.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()
    model_name = settings.llm_model

    if provider == "openai":
        return OpenAILLM(model_name, settings.openai_api_key)
    if provider == "anthropic":
        return AnthropicLLM(model_name, settings.anthropic_api_key)
    if provider == "ollama":
        return OllamaLLM(model_name)
    raise ValueError(
        f"Unknown LLM provider '{provider}'. Choose from: openai, anthropic, ollama."
    )
