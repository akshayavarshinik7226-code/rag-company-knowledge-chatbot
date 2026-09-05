"""Provider selection for the chat model, configured only through .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent


def get_llm(provider_name: str | None = None):
    """Return the configured LangChain chat model.

    Set LLM_PROVIDER=ollama or LLM_PROVIDER=openai in .env.  Model names and
    the optional Ollama base URL can also be overridden there.
    """
    load_dotenv(BASE_DIR / ".env")
    provider = (provider_name or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise RuntimeError("Ollama support requires the `langchain-ollama` package.") from error
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=0,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY in the .env file."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError("OpenAI support requires the `langchain-openai` package.") from error
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=0,
        )

    raise RuntimeError("Unsupported LLM_PROVIDER. Use `ollama` or `openai`.")
