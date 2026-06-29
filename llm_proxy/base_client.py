"""
Backend-agnostic LLM client protocol (Reviewer 2 #4, Reviewer 3).

Both GPTClient (OpenAI) and LlamaClient (vLLM/Ollama/HF) implement this
interface so Summarizer / QAHandler stay backend-free.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseLLMClient(Protocol):
    """Minimal chat-style LLM client."""

    model: str

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        ...

    def simple_prompt(self, prompt: str, **kwargs: Any) -> str:
        ...


def get_client_for_model(
    model: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Factory: route a logical model name to the correct backend.

    OpenAI logical names go to ``GPTClient``; everything else
    (currently ``llama-3-8b-instruct``) goes to ``LlamaClient``.

    The Llama backend is imported lazily so a missing optional
    dependency (e.g. ``transformers`` on CPU-only setups) does not
    prevent the OpenAI path from working.
    """
    if model in ("gpt-3.5-turbo", "gpt-4o-mini"):
        from .gpt_client import GPTClient
        return GPTClient(api_key=api_key, model=model)
    if model in ("llama-3-8b-instruct",):
        from .llama_client import LlamaClient
        return LlamaClient(model=model, **kwargs)
    # default: try OpenAI (older logical names)
    from .gpt_client import GPTClient
    return GPTClient(api_key=api_key, model=model)
