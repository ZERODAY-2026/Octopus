"""Ollama inference client supporting streaming response aggregation and health checks."""

from __future__ import annotations

import json
import logging
from typing import Any
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Asynchronous client for interacting with an Ollama instance."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model_name: str = settings.OLLAMA_MODEL,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the LLM client.

        Args:
            base_url: Base URL of the Ollama server (e.g. http://localhost:11434).
            model_name: Target model identifier.
            timeout: Maximum timeout in seconds for HTTP requests.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or lazily create an HTTP client instance."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> LLMClient:
        """Async context manager entry."""
        self._get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Send a generation prompt to Ollama and aggregate the streaming response.

        Args:
            prompt: User/task prompt text.
            system_prompt: Optional system instruction prompt.
            temperature: Sampling temperature (default 0.1 for deterministic legal output).

        Returns:
            The complete concatenated text response from Ollama.

        Raises:
            httpx.HTTPStatusError: If Ollama returns a non-2xx status code.
            httpx.RequestError: If network communication fails.
            RuntimeError: If response parsing fails.
        """
        client = self._get_client()
        url = "/api/generate"

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            # Ask Ollama to constrain output to valid JSON; the pipeline expects it.
            "format": "json",
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "seed": settings.OLLAMA_SEED,
                # Explicit window: Ollama's default is small and silently truncates
                # long prompts (system prompt + excerpts), dropping evidence.
                "num_ctx": settings.OLLAMA_NUM_CTX,
                "num_predict": -1,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        accumulated_tokens: list[str] = []

        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError as decode_err:
                        logger.warning("Failed to decode streaming JSON chunk: %s", decode_err)
                        continue

                    if "error" in chunk_data:
                        raise RuntimeError(f"Ollama error: {chunk_data['error']}")

                    token = chunk_data.get("response", "")
                    accumulated_tokens.append(token)

                    if chunk_data.get("done", False):
                        prompt_tokens = chunk_data.get("prompt_eval_count")
                        if isinstance(prompt_tokens, int) and prompt_tokens >= settings.OLLAMA_NUM_CTX - 64:
                            logger.warning(
                                "Prompt used %d of %d context tokens; it may have been truncated",
                                prompt_tokens,
                                settings.OLLAMA_NUM_CTX,
                            )
                        break

        except httpx.HTTPStatusError as err:
            logger.error("HTTP error during Ollama generation: %s", err)
            raise
        except httpx.RequestError as err:
            logger.error("Network error communicating with Ollama: %s", err)
            raise

        return "".join(accumulated_tokens).strip()

    async def health_check(self) -> bool:
        """Check whether the Ollama server is running and the specified model is available.

        Returns:
            True if Ollama is accessible and the configured model is installed, False otherwise.
        """
        client = self._get_client()
        try:
            response = await client.get("/api/tags")
            if response.status_code != 200:
                logger.warning("Ollama health check returned HTTP %d", response.status_code)
                return False

            data = response.json()
            models = data.get("models", [])
            available_names = [m.get("name", "") for m in models]

            target = self.model_name.lower()
            model_found = any(
                name.lower() == target
                or name.lower().startswith(f"{target}:")
                or target.startswith(f"{name.lower()}:")
                for name in available_names
            )

            if not model_found:
                logger.warning(
                    "Configured model '%s' not found in available Ollama models: %s",
                    self.model_name,
                    available_names,
                )

            return model_found

        except Exception as err:
            logger.warning("Ollama health check failed with error: %s", err)
            return False
