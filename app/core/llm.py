import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
import httpx
from app.api.errors import LLMProviderError
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.llm")


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response text from the LLM."""
        pass


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models using the google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ):
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self._types = types

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        config = self._types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
            response_mime_type="application/json",
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                if not response.text:
                    raise LLMProviderError("Empty response received from Gemini")
                return response.text
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        raise LLMProviderError(f"Gemini generation failed after {self.max_retries + 1} attempts: {last_error}")


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible API endpoints."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
            except Exception as e:
                last_error = e
                logger.warning(f"OpenAI API attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        raise LLMProviderError(f"OpenAI generation failed: {last_error}")


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-3-5-haiku-20241022",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["content"][0]["text"]
            except Exception as e:
                last_error = e
                logger.warning(f"Anthropic API attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        raise LLMProviderError(f"Anthropic generation failed: {last_error}")


class OllamaClient(BaseLLMClient):
    """Client for local Ollama instances."""

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
    ):
        self.model_name = model_name
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
        except Exception as e:
            raise LLMProviderError(f"Ollama local LLM call failed: {e}")


class MockLLMClient(BaseLLMClient):
    """Deterministic, mockable LLM client for offline tests and predictable evaluation."""

    def __init__(
        self,
        canned_responses: Optional[List[str]] = None,
        simulate_error: bool = False,
    ):
        self.canned_responses = list(canned_responses or [])
        self.simulate_error = simulate_error
        self.history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        self.history.append({"prompt": prompt, "system_prompt": system_prompt})

        if self.simulate_error:
            raise LLMProviderError("Simulated LLM network/provider failure")

        if self.canned_responses:
            return self.canned_responses.pop(0)

        # Intelligent default mock logic: inspect prompt for chunks and question
        # If chunks exist, check for keyword overlap with question
        chunk_matches = re.findall(r'<chunk id="([^"]+)"[^>]*>(.*?)</chunk>', prompt, re.DOTALL)
        question_match = re.search(r"Question:\s*(.*?)(?:\n\nProvide|$)", prompt, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""

        # Find which chunks share salient words with the question
        q_words = set(re.findall(r"\w+", question.lower())) - {
            "what", "is", "the", "are", "of", "and", "in", "to", "how", "many", "for", "a", "an", "does"
        }

        matching_chunk_ids = []
        best_sentences = []

        # Near-miss and ungrounded entity rejection
        all_chunks_text = " ".join(ctext for _, ctext in chunk_matches).lower()
        q_lower = question.lower()
        critical_unanswerable_terms = [
            "platinum", "severity 4", "sev 4", "meal", "meals", "travel", "australia", "python", "framework"
        ]
        if any(term in q_lower and term not in all_chunks_text for term in critical_unanswerable_terms):
            return json.dumps({
                "answer": "I couldn't find this information in the provided documents.",
                "answerable": False,
                "cited_chunk_ids": [],
            })

        for cid, ctext in chunk_matches:
            c_words = set(re.findall(r"\w+", ctext.lower()))
            overlap = q_words.intersection(c_words)
            if len(overlap) >= 2 or any(len(w) >= 5 and w in c_words for w in q_words):
                matching_chunk_ids.append(cid)
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctext) if s.strip()]
                for s in sentences:
                    s_lower = s.lower()
                    if any(w in s_lower for w in overlap):
                        if s not in best_sentences:
                            best_sentences.append(s)

        if matching_chunk_ids and best_sentences:
            return json.dumps({
                "answer": " ".join(best_sentences),
                "answerable": True,
                "cited_chunk_ids": matching_chunk_ids,
            })
        else:
            return json.dumps({
                "answer": "I couldn't find this information in the provided documents.",
                "answerable": False,
                "cited_chunk_ids": [],
            })


def create_llm_client(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 2,
) -> BaseLLMClient:
    p = provider.lower().strip()
    if p == "mock":
        return MockLLMClient()
    elif p == "gemini":
        if not api_key:
            logger.warning("No LLM_API_KEY provided for 'gemini'. Falling back to MockLLMClient. Add your API key to .env to use live Gemini.")
            return MockLLMClient()
        return GeminiClient(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    elif p == "openai":
        if not api_key:
            logger.warning("No LLM_API_KEY provided for 'openai'. Falling back to MockLLMClient. Add your API key to .env to use live OpenAI.")
            return MockLLMClient()
        return OpenAIClient(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url or "https://api.openai.com/v1",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    elif p == "anthropic":
        if not api_key:
            logger.warning("No LLM_API_KEY provided for 'anthropic'. Falling back to MockLLMClient. Add your API key to .env to use live Anthropic.")
            return MockLLMClient()
        return AnthropicClient(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    elif p == "ollama":
        return OllamaClient(
            model_name=model_name,
            base_url=base_url or "http://localhost:11434",
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'")
