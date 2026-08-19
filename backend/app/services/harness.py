"""Model Harness managing structured LLM execution, Tenacity retries, and Gemini fallback."""

import json
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.schemas.response import CitationItem
from app.services.retrieval_service import RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger("app.services.harness")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


class HarnessError(Exception):
    """Base exception for model harness errors."""
    pass


class HarnessTransientError(HarnessError):
    """Transient network or timeout failure eligible for Tenacity retry."""
    pass


class HarnessInput(BaseModel):
    """Structured input payload passed to the Model Harness."""

    query: str = Field(..., description="User input query")
    context_text: str = Field(default="", description="Retrieved context passages formatted as text")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunk objects")
    language: str = Field(default="en", description="Target response language")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Trace request ID")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=512, description="Max generated tokens")


class HarnessOutput(BaseModel):
    """Structured output returned from the Model Harness."""

    answer: str = Field(..., description="Grounded response text")
    confidence_score: float = Field(default=0.9, description="Confidence score (0.0 to 1.0)")
    citations: List[CitationItem] = Field(default_factory=list, description="Citations tied to retrieved chunks")
    status: str = Field(default="SUCCESS", description="Execution status: SUCCESS or REFUSED")
    latency_breakdown: Dict[str, float] = Field(default_factory=dict, description="Latency breakdown in ms")
    model_used: str = Field(default="llama-3.1-8b-instant", description="Exact model name used")
    provider_used: str = Field(default="groq", description="Provider used: groq, gemini, or local-fallback")
    is_fallback: bool = Field(default=False, description="True if response came from fallback engine")
    ttft_ms: float = Field(default=0.0, description="Time to first token / generation latency in ms")


SYSTEM_PROMPT = """You are a precise, multilingual AI research assistant powered by a curated knowledge base (MSMARCO-XI).

CRITICAL INSTRUCTIONS:
1. Answer the user question STRICTLY and ONLY using the provided retrieved context passages.
2. If the retrieved context does NOT contain sufficient facts to answer accurately, explicitly respond:
   "I could not find sufficient grounded evidence in MSMARCO-XI to answer accurately."
3. DO NOT invent facts, speculate, or draw upon outside information.
4. Respond in the requested language (e.g. English, Hindi, Telugu, Tamil, Marathi, Bengali, or Hinglish).
5. Always provide citations corresponding to the passage IDs that support your answer.
6. Provide a confidence score between 0.0 and 1.0.

Output format must be a valid JSON object:
{
  "answer": "Your grounded response text here.",
  "confidence_score": 0.95,
  "cited_passage_ids": ["passage_id_1"]
}
"""


class ModelHarness:
    """Enterprise model harness orchestrating Groq primary, Gemini fallback, and local fallback."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        primary_model: str = "llama-3.1-8b-instant",
        fallback_model: str = "gemini-1.5-flash",
        timeout_seconds: float = 6.0,
    ) -> None:
        self.settings = get_settings()
        self.groq_api_key = groq_api_key if groq_api_key is not None else self.settings.GROQ_API_KEY
        self.gemini_api_key = gemini_api_key if gemini_api_key is not None else self.settings.GEMINI_API_KEY
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout_seconds = timeout_seconds

    @property
    def has_groq_credentials(self) -> bool:
        if getattr(self, "_groq_auth_failed", False):
            return False
        return bool(self.groq_api_key and not self.groq_api_key.startswith("your_"))

    @property
    def has_gemini_credentials(self) -> bool:
        if getattr(self, "_gemini_auth_failed", False):
            return False
        return bool(self.gemini_api_key and not self.gemini_api_key.startswith("your_"))

    def _build_groq_payload(self, inp: HarnessInput) -> Dict[str, Any]:
        user_content = f"""Target Language: {inp.language}
User Question: {inp.query}

Retrieved Context:
{inp.context_text if inp.context_text else "[NO RETRIEVED CONTEXT]"}

Please provide your grounded answer in JSON format."""

        return {
            "model": self.primary_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": inp.temperature,
            "max_tokens": inp.max_tokens,
            "response_format": {"type": "json_object"},
        }

    def _build_gemini_payload(self, inp: HarnessInput) -> Dict[str, Any]:
        prompt = f"""{SYSTEM_PROMPT}

Target Language: {inp.language}
User Question: {inp.query}

Retrieved Context:
{inp.context_text if inp.context_text else "[NO RETRIEVED CONTEXT]"}"""

        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": inp.temperature,
                "maxOutputTokens": inp.max_tokens,
                "responseMimeType": "application/json",
            },
        }

    def _synthesize_local_fallback(self, inp: HarnessInput) -> HarnessOutput:
        """Deterministic local grounded synthesis when external APIs are unavailable."""
        if not inp.retrieved_chunks:
            return HarnessOutput(
                answer="I could not find sufficient grounded evidence in MSMARCO-XI to answer accurately.",
                confidence_score=0.0,
                citations=[],
                status="REFUSED",
                latency_breakdown={"llm_ttft": 1.0},
                model_used="local-fallback-engine",
                provider_used="local-fallback",
                is_fallback=True,
                ttft_ms=1.0,
            )

        top_chunk = inp.retrieved_chunks[0]
        # Keep abbreviations such as "Dr." inside the retrieved passage.
        grounded_text = top_chunk.text.strip()
        answer = f"Based on the retrieved knowledge: {grounded_text}"
        citations = [
            CitationItem(
                id=top_chunk.chunk_id,
                text=top_chunk.text[:250],
                score=top_chunk.score,
                metadata={"document_id": top_chunk.document_id, "language": top_chunk.language},
            )
        ]

        return HarnessOutput(
            answer=answer,
            confidence_score=round(min(top_chunk.score, 0.98), 2),
            citations=citations,
            status="SUCCESS",
            latency_breakdown={"llm_ttft": 3.0},
            model_used="local-fallback-engine",
            provider_used="local-fallback",
            is_fallback=True,
            ttft_ms=3.0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, min=0.3, max=1.5),
        retry=retry_if_exception_type(HarnessTransientError),
        reraise=True,
    )
    async def _call_groq_primary(self, inp: HarnessInput) -> HarnessOutput:
        """Execute primary LLM call via Groq with Tenacity retries."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_groq_payload(inp)

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(GROQ_CHAT_URL, headers=headers, json=payload)
                ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                if response.status_code in (401, 403):
                    self._groq_auth_failed = True
                    raise HarnessError(f"Groq authentication failed (HTTP {response.status_code})")

                if response.status_code in (500, 502, 503, 504, 429):
                    raise HarnessTransientError(f"Groq transient server error (HTTP {response.status_code})")

                if response.status_code != 200:
                    raise HarnessError(f"Groq API error (HTTP {response.status_code}): {response.text}")

                res_json = response.json()
                content = res_json["choices"][0]["message"]["content"]
                return self._parse_json_response(
                    content=content,
                    retrieved_chunks=inp.retrieved_chunks,
                    model_name=self.primary_model,
                    provider="groq",
                    ttft_ms=ttft_ms,
                    is_fallback=False,
                )

        except httpx.TimeoutException as e:
            ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.warning("Groq call timed out after %.2f ms (attempt failed)", ttft_ms)
            raise HarnessTransientError(f"Groq timeout: {e}") from e
        except httpx.NetworkError as e:
            logger.warning("Groq network error: %s (attempt failed)", str(e))
            raise HarnessTransientError(f"Groq network error: {e}") from e

    async def _call_gemini_fallback(self, inp: HarnessInput) -> HarnessOutput:
        """Execute secondary fallback LLM call via Google Gemini 1.5 Flash."""
        url = f"{GEMINI_GENERATE_URL}?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        payload = self._build_gemini_payload(inp)

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                if response.status_code != 200:
                    raise HarnessError(f"Gemini API returned HTTP {response.status_code}: {response.text}")

                res_json = response.json()
                candidates = res_json.get("candidates", [])
                if not candidates:
                    raise HarnessError("Gemini returned empty candidates.")

                content = candidates[0]["content"]["parts"][0]["text"]
                return self._parse_json_response(
                    content=content,
                    retrieved_chunks=inp.retrieved_chunks,
                    model_name=self.fallback_model,
                    provider="gemini",
                    ttft_ms=ttft_ms,
                    is_fallback=True,
                )
        except Exception as e:
            logger.error("Gemini fallback failed: %s", str(e))
            raise HarnessError(f"Gemini fallback error: {e}") from e

    def _parse_json_response(
        self,
        content: str,
        retrieved_chunks: List[RetrievedChunk],
        model_name: str,
        provider: str,
        ttft_ms: float,
        is_fallback: bool,
    ) -> HarnessOutput:
        """Parse structured JSON from model and build citation references."""
        try:
            parsed = json.loads(content)
            answer = str(parsed.get("answer", content)).strip()
            confidence = float(parsed.get("confidence_score", 0.85))
            cited_ids = set(parsed.get("cited_passage_ids", []))
        except json.JSONDecodeError:
            answer = content.strip()
            confidence = 0.80
            cited_ids = set()

        citations: List[CitationItem] = []
        for chunk in retrieved_chunks:
            if chunk.chunk_id in cited_ids or not cited_ids:
                citations.append(
                    CitationItem(
                        id=chunk.chunk_id,
                        text=chunk.text[:300],
                        score=chunk.score,
                        metadata={"document_id": chunk.document_id, "language": chunk.language},
                    )
                )

        status = "REFUSED" if "could not find sufficient" in answer.lower() else "SUCCESS"

        return HarnessOutput(
            answer=answer,
            confidence_score=round(confidence, 2),
            citations=citations,
            status=status,
            latency_breakdown={"llm_ttft": ttft_ms},
            model_used=model_name,
            provider_used=provider,
            is_fallback=is_fallback,
            ttft_ms=ttft_ms,
        )

    async def execute(self, inp: HarnessInput) -> HarnessOutput:
        """Main harness entrypoint executing primary provider with fallback chain."""
        # Check if context is completely empty
        if not inp.retrieved_chunks or not inp.context_text.strip():
            return HarnessOutput(
                answer="I could not find sufficient grounded evidence in MSMARCO-XI to answer accurately.",
                confidence_score=0.0,
                citations=[],
                status="REFUSED",
                latency_breakdown={"llm_ttft": 0.0},
                model_used=self.primary_model,
                provider_used="none",
                is_fallback=False,
                ttft_ms=0.0,
            )

        # 1. Attempt Primary: Groq Llama 3.1 8B Instant
        if self.has_groq_credentials:
            try:
                logger.info("Executing Primary LLM (Groq Llama 3.1) for req_id=%s", inp.request_id)
                return await self._call_groq_primary(inp)
            except Exception as e:
                logger.warning("Groq primary failed (%s); transitioning to fallback.", str(e))

        # 2. Attempt Secondary Fallback: Google Gemini 1.5 Flash
        if self.has_gemini_credentials:
            try:
                logger.info("Executing Secondary Fallback LLM (Gemini 1.5 Flash) for req_id=%s", inp.request_id)
                return await self._call_gemini_fallback(inp)
            except Exception as e:
                logger.warning("Gemini secondary fallback failed (%s); transitioning to local synthesis.", str(e))

        # 3. Tertiary Fallback: Local Grounded Synthesis Engine
        logger.info("Executing Local Deterministic Synthesis Engine for req_id=%s", inp.request_id)
        return self._synthesize_local_fallback(inp)


@lru_cache()
def get_model_harness() -> ModelHarness:
    """Return singleton instance of ModelHarness."""
    return ModelHarness()
