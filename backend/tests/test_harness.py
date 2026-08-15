"""Tests for ModelHarness (Groq primary, Tenacity retry, Gemini fallback, and local fallback)."""

import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from app.services.harness import HarnessInput, HarnessOutput, HarnessTransientError, ModelHarness
from app.services.retrieval_service import RetrievedChunk


@pytest.mark.asyncio
async def test_harness_local_fallback_when_unconfigured() -> None:
    """Test deterministic local synthesis when no API keys are provided."""
    harness = ModelHarness(groq_api_key="", gemini_api_key="")
    chunk = RetrievedChunk(
        document_id="doc_1",
        chunk_id="doc_1_chk_0",
        text="Vikram Sarabhai established ISRO in 1969. The headquarters is in Bengaluru.",
        score=0.91,
        language="en",
    )
    inp = HarnessInput(
        query="Who founded ISRO?",
        context_text=chunk.text,
        retrieved_chunks=[chunk],
        language="en",
        request_id="test-req-harness-1",
    )

    out: HarnessOutput = await harness.execute(inp)

    assert out.status == "SUCCESS"
    assert out.is_fallback is True
    assert out.provider_used == "local-fallback"
    assert "ISRO" in out.answer or "Vikram Sarabhai" in out.answer
    assert out.confidence_score > 0.8
    assert len(out.citations) == 1
    assert out.citations[0].id == "doc_1_chk_0"


@pytest.mark.asyncio
async def test_harness_groq_primary_success() -> None:
    """Test successful primary Groq LLM execution."""
    harness = ModelHarness(groq_api_key="gsk_valid_key", gemini_api_key="")
    chunk = RetrievedChunk(
        document_id="doc_1",
        chunk_id="doc_1_chk_0",
        text="The Manhattan Project was led by the United States with UK support.",
        score=0.95,
        language="en",
    )
    inp = HarnessInput(
        query="What was the Manhattan Project?",
        context_text=chunk.text,
        retrieved_chunks=[chunk],
        language="en",
        request_id="test-req-groq-1",
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "The Manhattan Project was a nuclear research initiative during WWII.", "confidence_score": 0.96, "cited_passage_ids": ["doc_1_chk_0"]}'
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        out = await harness.execute(inp)

        assert out.status == "SUCCESS"
        assert out.provider_used == "groq"
        assert out.is_fallback is False
        assert "nuclear research" in out.answer
        assert out.confidence_score == 0.96
        assert len(out.citations) == 1


@pytest.mark.asyncio
async def test_harness_gemini_fallback_on_groq_failure() -> None:
    """Test automatic transition to Gemini 1.5 Flash fallback when Groq primary fails."""
    harness = ModelHarness(groq_api_key="gsk_failing_key", gemini_api_key="AIzaSy_valid_gemini_key")
    chunk = RetrievedChunk(
        document_id="doc_1",
        chunk_id="doc_1_chk_0",
        text="Photosynthesis converts solar light into chemical energy in green plants.",
        score=0.89,
        language="en",
    )
    inp = HarnessInput(
        query="How does photosynthesis work?",
        context_text=chunk.text,
        retrieved_chunks=[chunk],
        language="en",
        request_id="test-req-fallback-1",
    )

    # Groq returns 500 server error, Gemini returns 200 success
    groq_err_resp = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    gemini_resp = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"answer": "Photosynthesis transforms sunlight into chemical energy.", "confidence_score": 0.92, "cited_passage_ids": ["doc_1_chk_0"]}'
                            }
                        ]
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"),
    )

    async def mock_post_dispatch(url, *args, **kwargs):
        if "groq.com" in str(url):
            return groq_err_resp
        else:
            return gemini_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post_dispatch):
        out = await harness.execute(inp)

        assert out.status == "SUCCESS"
        assert out.provider_used == "gemini"
        assert out.is_fallback is True
        assert "Photosynthesis" in out.answer
        assert out.confidence_score == 0.92
