"""Tests for LLM service."""

from unittest.mock import AsyncMock, patch
import httpx
import pytest
from app.services.llm_service import LLMResponse, LLMService, LLMServiceError
from app.services.retrieval_service import RetrievedChunk


@pytest.mark.asyncio
async def test_llm_local_fallback_when_unconfigured() -> None:
    """Test deterministic grounded fallback when GROQ_API_KEY is not set."""
    service = LLMService(api_key="")
    assert not service.is_live_configured

    chunks = [
        RetrievedChunk(
            document_id="doc_1",
            chunk_id="doc_1_chk_0",
            text="Vikram Sarabhai established ISRO in 1969. The agency launched Aryabhata.",
            score=0.91,
            language="en",
        )
    ]

    res = await service.generate_answer(
        query="Who founded ISRO?",
        retrieved_chunks=chunks,
        context_text=chunks[0].text,
        language="en",
    )

    assert isinstance(res, LLMResponse)
    assert "ISRO" in res.answer or "Vikram Sarabhai" in res.answer
    assert res.confidence_score > 0.8
    assert len(res.citations) == 1
    assert res.citations[0].id == "doc_1_chk_0"


@pytest.mark.asyncio
async def test_llm_live_response_mocked() -> None:
    """Test Groq API call parsing when key is provided."""
    service = LLMService(api_key="gsk_valid_key")
    assert service.is_live_configured

    chunks = [
        RetrievedChunk(
            document_id="doc_1",
            chunk_id="doc_1_chk_0",
            text="The Manhattan project created nuclear weapons.",
            score=0.95,
            language="en",
        )
    ]

    mock_groq_json = {
        "choices": [
            {
                "message": {
                    "content": '{"answer": "The Manhattan Project developed the first atomic weapons during WWII.", "confidence_score": 0.96, "cited_passage_ids": ["doc_1_chk_0"]}'
                }
            }
        ]
    }

    mock_resp = httpx.Response(
        status_code=200,
        json=mock_groq_json,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await service.generate_answer(
            query="What was the Manhattan Project?",
            retrieved_chunks=chunks,
            context_text=chunks[0].text,
            language="en",
        )

        assert "atomic weapons" in res.answer
        assert res.confidence_score == 0.96
        assert len(res.citations) == 1
        assert res.citations[0].id == "doc_1_chk_0"
