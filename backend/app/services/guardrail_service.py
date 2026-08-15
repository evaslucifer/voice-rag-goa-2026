"""Four-Tier Guardrail System: Tier 1 Safety, Tier 2 Scope, Tier 3 Relevance, Tier 4 Grounding."""

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.config import get_settings
from app.services.retrieval_service import RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger("app.services.guardrail")

# Tier 1: Prompt Injection & Adversarial Attack Heuristics
PROMPT_INJECTION_PATTERNS = [
    r"(?i)\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|guidelines)",
    r"(?i)\b(system\s+prompt|reveal\s+secret|bypass\s+safety|jailbreak|escape\s+sandbox)\b",
    r"(?i)\b(you\s+are\s+now\s+in\s+developer\s+mode|dan\s+mode|unrestricted\s+mode)\b",
    r"(?i)\b(sudo|admin\s+override|drop\s+database|format\s+drive|rm\s+-rf)\b",
    r"(?i)\b(disable\s+guardrails|ignore\s+safety|hack\s+the\s+system)\b",
]

# Tier 2: Clearly Off-Topic Scope Patterns (Casual chit-chat, weather, personal advice, coding games)
OFF_TOPIC_PATTERNS = [
    r"(?i)^(hi|hello|hey|good\s+morning|good\s+evening|how\s+are\s+you)\b",
    r"(?i)\b(weather\s+today|weather\s+forecast|will\s+it\s+rain\s+tomorrow|weather\s+in)\b",
    r"(?i)\b(write\s+(me\s+)?a\s+.*(game|script)|code\s+a\s+.*game|build\s+a\s+calculator)\b",
    r"(?i)\b(relationship|dating|love)\s+(advice|tips)\b",
    r"(?i)\b(advice\s+on\s+how\s+to\s+talk|should\s+i\s+break\s+up|how\s+to\s+date)\b",
    r"(?i)\b(tell\s+me\s+a\s+joke|make\s+me\s+laugh|sing\s+a\s+song)\b",
]


class GuardrailCheckResult(BaseModel):
    """Result of a guardrail verification tier."""

    passed: bool = Field(..., description="Whether the check passed without violations")
    tier: str = Field(..., description="Tier evaluated: tier1_safety, tier2_scope, tier3_relevance, tier4_grounding")
    stage: str = Field(default="pre_retrieval", description="Pipeline stage: pre_retrieval, retrieval, post_generation")
    action: str = Field(default="PROCEED", description="Action recommended: PROCEED, ABSTAIN, or BLOCK")
    reason: Optional[str] = Field(default=None, description="Diagnostic reason for refusal or blockage")
    safe_response: Optional[str] = Field(default=None, description="Standardized refusal message")


class GuardrailService:
    """Four-Tier Guardrail Enforcement Engine for sub-200ms RAG pipelines."""

    def __init__(self, score_threshold: Optional[float] = None) -> None:
        self.settings = get_settings()
        self.score_threshold = score_threshold if score_threshold is not None else getattr(self.settings, "RETRIEVAL_SCORE_THRESHOLD", 0.65)
        self._injection_regexes = [re.compile(p) for p in PROMPT_INJECTION_PATTERNS]
        self._off_topic_regexes = [re.compile(p) for p in OFF_TOPIC_PATTERNS]

    # --------------------------------------------------------------------------
    # Tier 1: Input Safety Guardrail (Pre-Retrieval)
    # --------------------------------------------------------------------------
    def check_tier1_safety(self, query: str) -> GuardrailCheckResult:
        """Tier 1: Fast regex-based check for prompt injection and malicious inputs."""
        clean_query = query.strip()

        if len(clean_query) < 2:
            return GuardrailCheckResult(
                passed=False,
                tier="tier1_safety",
                stage="pre_retrieval",
                action="ABSTAIN",
                reason="Query is too short to process.",
                safe_response="Please provide a more specific question.",
            )

        if len(clean_query) > 2000:
            return GuardrailCheckResult(
                passed=False,
                tier="tier1_safety",
                stage="pre_retrieval",
                action="BLOCK",
                reason="Query exceeds maximum allowed length (2000 characters).",
                safe_response="Your query exceeds the maximum allowed length. Please shorten your question.",
            )

        for pattern in self._injection_regexes:
            if pattern.search(clean_query):
                logger.warning("Tier 1 Guardrail Triggered (Prompt Injection): '%s'", clean_query[:80])
                return GuardrailCheckResult(
                    passed=False,
                    tier="tier1_safety",
                    stage="pre_retrieval",
                    action="BLOCK",
                    reason="Potential prompt injection or adversarial instruction detected.",
                    safe_response="I cannot execute system instructions. Please ask a factual question related to the knowledge base.",
                )

        return GuardrailCheckResult(passed=True, tier="tier1_safety", stage="pre_retrieval", action="PROCEED")

    # --------------------------------------------------------------------------
    # Tier 2: Domain Scope Guardrail (Pre-Retrieval)
    # --------------------------------------------------------------------------
    def check_tier2_scope(self, query: str) -> GuardrailCheckResult:
        """Tier 2: Check whether query is clearly outside the domain scope of MSMARCO-XI."""
        clean_query = query.strip()

        for pattern in self._off_topic_regexes:
            if pattern.search(clean_query):
                logger.info("Tier 2 Guardrail Triggered (Off-Topic): '%s'", clean_query[:80])
                return GuardrailCheckResult(
                    passed=False,
                    tier="tier2_scope",
                    stage="pre_retrieval",
                    action="ABSTAIN",
                    reason="Query falls outside the factual knowledge base domain scope.",
                    safe_response="This question is outside the scope of the MSMARCO-XI knowledge base. Please ask a factual question.",
                )

        return GuardrailCheckResult(passed=True, tier="tier2_scope", stage="pre_retrieval", action="PROCEED")

    # --------------------------------------------------------------------------
    # Tier 3: Groundedness Relevance Threshold (Post-Retrieval)
    # --------------------------------------------------------------------------
    def check_tier3_relevance(
        self, query: str, retrieved_chunks: List[RetrievedChunk], top_score: float
    ) -> GuardrailCheckResult:
        """Tier 3: Ensure top-1 retrieval similarity meets score threshold (0.65) before invoking LLM."""
        if not retrieved_chunks or top_score < self.score_threshold:
            logger.info(
                "Tier 3 Guardrail Triggered (Low Relevance): query='%s', top_score=%.3f < threshold=%.3f",
                query[:60],
                top_score,
                self.score_threshold,
            )
            return GuardrailCheckResult(
                passed=False,
                tier="tier3_relevance",
                stage="retrieval",
                action="ABSTAIN",
                reason=f"Top similarity score ({top_score:.3f}) below required relevance threshold ({self.score_threshold:.3f}).",
                safe_response="No sufficiently relevant information was found in MSMARCO-XI to answer accurately.",
            )

        return GuardrailCheckResult(passed=True, tier="tier3_relevance", stage="retrieval", action="PROCEED")

    # --------------------------------------------------------------------------
    # Tier 4: Hallucination & Grounding Verification (Post-Generation)
    # --------------------------------------------------------------------------
    def check_tier4_grounding(
        self, query: str, answer: str, context_text: str, confidence_score: float
    ) -> GuardrailCheckResult:
        """Tier 4: Verify that the synthesized answer is strictly grounded in retrieved context."""
        if not answer or not answer.strip():
            return GuardrailCheckResult(
                passed=False,
                tier="tier4_grounding",
                stage="post_generation",
                action="ABSTAIN",
                reason="Generated answer is empty.",
                safe_response="No sufficiently relevant information was found in MSMARCO-XI to answer accurately.",
            )

        # Standard refusal pass-through
        if "could not find sufficient" in answer.lower() or "knowledge base" in answer.lower() and confidence_score == 0.0:
            return GuardrailCheckResult(
                passed=True,
                tier="tier4_grounding",
                stage="post_generation",
                action="ABSTAIN",
                reason="Model abstained correctly due to lack of evidence.",
                safe_response=answer,
            )

        # Lexical grounding & entity check
        context_words = set(re.findall(r"\w+", context_text.lower()))
        answer_words = [w for w in re.findall(r"\w+", answer.lower()) if len(w) > 3]

        if answer_words:
            grounded_count = sum(1 for w in answer_words if w in context_words)
            overlap_ratio = grounded_count / len(answer_words)

            if overlap_ratio < 0.25 and confidence_score < 0.70:
                logger.warning(
                    "Tier 4 Guardrail Triggered (Hallucination): overlap=%.2f, confidence=%.2f for query '%s'",
                    overlap_ratio,
                    confidence_score,
                    query[:60],
                )
                return GuardrailCheckResult(
                    passed=False,
                    tier="tier4_grounding",
                    stage="post_generation",
                    action="ABSTAIN",
                    reason=f"Generated answer has insufficient context grounding overlap ({overlap_ratio:.2f}).",
                    safe_response="No sufficiently relevant information was found in MSMARCO-XI to answer accurately.",
                )

        return GuardrailCheckResult(passed=True, tier="tier4_grounding", stage="post_generation", action="PROCEED")

    # --------------------------------------------------------------------------
    # Backward Compatibility Helper Methods
    # --------------------------------------------------------------------------
    def check_pre_retrieval(self, query: str) -> GuardrailCheckResult:
        """Combined Tier 1 and Tier 2 pre-retrieval validation."""
        t1 = self.check_tier1_safety(query)
        if not t1.passed:
            return t1
        t2 = self.check_tier2_scope(query)
        if not t2.passed:
            return t2
        return GuardrailCheckResult(passed=True, tier="tier1_tier2", stage="pre_retrieval", action="PROCEED")

    def check_retrieval_relevance(
        self, query: str, retrieved_chunks: List[RetrievedChunk], top_score: float
    ) -> GuardrailCheckResult:
        return self.check_tier3_relevance(query, retrieved_chunks, top_score)

    def check_grounding(
        self, query: str, answer: str, context_text: str, confidence_score: float
    ) -> GuardrailCheckResult:
        return self.check_tier4_grounding(query, answer, context_text, confidence_score)


@lru_cache()
def get_guardrail_service() -> GuardrailService:
    """Return singleton instance of GuardrailService."""
    return GuardrailService()
