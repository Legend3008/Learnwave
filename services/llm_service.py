"""
LLM Service
Handles all interactions with the LLM (Ollama) including:
  - Prompt construction with system prompt, context, and conversation history
  - Streaming and non-streaming generation
  - Summary generation, question suggestion, concept extraction
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator, Optional

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_TIMEOUT,
    SYSTEM_PROMPT,
)
from models.schemas import (
    Citation,
    ConversationTurn,
    QueryIntent,
)

logger = logging.getLogger(__name__)


class LLMService:
    """LLM interaction layer with prompt engineering and streaming support."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_LLM_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._generate_url = f"{self.base_url}/api/generate"
        self._chat_url = f"{self.base_url}/api/chat"

    # ── Public API ────────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        context: str,
        citations: list[Citation],
        conversation_history: Optional[list[ConversationTurn]] = None,
        intent: QueryIntent = QueryIntent.FACTUAL,
    ) -> str:
        """
        Generate a grounded answer using retrieved context.

        Args:
            query: User's question.
            context: Assembled context from retrieval service.
            citations: Citation metadata for reference.
            conversation_history: Previous turns for multi-turn support.
            intent: Classified query intent.

        Returns:
            Generated answer text.
        """
        prompt = self._build_prompt(query, context, conversation_history, intent)

        logger.info("Generating answer for: '%s' (intent: %s)", query[:80], intent.value)
        response = self._call_generate(prompt)

        return response

    def generate_answer_stream(
        self,
        query: str,
        context: str,
        citations: list[Citation],
        conversation_history: Optional[list[ConversationTurn]] = None,
        intent: QueryIntent = QueryIntent.FACTUAL,
    ) -> Generator[str, None, None]:
        """
        Stream a grounded answer token by token.

        Yields:
            Individual tokens/text chunks.
        """
        prompt = self._build_prompt(query, context, conversation_history, intent)

        logger.info("Streaming answer for: '%s'", query[:80])
        yield from self._call_generate_stream(prompt)

    def generate_summary(self, text: str, filename: str) -> str:
        """Generate an executive summary of a document."""
        prompt = f"""Provide a comprehensive executive summary of the following document: "{filename}"

Document content:
{text[:8000]}

Your summary should include:
1. **Overview**: A 2-3 sentence high-level summary
2. **Key Topics**: The main subjects covered
3. **Key Findings/Points**: The most important information
4. **Structure**: How the document is organized

Format your response in clear markdown."""

        return self._call_generate(prompt)

    def extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts, entities, and definitions from text."""
        prompt = f"""Extract the key concepts, important terms, entities, and definitions from the following text.
Return ONLY a JSON array of strings, no other text.

Text:
{text[:6000]}

Example output: ["Machine Learning", "Neural Network", "Backpropagation"]

JSON array:"""

        response = self._call_generate(prompt)

        # Parse JSON array from response
        try:
            # Find the JSON array in the response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                concepts = json.loads(response[start:end])
                return [str(c) for c in concepts if isinstance(c, str)]
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse concepts JSON, falling back to line split")

        # Fallback: split by newlines
        return [
            line.strip().strip("- ").strip('"')
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 2
        ][:20]

    def suggest_questions(self, context: str, current_query: str = "") -> list[str]:
        """Generate follow-up questions based on context and current query."""
        prompt = f"""Based on the following document content, suggest 5 insightful follow-up questions that a user might want to ask.

{"Current question: " + current_query if current_query else ""}

Document context:
{context[:4000]}

Return ONLY a JSON array of 5 question strings, no other text.

JSON array:"""

        response = self._call_generate(prompt)

        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                questions = json.loads(response[start:end])
                return [str(q) for q in questions if isinstance(q, str)][:5]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback
        return [
            line.strip().strip("- ").strip('"').strip("0123456789. ")
            for line in response.split("\n")
            if line.strip() and "?" in line
        ][:5]

    def reformulate_query(
        self, query: str, history: list[ConversationTurn]
    ) -> str:
        """
        Reformulate a follow-up question as a standalone query
        using conversation history for context.
        """
        if not history:
            return query

        history_text = "\n".join(
            f"{turn.role}: {turn.content[:200]}" for turn in history[-4:]
        )

        prompt = f"""Given the conversation history below, reformulate the follow-up question as a complete, standalone question that can be understood without the conversation context.

Conversation history:
{history_text}

Follow-up question: {query}

Standalone question:"""

        reformulated = self._call_generate(prompt).strip()

        # Sanity check: if the LLM returned garbage, use the original
        if len(reformulated) < 5 or len(reformulated) > len(query) * 3:
            return query

        logger.info("Reformulated query: '%s' → '%s'", query[:50], reformulated[:50])
        return reformulated

    # ── Health Check ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the Ollama LLM service is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Prompt Construction ───────────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[list[ConversationTurn]] = None,
        intent: QueryIntent = QueryIntent.FACTUAL,
    ) -> str:
        """Build the full prompt with system instructions, context, and history."""
        parts = [SYSTEM_PROMPT]

        # Add intent-specific instructions
        intent_instructions = self._get_intent_instructions(intent)
        if intent_instructions:
            parts.append(f"\nSpecial instructions for this query:\n{intent_instructions}")

        # Add conversation history
        if conversation_history:
            parts.append("\n--- Conversation History ---")
            for turn in conversation_history[-6:]:  # Last 6 turns
                parts.append(f"{turn.role.capitalize()}: {turn.content[:500]}")
            parts.append("--- End History ---\n")

        # Add retrieved context
        parts.append("--- Retrieved Context ---")
        parts.append(context)
        parts.append("--- End Context ---\n")

        # Add the user query
        parts.append(f"User Question: {query}")
        parts.append("\nAnswer:")

        return "\n".join(parts)

    def _get_intent_instructions(self, intent: QueryIntent) -> str:
        """Get intent-specific prompt additions."""
        if intent == QueryIntent.SUMMARY:
            return "Provide a comprehensive summary. Organize by key themes and topics."
        elif intent == QueryIntent.COMPARISON:
            return "Structure your answer as a comparison. Use a table or side-by-side format where helpful."
        elif intent == QueryIntent.DEFINITION:
            return "Provide a clear, concise definition followed by explanation and examples from the context."
        elif intent == QueryIntent.ANALYSIS:
            return "Provide an analytical, reasoned response. Consider multiple perspectives from the context."
        return ""

    # ── Ollama API Calls ──────────────────────────────────────────────────

    def _call_generate(self, prompt: str, max_retries: int = 3) -> str:
        """Call Ollama generate API (non-streaming)."""
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self._generate_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                return data.get("response", "")

            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(
                    "LLM request timed out (attempt %d/%d)", attempt, max_retries
                )
            except requests.exceptions.ConnectionError:
                last_error = f"Cannot connect to Ollama at {self.base_url}"
                logger.warning(
                    "Cannot connect to Ollama (attempt %d/%d)", attempt, max_retries
                )
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}"
                logger.warning(
                    "Ollama HTTP error (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    last_error,
                )
            except Exception as e:
                last_error = str(e)
                logger.error("LLM error: %s", last_error)

            if attempt < max_retries:
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"LLM generation failed after {max_retries} attempts: {last_error}"
        )

    def _call_generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Call Ollama generate API with streaming."""
        try:
            response = requests.post(
                self._generate_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                },
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error("Streaming generation error: %s", e)
            yield f"\n\n[Error: {e}]"
