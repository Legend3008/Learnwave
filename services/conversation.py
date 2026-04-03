"""
Conversation Memory Manager
Manages multi-turn conversation state with:
  - Session-scoped history
  - Sliding window with summary compression
  - Conversation-aware query reformulation
  - Persistent storage (JSON file-based for simplicity; swap to Redis/Postgres for prod)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DATA_DIR, MAX_CONVERSATION_TURNS, CONVERSATION_SUMMARY_AFTER
from models.schemas import ConversationHistory, ConversationTurn

logger = logging.getLogger(__name__)

SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class ConversationMemory:
    """
    Session-scoped conversation memory with sliding window and summary compression.
    """

    def __init__(self):
        self._active_sessions: dict[str, ConversationHistory] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def create_session(self, doc_ids: Optional[list[str]] = None, title: str = "Untitled notebook") -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        session = ConversationHistory(
            session_id=session_id,
            doc_ids=doc_ids or [],
            title=title,
        )
        self._active_sessions[session_id] = session
        self._save_session(session)
        logger.info("Created session %s with docs %s", session_id[:8], doc_ids)
        return session_id

    def rename_session(self, session_id: str, title: str) -> bool:
        """Rename a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        session.title = title
        self._save_session(session)
        return True

    def get_session(self, session_id: str) -> Optional[ConversationHistory]:
        """Retrieve a session, loading from disk if needed."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # Try loading from disk
        session = self._load_session(session_id)
        if session:
            self._active_sessions[session_id] = session
        return session

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[list] = None,
    ) -> ConversationHistory:
        """
        Add a conversation turn to the session.

        Args:
            session_id: Session identifier.
            role: "user" or "assistant".
            content: Message content.
            citations: Optional citations (for assistant turns).

        Returns:
            Updated ConversationHistory.
        """
        session = self.get_session(session_id)
        if not session:
            # Auto-create session if it doesn't exist
            self.create_session()
            session = self._active_sessions[session_id] = ConversationHistory(
                session_id=session_id
            )

        turn = ConversationTurn(
            role=role,
            content=content,
            citations=citations or [],
        )
        session.turns.append(turn)

        # Apply sliding window if history is too long
        if len(session.turns) > MAX_CONVERSATION_TURNS:
            self._compress_history(session)

        self._save_session(session)
        return session

    def get_recent_history(
        self,
        session_id: str,
        max_turns: int = 10,
    ) -> list[ConversationTurn]:
        """Get the most recent turns for context injection."""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.turns[-max_turns:]

    def get_full_context(self, session_id: str) -> str:
        """Get conversation history as formatted text for prompt injection."""
        session = self.get_session(session_id)
        if not session or not session.turns:
            return ""

        parts = []
        for turn in session.turns[-8:]:  # Last 8 turns
            prefix = "User" if turn.role == "user" else "Assistant"
            parts.append(f"{prefix}: {turn.content[:500]}")

        return "\n".join(parts)

    def delete_session(self, session_id: str):
        """Delete a session from memory and disk."""
        self._active_sessions.pop(session_id, None)
        session_path = SESSIONS_DIR / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
            logger.info("Deleted session %s", session_id[:8])

    def list_sessions(self) -> list[dict]:
        """List all available sessions with title and preview."""
        sessions = []
        for path in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                turns = data.get("turns", [])
                # Extract first user message as preview
                preview = ""
                for t in turns:
                    if t.get("role") == "user":
                        preview = t.get("content", "")[:80]
                        break
                sessions.append({
                    "session_id": data.get("session_id", path.stem),
                    "title": data.get("title", "Untitled notebook"),
                    "created_at": data.get("created_at", ""),
                    "turn_count": len(turns),
                    "doc_ids": data.get("doc_ids", []),
                    "preview": preview,
                })
            except Exception:
                continue
        # Sort by created_at descending (newest first)
        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions

    # ── History Compression ───────────────────────────────────────────────

    def _compress_history(self, session: ConversationHistory):
        """
        Compress older turns into a summary to stay within context limits.
        Keeps the most recent turns and summarizes the rest.
        """
        if len(session.turns) <= CONVERSATION_SUMMARY_AFTER:
            return

        # Keep recent turns, summarize older ones
        keep_count = MAX_CONVERSATION_TURNS // 2
        to_summarize = session.turns[:-keep_count]
        to_keep = session.turns[-keep_count:]

        # Create a simple summary of older turns
        summary_parts = []
        for turn in to_summarize:
            if turn.role == "user":
                summary_parts.append(f"Q: {turn.content[:100]}")
            else:
                summary_parts.append(f"A: {turn.content[:150]}")

        summary_text = "[Summary of earlier conversation]\n" + "\n".join(summary_parts)

        summary_turn = ConversationTurn(
            role="system",
            content=summary_text,
        )

        session.turns = [summary_turn] + to_keep
        logger.info(
            "Compressed session %s: %d turns → %d turns + summary",
            session.session_id[:8],
            len(to_summarize) + len(to_keep),
            len(session.turns),
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_session(self, session: ConversationHistory):
        """Save session to disk as JSON."""
        session_path = SESSIONS_DIR / f"{session.session_id}.json"
        try:
            data = session.model_dump(mode="json")
            session_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error("Failed to save session %s: %s", session.session_id[:8], e)

    def _load_session(self, session_id: str) -> Optional[ConversationHistory]:
        """Load session from disk."""
        session_path = SESSIONS_DIR / f"{session_id}.json"
        if not session_path.exists():
            return None
        try:
            data = json.loads(session_path.read_text())
            return ConversationHistory(**data)
        except Exception as e:
            logger.error("Failed to load session %s: %s", session_id[:8], e)
            return None
