from __future__ import annotations

from threading import RLock

from bar_replayer.session import DualReplaySession, ReplaySession

ReplaySessionType = ReplaySession | DualReplaySession


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ReplaySessionType] = {}
        self._lock = RLock()

    def add(self, session: ReplaySessionType) -> ReplaySessionType:
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ReplaySessionType:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError("Replay session not found. Load a dataset to start a session.") from exc
