from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trigger:
    """A Layer 4 hypothesis trigger before shared order geometry is applied."""

    hypothesis: str
    trade_type: str
    wave: str
    side: str
    trigger_close: str
    entry_level: float
    sl_reference: float | None
    arm_time: str
    expires_at: str | None
    retry: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "type": self.trade_type,
            "wave": self.wave,
            "side": self.side,
            "trigger_close": self.trigger_close,
            "entry_level": float(self.entry_level),
            "sl_reference": float(self.sl_reference) if self.sl_reference is not None else None,
            "arm_time": self.arm_time,
            "expires_at": self.expires_at,
            "retry": bool(self.retry),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Trigger:
        return cls(
            hypothesis=str(value["hypothesis"]),
            trade_type=str(value["type"]),
            wave=str(value["wave"]),
            side=str(value["side"]),
            trigger_close=str(value["trigger_close"]),
            entry_level=float(value["entry_level"]),
            sl_reference=(float(value["sl_reference"]) if value.get("sl_reference") is not None else None),
            arm_time=str(value["arm_time"]),
            expires_at=(str(value["expires_at"]) if value.get("expires_at") is not None else None),
            retry=bool(value.get("retry", False)),
            metadata=deepcopy(value.get("metadata") or {}),
        )


class Hypothesis(ABC):
    """One Layer 4 trade idea with independent arms, chains and retry searches."""

    id: str
    code: int

    @abstractmethod
    def step(self, ctx: dict[str, Any], core_view: dict[str, Any]) -> list[Trigger]:
        """Advance this hypothesis once at a 5m close and return new triggers."""

    @abstractmethod
    def open_retry(self, record: dict[str, Any]) -> bool:
        """Open an eligible post-SL re-entry search from a shadow record."""

    def register_order(self, order: dict[str, Any]) -> None:
        """Remember an emitted order while its shadow record is live, if needed."""

    def release_order(self, chain_id: str) -> None:
        """Forget a triggered chain once its shadow record is expired or retired."""

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Return a JSON-safe hypothesis snapshot."""

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot previously returned by ``state()``."""
