from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import Any

import pandas as pd


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Shadow timestamps must be valid")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _iso(value: Any) -> str:
    return _utc(value).isoformat()


def _finite_price(value: Any, name: str) -> float:
    price = float(value)
    if not isfinite(price):
        raise ValueError(f"Shadow {name} must be finite")
    return price


@dataclass
class ShadowRecord:
    """One hypothetical order and its plain-geometry market outcome."""

    record_id: str
    asset: str
    symbol_code: str
    hypothesis: str
    magic: int
    comment: str
    type: str
    wave: str
    side: str
    arm_time: str
    trigger_close: str
    limit: float
    limit_source: str
    sl: float
    sl_pips: float
    sl_source: str
    tp: float
    placed_at: str
    expires_at: str | None
    state: str = "pending"
    filled_at: str | None = None
    fill_price: float | None = None
    closed_at: str | None = None
    exit_price: float | None = None
    r: float | None = None
    max_fav_r: float = 0.0
    max_adv_r: float = 0.0
    reached_retry: str | None = None
    retry_budget: int = 1
    retry: bool = False
    chain_id: str | None = None
    retry_record_id: str | None = None
    expiry_reason: str | None = None
    retry_r: float = 2.0
    risk_price: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ShadowRecord:
        if not isinstance(value, dict):
            raise TypeError("Shadow record state must be a mapping")
        allowed = {item.name for item in fields(cls)}
        payload = {key: deepcopy(item) for key, item in value.items() if key in allowed}
        return cls(**payload)

    @classmethod
    def from_order(
        cls,
        order: dict[str, Any],
        *,
        asset: str,
        retry_r: float,
    ) -> ShadowRecord:
        if not isinstance(order, dict):
            raise TypeError("Layer 4 order must be a mapping")
        magic = int(order["magic"])
        symbol_code = str(order.get("magic_text", ""))[:3]
        if len(symbol_code) != 3 or not symbol_code.isdigit():
            symbol_code = str(order.get("symbol_code") or "")
        if len(symbol_code) != 3 or not symbol_code.isdigit():
            raise ValueError("Shadow order requires a three-digit symbol code")
        trigger_close = _iso(order["trigger_close"])
        metadata = deepcopy(order.get("metadata") or {})
        retry = bool(order.get("retry", False))
        return cls(
            record_id=str(order["record_id"]),
            asset=str(asset).upper(),
            symbol_code=symbol_code,
            hypothesis=str(order["hypothesis"]),
            magic=magic,
            comment=str(order["comment"]),
            type=str(order["type"]),
            wave=str(order["wave"]),
            side=str(order["side"]),
            arm_time=_iso(order["arm_time"]),
            trigger_close=trigger_close,
            limit=_finite_price(order.get("limit", order.get("entry")), "limit"),
            limit_source=str(order.get("limit_source") or ""),
            sl=_finite_price(order["sl"], "SL"),
            sl_pips=_finite_price(order["sl_pips"], "SL pips"),
            sl_source=str(order.get("sl_source") or ""),
            tp=_finite_price(order["tp"], "TP"),
            placed_at=trigger_close,
            expires_at=_iso(order["expires_at"]) if order.get("expires_at") is not None else None,
            retry_budget=0 if retry else 1,
            retry=retry,
            chain_id=(str(metadata["chain_id"]) if metadata.get("chain_id") is not None else None),
            retry_record_id=(
                str(metadata["retry_record_id"])
                if metadata.get("retry_record_id") is not None
                else None
            ),
            expiry_reason=(
                "old-side order"
                if order.get("expires_at") is not None and metadata.get("old_side")
                else "trigger expiry"
                if order.get("expires_at") is not None
                else None
            ),
            retry_r=float(retry_r),
            metadata=metadata,
        )


class ShadowInventory:
    """Per-symbol pending/open order book, updated once per closed 5m bar."""

    STATE_VERSION = 1
    RETIRED_LIMIT = 10

    def __init__(self, *, asset: str, retry_r: float = 2.0) -> None:
        self.asset = str(asset).strip().upper()
        if not self.asset:
            raise ValueError("Shadow inventory requires an asset")
        self.retry_r = float(retry_r)
        if not isfinite(self.retry_r) or self.retry_r <= 0:
            raise ValueError("retry_r must be a positive finite number")
        self.active: dict[str, ShadowRecord] = {}
        self.retired: list[ShadowRecord] = []

    @staticmethod
    def _event(kind: str, time: str, record: ShadowRecord, **details: Any) -> dict[str, Any]:
        return {
            "kind": kind,
            "time": time,
            "record_id": record.record_id,
            "magic": record.magic,
            "chain_id": record.chain_id,
            "record": record.to_dict(),
            **details,
        }

    def _apply_order_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        now: str,
        events: list[dict[str, Any]],
    ) -> None:
        for update in updates:
            if not isinstance(update, dict):
                raise TypeError("Layer 4 order updates must be mappings")
            chain_id = str(update.get("chain_id") or "")
            if not chain_id:
                continue
            reason = str(update.get("reason") or "hypothesis update")
            for record in self.active.values():
                if record.chain_id != chain_id:
                    continue
                changed: dict[str, Any] = {}
                proposed_expiry = update.get("expires_at")
                if proposed_expiry is not None and record.state == "pending":
                    new_expiry = _iso(proposed_expiry)
                    if record.expires_at is None or _utc(new_expiry) < _utc(record.expires_at):
                        record.expires_at = new_expiry
                        record.expiry_reason = reason
                        changed["expires_at"] = new_expiry
                if update.get("retry_budget") is not None:
                    new_budget = int(update["retry_budget"])
                    if new_budget < 0:
                        raise ValueError("Shadow retry_budget cannot be negative")
                    if new_budget == 0 and record.retry_budget != 0:
                        record.retry_budget = 0
                        changed["retry_budget"] = 0
                if changed:
                    events.append(self._event("order_updated", now, record, reason=reason, changes=changed))

    @staticmethod
    def _fill_price(record: ShadowRecord, bar: dict[str, float]) -> float | None:
        open_price, high, low = bar["open"], bar["high"], bar["low"]
        if record.side == "long":
            if low > record.limit:
                return None
            return open_price if open_price <= record.limit else record.limit
        if record.side == "short":
            if high < record.limit:
                return None
            return open_price if open_price >= record.limit else record.limit
        raise ValueError(f"Unsupported shadow side: {record.side}")

    @staticmethod
    def _risk(record: ShadowRecord) -> float:
        fill = float(record.fill_price)
        actual = abs(fill - record.sl)
        if actual > 1e-12:
            return actual
        planned = abs(record.limit - record.sl)
        if planned <= 1e-12:
            raise ValueError(f"Shadow record {record.record_id} has zero risk")
        return planned

    def _exit_on_bar(self, record: ShadowRecord, bar: dict[str, float]) -> tuple[str, float] | None:
        open_price, high, low = bar["open"], bar["high"], bar["low"]
        if record.side == "long":
            if open_price <= record.sl:
                return "sl", open_price
            if open_price >= record.tp:
                return "tp", open_price
            sl_hit, tp_hit = low <= record.sl, high >= record.tp
        else:
            if open_price >= record.sl:
                return "sl", open_price
            if open_price <= record.tp:
                return "tp", open_price
            sl_hit, tp_hit = high >= record.sl, low <= record.tp
        if sl_hit:
            return "sl", record.sl
        if tp_hit:
            return "tp", record.tp
        return None

    def _advance_open(
        self,
        record: ShadowRecord,
        bar: dict[str, float],
        *,
        now: str,
        events: list[dict[str, Any]],
    ) -> tuple[bool, dict[str, Any] | None]:
        if record.fill_price is None:
            raise ValueError(f"Open shadow record {record.record_id} has no fill price")
        risk = self._risk(record)
        if record.side == "long":
            fav_r = (bar["high"] - record.fill_price) / risk
            adv_r = (record.fill_price - bar["low"]) / risk
        else:
            fav_r = (record.fill_price - bar["low"]) / risk
            adv_r = (bar["high"] - record.fill_price) / risk
        previous_fav, previous_adv = record.max_fav_r, record.max_adv_r
        record.max_fav_r = max(record.max_fav_r, float(fav_r), 0.0)
        record.max_adv_r = max(record.max_adv_r, float(adv_r), 0.0)
        metrics_changed = (
            record.max_fav_r > previous_fav + 1e-12
            or record.max_adv_r > previous_adv + 1e-12
        )

        outcome = self._exit_on_bar(record, bar)
        if outcome is not None:
            state, exit_price = outcome
            record.state = state
            record.closed_at = now
            record.exit_price = float(exit_price)
            direction = 1.0 if record.side == "long" else -1.0
            record.r = direction * (record.exit_price - float(record.fill_price)) / risk
            if state == "tp":
                if record.reached_retry is None and record.max_fav_r >= record.retry_r:
                    record.reached_retry = now
                record.retry_budget = 0
            elif record.reached_retry is None:
                # A bar that hits both the retry threshold and SL resolves at
                # SL; its intrabar order is unknowable and retry threshold is not reached.
                pass
            if metrics_changed:
                events.append(self._event("excursion_updated", now, record))
            events.append(self._event("closed", now, record, outcome=state))
            retry_request = (
                record.to_dict()
                if state == "sl"
                and record.reached_retry is None
                and record.retry_budget == 1
                and not record.retry
                else None
            )
            return True, retry_request

        if record.reached_retry is None and record.max_fav_r >= record.retry_r:
            record.reached_retry = now
            record.retry_budget = 0
            events.append(self._event("retry_reached", now, record, retry_r=record.retry_r))
        if metrics_changed:
            events.append(self._event("excursion_updated", now, record))
        return False, None

    def step(
        self,
        ctx: dict[str, Any],
        *,
        new_orders: list[dict[str, Any]] | None = None,
        order_updates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(ctx, dict) or not ctx.get("time"):
            raise ValueError("ShadowInventory.step requires a context_at close time")
        now = _iso(ctx["time"])
        raw_bar = ctx.get("bar_5m") or {}
        try:
            bar = {name: _finite_price(raw_bar[name], name) for name in ("open", "high", "low", "close")}
        except (KeyError, TypeError) as exc:
            raise ValueError("Shadow inventory requires 5m OHLC on every close") from exc
        if bar["high"] < max(bar["open"], bar["close"]) or bar["low"] > min(bar["open"], bar["close"]):
            raise ValueError("Shadow 5m OHLC range is inconsistent")

        events: list[dict[str, Any]] = []
        retry_requests: list[dict[str, Any]] = []
        released_chain_ids: list[str] = []
        self._apply_order_updates(order_updates or [], now=now, events=events)

        for record in list(self.active.values()):
            if record.state == "pending":
                fill_price = self._fill_price(record, bar)
                if fill_price is None:
                    if record.expires_at is not None and _utc(now) >= _utc(record.expires_at):
                        record.state = "expired"
                        record.closed_at = now
                        events.append(
                            self._event("expired", now, record, reason=record.expiry_reason or "order expiry")
                        )
                        del self.active[record.record_id]
                        if record.chain_id:
                            released_chain_ids.append(record.chain_id)
                    continue
                record.state = "open"
                record.filled_at = now
                record.fill_price = float(fill_price)
                record.risk_price = self._risk(record)
                events.append(self._event("filled", now, record))

            if record.state == "open":
                closed, retry_request = self._advance_open(record, bar, now=now, events=events)
                if closed:
                    del self.active[record.record_id]
                    self.retired.append(record)
                    self.retired = self.retired[-self.RETIRED_LIMIT :]
                    if record.chain_id:
                        released_chain_ids.append(record.chain_id)
                    if retry_request is not None:
                        retry_requests.append(retry_request)

        for order in new_orders or []:
            record = ShadowRecord.from_order(order, asset=self.asset, retry_r=self.retry_r)
            if record.record_id in self.active or any(item.record_id == record.record_id for item in self.retired):
                prior = self.active.get(record.record_id) or next(
                    (item for item in reversed(self.retired) if item.record_id == record.record_id),
                    None,
                )
                prior_summary = (
                    {
                        key: getattr(prior, key)
                        for key in ("state", "type", "wave", "trigger_close", "chain_id", "retry_record_id")
                    }
                    if prior is not None
                    else None
                )
                incoming_summary = {
                    key: getattr(record, key)
                    for key in ("state", "type", "wave", "trigger_close", "chain_id", "retry_record_id")
                }
                raise ValueError(
                    "Duplicate Layer 4 record identity: "
                    f"{record.record_id}; prior={prior_summary}; incoming={incoming_summary}"
                )
            self.active[record.record_id] = record
            events.append(self._event("placed", now, record))

        return {
            "events": events,
            "retry_requests": retry_requests,
            "released_chain_ids": list(dict.fromkeys(released_chain_ids)),
        }

    def finish_retry(
        self,
        record_id: str,
        *,
        attempted: bool,
        accepted: bool,
        record_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        record = next((item for item in reversed(self.retired) if item.record_id == str(record_id)), None)
        if record is None or not attempted:
            if not attempted or record_snapshot is None:
                return None
            record = ShadowRecord.from_dict(record_snapshot)
        record.retry_budget = 0
        return self._event(
            "retry_opened" if accepted else "retry_not_opened",
            record.closed_at or record.trigger_close,
            record,
            reason="eligible SL before retry threshold" if accepted else "hypothesis declined retry",
        )

    def active_records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.active.values()]

    def records(self) -> list[dict[str, Any]]:
        return [*self.active_records(), *(record.to_dict() for record in self.retired)]

    def state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "asset": self.asset,
            "retry_r": self.retry_r,
            "active": [record.to_dict() for record in self.active.values()],
            "retired": [record.to_dict() for record in self.retired],
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", self.STATE_VERSION)) != self.STATE_VERSION:
            raise ValueError("Unsupported ShadowInventory state version")
        if str(state.get("asset", self.asset)).upper() != self.asset:
            raise ValueError("ShadowInventory state asset does not match")
        retry_r = float(state.get("retry_r", self.retry_r))
        if not isfinite(retry_r) or retry_r <= 0:
            raise ValueError("ShadowInventory state retry_r must be positive")
        self.retry_r = retry_r
        self.active = {}
        for raw in state.get("active") or []:
            record = ShadowRecord.from_dict(raw)
            if record.state not in {"pending", "open"}:
                raise ValueError(f"Non-live record in ShadowInventory active state: {record.state}")
            if record.record_id in self.active:
                raise ValueError(f"Duplicate ShadowInventory record: {record.record_id}")
            self.active[record.record_id] = record
        self.retired = [ShadowRecord.from_dict(raw) for raw in state.get("retired") or []]
        if any(record.state not in {"tp", "sl"} for record in self.retired):
            raise ValueError("ShadowInventory retired records must be TP or SL outcomes")
        self.retired = self.retired[-self.RETIRED_LIMIT :]

    @classmethod
    def from_state(cls, state: dict[str, Any], *, asset: str | None = None) -> ShadowInventory:
        inventory = cls(
            asset=str(asset or state.get("asset") or ""),
            retry_r=float(state.get("retry_r", 2.0)),
        )
        inventory.load_state(deepcopy(state))
        return inventory
