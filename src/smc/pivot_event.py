from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


PivotMode = Literal["conservative", "aggressive"]
PivotSide = Literal[1, -1]


@dataclass
class PivotEvent:
    event_code: str
    event_name: str
    event_group: str
    tier: str
    event_timestamp: pd.Timestamp
    pivot_timestamp: pd.Timestamp
    pivot_price: float
    pivot_side: PivotSide
    mode: PivotMode
    survival_bars: int
    source_tier: str | None = None
    source_event_code: str | None = None
    previous_same_side_timestamp: pd.Timestamp | None = None
    previous_same_side_price: float | None = None
    confirmation_timestamp: pd.Timestamp | None = None
    confirmation_event_timestamp: pd.Timestamp | None = None
    confirmation_price: float | None = None
    relation: str | None = None
    timeframe: str | None = None
    source_bar_indexes: list[int] = field(default_factory=list)
    source_candle_refs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        return ":".join(
            [
                self.event_group,
                self.event_code,
                self.timeframe or "NA",
                self.event_timestamp.isoformat(),
                self.pivot_timestamp.isoformat(),
                f"{float(self.pivot_price):.6f}",
            ]
        )

    @property
    def causal_available_at(self) -> pd.Timestamp:
        return self.event_timestamp

    @property
    def anchor_timestamp(self) -> pd.Timestamp:
        return self.pivot_timestamp

    @property
    def price(self) -> float:
        return self.pivot_price

    @property
    def side(self) -> str:
        return "high" if self.pivot_side == 1 else "low"

    @property
    def direction(self) -> str:
        return self.side

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "eventId": self.event_id,
            "eventCode": self.event_code,
            "eventName": self.event_name,
            "eventGroup": self.event_group,
            "timeframe": self.timeframe,
            "tier": self.tier,
            "side": self.side,
            "direction": self.direction,
            "eventTimestamp": self.event_timestamp,
            "causalAvailableAt": self.causal_available_at,
            "anchorTimestamp": self.anchor_timestamp,
            "pivotTimestamp": self.pivot_timestamp,
            "price": self.price,
            "pivotPrice": self.pivot_price,
            "pivotSide": self.pivot_side,
            "mode": self.mode,
            "survivalBars": self.survival_bars,
            "sourceBarIndexes": list(self.source_bar_indexes),
            "sourceCandleRefs": list(self.source_candle_refs),
            "metadata": dict(self.metadata),
        }
        if self.source_tier is not None:
            payload["sourceTier"] = self.source_tier
        if self.source_event_code is not None:
            payload["sourceEventCode"] = self.source_event_code
        if self.previous_same_side_timestamp is not None:
            payload["previousSameSideTimestamp"] = self.previous_same_side_timestamp
        if self.previous_same_side_price is not None:
            payload["previousSameSidePrice"] = self.previous_same_side_price
        if self.confirmation_timestamp is not None:
            payload["confirmationTimestamp"] = self.confirmation_timestamp
        if self.confirmation_event_timestamp is not None:
            payload["confirmationEventTimestamp"] = self.confirmation_event_timestamp
        if self.confirmation_price is not None:
            payload["confirmationPrice"] = self.confirmation_price
        if self.relation is not None:
            payload["relation"] = self.relation
        return payload


@dataclass
class _Candidate:
    side: PivotSide
    price: float
    timestamp: pd.Timestamp
    index: int
    source_candle_ref: dict[str, Any]
    age: int = 0
    broken: bool = False


def _iso(ts: pd.Timestamp | None) -> str | None:
    return ts.isoformat() if ts is not None else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    return pd.Timestamp(value)


def pivot_event_to_state(event: PivotEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "event_code": event.event_code,
        "event_name": event.event_name,
        "event_group": event.event_group,
        "tier": event.tier,
        "event_timestamp": event.event_timestamp.isoformat(),
        "pivot_timestamp": event.pivot_timestamp.isoformat(),
        "pivot_price": event.pivot_price,
        "pivot_side": event.pivot_side,
        "mode": event.mode,
        "survival_bars": event.survival_bars,
        "source_tier": event.source_tier,
        "source_event_code": event.source_event_code,
        "previous_same_side_timestamp": _iso(event.previous_same_side_timestamp),
        "previous_same_side_price": event.previous_same_side_price,
        "confirmation_timestamp": _iso(event.confirmation_timestamp),
        "confirmation_event_timestamp": _iso(event.confirmation_event_timestamp),
        "confirmation_price": event.confirmation_price,
        "relation": event.relation,
        "timeframe": event.timeframe,
        "source_bar_indexes": list(event.source_bar_indexes),
        "source_candle_refs": list(event.source_candle_refs),
        "metadata": dict(event.metadata),
    }


def pivot_event_from_state(state: dict[str, Any] | None) -> PivotEvent | None:
    if not state:
        return None
    return PivotEvent(
        event_code=str(state["event_code"]),
        event_name=str(state["event_name"]),
        event_group=str(state["event_group"]),
        tier=str(state["tier"]),
        event_timestamp=pd.Timestamp(state["event_timestamp"]),
        pivot_timestamp=pd.Timestamp(state["pivot_timestamp"]),
        pivot_price=float(state["pivot_price"]),
        pivot_side=1 if int(state["pivot_side"]) == 1 else -1,
        mode=str(state["mode"]),  # type: ignore[arg-type]
        survival_bars=int(state["survival_bars"]),
        source_tier=state.get("source_tier"),
        source_event_code=state.get("source_event_code"),
        previous_same_side_timestamp=_timestamp(state.get("previous_same_side_timestamp")),
        previous_same_side_price=(
            float(state["previous_same_side_price"])
            if state.get("previous_same_side_price") is not None
            else None
        ),
        confirmation_timestamp=_timestamp(state.get("confirmation_timestamp")),
        confirmation_event_timestamp=_timestamp(state.get("confirmation_event_timestamp")),
        confirmation_price=(
            float(state["confirmation_price"]) if state.get("confirmation_price") is not None else None
        ),
        relation=state.get("relation"),
        timeframe=state.get("timeframe"),
        source_bar_indexes=[int(idx) for idx in state.get("source_bar_indexes", [])],
        source_candle_refs=[dict(ref) for ref in state.get("source_candle_refs", [])],
        metadata=dict(state.get("metadata") or {}),
    )


def _candidate_to_state(candidate: _Candidate) -> dict[str, Any]:
    return {
        "side": candidate.side,
        "price": candidate.price,
        "timestamp": candidate.timestamp.isoformat(),
        "index": candidate.index,
        "source_candle_ref": dict(candidate.source_candle_ref),
        "age": candidate.age,
        "broken": candidate.broken,
    }


def _candidate_from_state(state: dict[str, Any]) -> _Candidate:
    return _Candidate(
        side=1 if int(state["side"]) == 1 else -1,
        price=float(state["price"]),
        timestamp=pd.Timestamp(state["timestamp"]),
        index=int(state["index"]),
        source_candle_ref=dict(state.get("source_candle_ref") or {}),
        age=int(state.get("age", 0)),
        broken=bool(state.get("broken", False)),
    )


class PivotEventEngine:
    """
    Pivot engine based on tiered compression.

    A candle high/low becomes STR when it remains unbroken for N future candles.
    ITR compresses confirmed STR extremes.
    LTR compresses confirmed ITR extremes.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.mode: PivotMode = str(cfg.get("mode", "conservative")).strip().lower().rstrip(".")  # type: ignore[assignment]
        if self.mode not in {"conservative", "aggressive"}:
            raise ValueError(f"Unsupported pivot event mode: {self.mode}")
        self.timeframe: str | None = cfg.get("timeframe")

        self.str_bars = int(cfg.get("str_bars", cfg.get("survival_bars", 1)))
        if self.str_bars < 1:
            raise ValueError("str_bars must be >= 1")

        compute = cfg.get("compute", cfg.get("tiers", {}))
        self.str_enabled = bool(compute.get("str", True))
        self.itr_enabled = self.str_enabled and bool(compute.get("itr", False))
        self.ltr_enabled = self.itr_enabled and bool(compute.get("ltr", False))

        self._candidates: list[_Candidate] = []
        self._bar_index: int = -1
        self._last_confirmed_side: PivotSide | None = None
        self._last_str_high: PivotEvent | None = None
        self._last_str_low: PivotEvent | None = None
        self._last_itr_side: PivotSide | None = None
        self._last_itr_high: PivotEvent | None = None
        self._last_itr_low: PivotEvent | None = None
        self._last_ltr_side: PivotSide | None = None
        self._last_ltr_high: PivotEvent | None = None
        self._last_ltr_low: PivotEvent | None = None

    def export_state(self) -> dict[str, Any]:
        return {
            "candidates": [_candidate_to_state(candidate) for candidate in self._candidates],
            "bar_index": self._bar_index,
            "last_confirmed_side": self._last_confirmed_side,
            "last_str_high": pivot_event_to_state(self._last_str_high),
            "last_str_low": pivot_event_to_state(self._last_str_low),
            "last_itr_side": self._last_itr_side,
            "last_itr_high": pivot_event_to_state(self._last_itr_high),
            "last_itr_low": pivot_event_to_state(self._last_itr_low),
            "last_ltr_side": self._last_ltr_side,
            "last_ltr_high": pivot_event_to_state(self._last_ltr_high),
            "last_ltr_low": pivot_event_to_state(self._last_ltr_low),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._candidates = [_candidate_from_state(item) for item in state.get("candidates", [])]
        self._bar_index = int(state.get("bar_index", -1))
        last_confirmed_side = state.get("last_confirmed_side")
        self._last_confirmed_side = (
            1 if int(last_confirmed_side) == 1 else -1
        ) if last_confirmed_side is not None else None
        self._last_str_high = pivot_event_from_state(state.get("last_str_high"))
        self._last_str_low = pivot_event_from_state(state.get("last_str_low"))
        last_itr_side = state.get("last_itr_side")
        self._last_itr_side = (1 if int(last_itr_side) == 1 else -1) if last_itr_side is not None else None
        self._last_itr_high = pivot_event_from_state(state.get("last_itr_high"))
        self._last_itr_low = pivot_event_from_state(state.get("last_itr_low"))
        last_ltr_side = state.get("last_ltr_side")
        self._last_ltr_side = (1 if int(last_ltr_side) == 1 else -1) if last_ltr_side is not None else None
        self._last_ltr_high = pivot_event_from_state(state.get("last_ltr_high"))
        self._last_ltr_low = pivot_event_from_state(state.get("last_ltr_low"))

    def on_bar(self, row: pd.Series) -> list[PivotEvent]:
        self._bar_index += 1
        ts = row.name
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        current_source_ref = {
            "barIndex": self._bar_index,
            "timestamp": ts.isoformat(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }
        self._current_source_ref = current_source_ref

        emitted: list[PivotEvent] = []
        survivors: list[_Candidate] = []
        eligible: list[_Candidate] = []

        for candidate in self._candidates:
            if self._breaks_candidate(candidate, open_price, high, low, close):
                continue

            candidate.age += 1
            if candidate.age >= self.str_bars:
                eligible.append(candidate)
                continue

            survivors.append(candidate)

        selected = self._select_eligible_candidate(eligible)
        if selected is not None:
            str_event = self._event_from_candidate(selected, ts)
            if self.str_enabled:
                emitted.append(str_event)
            itr_event = self._itr_event_from_str(str_event) if self.itr_enabled else None
            if self.itr_enabled and itr_event is not None and self._itr_side_is_expected(itr_event.pivot_side):
                emitted.append(itr_event)
                self._last_itr_side = itr_event.pivot_side
                ltr_event = self._ltr_event_from_itr(itr_event) if self.ltr_enabled else None
                if ltr_event is not None and self._ltr_side_is_expected(ltr_event.pivot_side):
                    emitted.append(ltr_event)
                    self._last_ltr_side = ltr_event.pivot_side
                    self._remember_ltr_event(ltr_event)
                self._remember_itr_event(itr_event)
            self._remember_str_event(str_event)
            self._last_confirmed_side = selected.side

        survivors.append(_Candidate(side=1, price=high, timestamp=ts, index=self._bar_index, source_candle_ref=current_source_ref))
        survivors.append(_Candidate(side=-1, price=low, timestamp=ts, index=self._bar_index, source_candle_ref=current_source_ref))
        self._candidates = survivors

        return emitted

    def _breaks_candidate(
        self,
        candidate: _Candidate,
        open_price: float,
        high: float,
        low: float,
        close: float,
    ) -> bool:
        if self.mode == "aggressive":
            if candidate.side == 1:
                return high > candidate.price
            return low < candidate.price

        body_high = max(open_price, close)
        body_low = min(open_price, close)
        if candidate.side == 1:
            return body_high > candidate.price
        return body_low < candidate.price

    def _select_eligible_candidate(self, candidates: list[_Candidate]) -> _Candidate | None:
        expected_side = self._expected_side()
        eligible = [candidate for candidate in candidates if candidate.side == expected_side]
        if not eligible:
            return None

        if expected_side == 1:
            return max(eligible, key=lambda candidate: candidate.price)
        return min(eligible, key=lambda candidate: candidate.price)

    def _expected_side(self) -> PivotSide:
        if self._last_confirmed_side == 1:
            return -1
        return 1

    def _event_from_candidate(self, candidate: _Candidate, event_ts: pd.Timestamp) -> PivotEvent:
        if candidate.side == 1:
            return PivotEvent(
                event_code="PE01",
                event_name="strHighConfirmed",
                event_group="PE",
                tier="str",
                event_timestamp=event_ts,
                pivot_timestamp=candidate.timestamp,
                pivot_price=candidate.price,
                pivot_side=1,
                mode=self.mode,
                survival_bars=self.str_bars,
                timeframe=self.timeframe,
                source_bar_indexes=[candidate.index, self._bar_index],
                source_candle_refs=[candidate.source_candle_ref, self._current_source_ref],
            )

        return PivotEvent(
            event_code="PE02",
            event_name="strLowConfirmed",
            event_group="PE",
            tier="str",
            event_timestamp=event_ts,
            pivot_timestamp=candidate.timestamp,
            pivot_price=candidate.price,
            pivot_side=-1,
            mode=self.mode,
            survival_bars=self.str_bars,
            timeframe=self.timeframe,
            source_bar_indexes=[candidate.index, self._bar_index],
            source_candle_refs=[candidate.source_candle_ref, self._current_source_ref],
        )

    def _itr_event_from_str(self, str_event: PivotEvent) -> PivotEvent | None:
        if str_event.pivot_side == 1:
            extreme = self._last_str_high
            if extreme is None or str_event.pivot_price >= extreme.pivot_price:
                return None
            return PivotEvent(
                event_code="PE03",
                event_name="itrHighConfirmed",
                event_group="PE",
                tier="itr",
                event_timestamp=str_event.event_timestamp,
                pivot_timestamp=extreme.pivot_timestamp,
                pivot_price=extreme.pivot_price,
                pivot_side=1,
                mode=self.mode,
                survival_bars=self.str_bars,
                source_tier="str",
                source_event_code=str_event.event_code,
                confirmation_timestamp=str_event.pivot_timestamp,
                confirmation_event_timestamp=str_event.event_timestamp,
                confirmation_price=str_event.pivot_price,
                relation="high",
                timeframe=self.timeframe,
                source_bar_indexes=[*extreme.source_bar_indexes, *str_event.source_bar_indexes],
                source_candle_refs=[*extreme.source_candle_refs, *str_event.source_candle_refs],
            )

        extreme = self._last_str_low
        if extreme is None or str_event.pivot_price <= extreme.pivot_price:
            return None
        return PivotEvent(
            event_code="PE04",
            event_name="itrLowConfirmed",
            event_group="PE",
            tier="itr",
            event_timestamp=str_event.event_timestamp,
            pivot_timestamp=extreme.pivot_timestamp,
            pivot_price=extreme.pivot_price,
            pivot_side=-1,
            mode=self.mode,
            survival_bars=self.str_bars,
            source_tier="str",
            source_event_code=str_event.event_code,
            confirmation_timestamp=str_event.pivot_timestamp,
            confirmation_event_timestamp=str_event.event_timestamp,
            confirmation_price=str_event.pivot_price,
            relation="low",
            timeframe=self.timeframe,
            source_bar_indexes=[*extreme.source_bar_indexes, *str_event.source_bar_indexes],
            source_candle_refs=[*extreme.source_candle_refs, *str_event.source_candle_refs],
        )

    def _itr_side_is_expected(self, side: PivotSide) -> bool:
        if self._last_itr_side is None:
            return True
        return side != self._last_itr_side

    def _ltr_event_from_itr(self, itr_event: PivotEvent) -> PivotEvent | None:
        if itr_event.pivot_side == 1:
            extreme = self._last_itr_high
            if extreme is None or itr_event.pivot_price >= extreme.pivot_price:
                return None
            return PivotEvent(
                event_code="PE05",
                event_name="ltrHighConfirmed",
                event_group="PE",
                tier="ltr",
                event_timestamp=itr_event.event_timestamp,
                pivot_timestamp=extreme.pivot_timestamp,
                pivot_price=extreme.pivot_price,
                pivot_side=1,
                mode=self.mode,
                survival_bars=self.str_bars,
                source_tier="itr",
                source_event_code=itr_event.event_code,
                confirmation_timestamp=itr_event.pivot_timestamp,
                confirmation_event_timestamp=itr_event.event_timestamp,
                confirmation_price=itr_event.pivot_price,
                relation="high",
                timeframe=self.timeframe,
                source_bar_indexes=[*extreme.source_bar_indexes, *itr_event.source_bar_indexes],
                source_candle_refs=[*extreme.source_candle_refs, *itr_event.source_candle_refs],
            )

        extreme = self._last_itr_low
        if extreme is None or itr_event.pivot_price <= extreme.pivot_price:
            return None
        return PivotEvent(
            event_code="PE06",
            event_name="ltrLowConfirmed",
            event_group="PE",
            tier="ltr",
            event_timestamp=itr_event.event_timestamp,
            pivot_timestamp=extreme.pivot_timestamp,
            pivot_price=extreme.pivot_price,
            pivot_side=-1,
            mode=self.mode,
            survival_bars=self.str_bars,
            source_tier="itr",
            source_event_code=itr_event.event_code,
            confirmation_timestamp=itr_event.pivot_timestamp,
            confirmation_event_timestamp=itr_event.event_timestamp,
            confirmation_price=itr_event.pivot_price,
            relation="low",
            timeframe=self.timeframe,
            source_bar_indexes=[*extreme.source_bar_indexes, *itr_event.source_bar_indexes],
            source_candle_refs=[*extreme.source_candle_refs, *itr_event.source_candle_refs],
        )

    def _ltr_side_is_expected(self, side: PivotSide) -> bool:
        if self._last_ltr_side is None:
            return True
        return side != self._last_ltr_side

    def _remember_ltr_event(self, ltr_event: PivotEvent) -> None:
        if ltr_event.pivot_side == 1:
            self._last_ltr_high = ltr_event
            return
        self._last_ltr_low = ltr_event

    def _remember_str_event(self, str_event: PivotEvent) -> None:
        if str_event.pivot_side == 1:
            self._last_str_high = str_event
            return
        self._last_str_low = str_event

    def _remember_itr_event(self, itr_event: PivotEvent) -> None:
        if itr_event.pivot_side == 1:
            self._last_itr_high = itr_event
            return
        self._last_itr_low = itr_event
