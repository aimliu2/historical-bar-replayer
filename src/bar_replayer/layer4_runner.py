from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bar_replayer.config import DEFAULT_CONFIG_PATH, load_config
from bar_replayer.data_source import load_bars, parquet_path, resolve_data_root
from bar_replayer.session import DualReplaySession
from hypothesis import Layer4
from order_plan import OrderPlan
from hypothesis.persistence import Layer4SaveError, load_layer4, save_layer4
from smc.anchors import sha256_file


class Layer4Runner:
    """Connect Layer 4 save/resume and Layer 5 plans to the shared Tri context stream."""

    def __init__(
        self,
        *,
        runtime: Layer4,
        session: DualReplaySession,
        config: dict[str, Any],
        repo_root: Path,
        save_path: Path,
        load_info: dict[str, Any],
        input_files: dict[str, Any],
        save_each_close: bool,
    ) -> None:
        self.runtime = runtime
        self.order_plan = OrderPlan(config)
        self.session = session
        self.config = config
        self.repo_root = repo_root
        self.save_path = save_path
        self.load_info = load_info
        self.input_files = input_files
        self.save_each_close = bool(save_each_close)
        self.catch_up = bool(load_info.get("resumed") or load_info.get("cold_start"))

    @classmethod
    def from_repository(
        cls,
        *,
        symbol: str | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        repo_root: str | Path | None = None,
        data_root: str | Path | None = None,
        cold_start: bool = False,
        rebuild_from_genesis: bool = False,
        save_each_close: bool = True,
    ) -> Layer4Runner:
        root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        config = load_config(config_path)
        config_source = Path(config_path).resolve()
        selected = str(symbol or config["symbol"]).strip().upper()
        if selected not in config["symbols"]:
            raise Layer4SaveError(f"No Layer 4 symbol configuration exists for {selected}")
        config["symbol"] = selected
        asset = config["symbols"][selected]

        resolved_data_root = resolve_data_root(data_root or config["data_root"])
        daily = load_bars(selected, "1d", resolved_data_root)
        hourly = load_bars(selected, "1h", resolved_data_root)
        lower = load_bars(selected, "5m", resolved_data_root)
        bar_paths = {
            tf: parquet_path(selected, tf, resolved_data_root)
            for tf in ("1d", "1h", "5m")
        }
        hashes = {tf: sha256_file(path) for tf, path in bar_paths.items()}
        input_files = {
            tf: {"path": str(path.resolve()), "sha256": hashes[tf]}
            for tf, path in bar_paths.items()
        }
        if config_source.is_file():
            input_files["config"] = {
                "path": str(config_source),
                "sha256": sha256_file(config_source),
            }
        anchor_manifests: dict[str, Any] = {}
        for key, set_name, filename in (
            ("1d", asset.get("anchor_set_1d"), f"{selected}_1D_manifest.json"),
            ("1h5m", asset.get("anchor_set"), f"{selected}_1H5M_manifest.json"),
        ):
            manifest = root / "export" / selected / "anchor" / str(set_name or "") / filename
            anchor_manifests[key] = (
                {"path": str(manifest), "sha256": sha256_file(manifest)}
                if set_name and manifest.is_file()
                else {"path": str(manifest), "sha256": None, "available": False}
            )
        input_files["anchor_manifests"] = anchor_manifests
        save_path = root / "export" / selected / "layer4" / f"{selected}_layer4_save.json"

        if rebuild_from_genesis:
            runtime = Layer4(config)
            load_info = {
                "resumed": False,
                "cold_start": True,
                "reason": "explicit Layer 4 save rebuild from genesis",
                "last_close": None,
                "save_path": str(save_path),
            }
        else:
            runtime, load_info = load_layer4(
                config,
                repo_root=root,
                save_path=save_path,
                cold_start=cold_start,
            )

        if runtime.last_time is None:
            start_time = lower.index[0].isoformat()
        else:
            saved_close = pd.Timestamp(runtime.last_time)
            saved_close = saved_close.tz_localize("UTC") if saved_close.tzinfo is None else saved_close.tz_convert("UTC")
            if saved_close not in lower.index:
                raise Layer4SaveError(
                    f"Layer 4 save close {saved_close.isoformat()} is absent from the current 5m data; "
                    "refresh the save or explicitly cold-start"
                )
            start_time = (saved_close + pd.Timedelta(nanoseconds=1)).isoformat()

        lanes = config["lanes"]
        anchor_set = asset.get("anchor_set")
        daily_anchor_set = asset.get("anchor_set_1d")
        anchor_settings = (
            {"enabled": True, "set_version": anchor_set}
            if anchor_set
            else {"enabled": False}
        )
        daily_anchor_settings = (
            {"enabled": True, "set_version": daily_anchor_set}
            if daily_anchor_set
            else {"enabled": False}
        )
        session = DualReplaySession(
            lower,
            hourly,
            symbol=selected,
            clock="5m",
            start_time=start_time,
            window_bars=lanes["window_bars"],
            focus_context_bars=lanes["focus_context_bars"],
            structure_settings={**lanes["structure"], "enabled": True},
            orderflow_settings=lanes["orderflow"],
            str_defense_settings=lanes["str_defense"],
            ta_settings=lanes["ta"],
            str_bars_1d=lanes["str_bars_1d"],
            anchor_settings=anchor_settings,
            data_sha256=hashes,
            repository_root=root,
            mode="tri",
            daily_bars=daily,
            daily_anchor_settings=daily_anchor_settings,
            daily_lookback_bars=400,
            symbol_code=asset.get("code"),
            pip_size=asset.get("pip_size"),
        )
        return cls(
            runtime=runtime,
            session=session,
            config=config,
            repo_root=root,
            save_path=save_path,
            load_info=load_info,
            input_files=input_files,
            save_each_close=save_each_close,
        )

    def step(self) -> dict[str, Any] | None:
        """Advance one 5m close, publish Layer 4 plus Layer 5 plans, and save Layer 4."""
        if self.session.done:
            return None
        delta = self.session.step()
        context = delta.get("fusion_context")
        output = None
        if context is not None:
            at = pd.Timestamp(context["time"])
            at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
            last = pd.Timestamp(self.runtime.last_time) if self.runtime.last_time is not None else None
            if last is not None:
                last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
            if last is None or at > last:
                output = self.runtime.step(context, catch_up=self.catch_up)
                output["layer5_plans"] = self.order_plan.step(output)
                if self.save_each_close:
                    save_layer4(
                        self.runtime,
                        config=self.config,
                        repo_root=self.repo_root,
                        save_path=self.save_path,
                        inputs=self.input_files,
                    )
        return {"delta": delta, "layer4": output}

    def run_to_latest(self) -> dict[str, Any]:
        """Catch up through the currently stored data, saving each resumed close."""
        outputs = 0
        last_output: dict[str, Any] | None = None
        while not self.session.done:
            result = self.step()
            if result is None:
                break
            if result.get("layer4") is not None:
                outputs += 1
                last_output = result["layer4"]
        return {
            "symbol": self.runtime.symbol,
            "resumed": bool(self.load_info.get("resumed")),
            "cold_start": bool(self.load_info.get("cold_start")),
            "catch_up_closes": outputs,
            "last_close": self.runtime.last_time,
            "save_path": str(self.save_path),
            "last_output": last_output,
        }
