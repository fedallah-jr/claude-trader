"""Adaptive Regime Strategy — calibrated LONG-only strategy.

Uses the framework's calibration system to dynamically adapt:
  - Which regime gate to use (breadth, trend convergence, or volatility)
  - TP/SL scaling based on recent market conditions
  - Entry strictness (RSI and ret_24h thresholds)

Two fixed entry patterns gated by the calibrated regime filter:
  1. Dip-buy recovery: shallow pullback + bullish bar in uptrend
  2. Trend continuation: strong bar in smooth uptrend

22-symbol universe for broad coverage. 108 calibration combinations
(small grid to resist overfitting). Weekly recalibration with 30-day
lookback. Scoring uses profit_factor * trade_penalty * consistency
to favor stable edges over high-variance lucky combos.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtester.indicators import compute_indicator_frame, required_warmup
from backtester.models import MarketType, PositionType, Signal
from backtester.pipeline import PreparedMarketContext
from live.signal_generator import SignalGenerator
from marketdata import MarketDataRequest

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "NEAR/USDT", "APT/USDT", "RENDER/USDT", "INJ/USDT", "LTC/USDT",
    "ATOM/USDT", "FIL/USDT", "ARB/USDT", "OP/USDT", "TIA/USDT",
    "SUI/USDT", "SEI/USDT",
]

_INDICATORS = (
    "rsi_14", "atr_14", "atr_ratio",
    "ret_72h", "ret_24h",
    "mom_slope", "ema_20",
    "body", "body_ratio", "vol_ratio",
)

_WARMUP = max(required_warmup(_INDICATORS), 100)

_COOLDOWN_HOURS = 12.0
_MAX_HOLDING_HOURS = 72

# ── Entry strictness presets ──────────────────────────────────────────
_STRICTNESS = {
    "loose":  {"rsi_max": 70.0, "ret24_lo": -8.0, "ret24_hi": 2.0},
    "normal": {"rsi_max": 65.0, "ret24_lo": -6.0, "ret24_hi": 0.5},
    "strict": {"rsi_max": 60.0, "ret24_lo": -4.0, "ret24_hi": -0.5},
}

# ── Default params (used before first calibration) ────────────────────
_DEFAULTS: dict[str, Any] = {
    "regime_gate": "breadth",
    "tp_mult": 1.0,
    "sl_mult": 1.0,
    "entry_strictness": "normal",
}


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)


# ══════════════════════════════════════════════════════════════════════
# Cross-asset state computation
# ══════════════════════════════════════════════════════════════════════

def _build_cross_asset_cache(
    frames: dict[str, pd.DataFrame],
    timestamps: list[datetime],
) -> dict[datetime, dict[str, float]]:
    """Precompute breadth, convergence, median atr_ratio per timestamp."""
    ret_index: dict[str, dict[datetime, float]] = {}
    slope_index: dict[str, dict[datetime, float]] = {}
    atr_index: dict[str, dict[datetime, float]] = {}

    for symbol, frame in frames.items():
        if frame.empty:
            continue
        ct = frame["close_time"]
        ret_index[symbol] = dict(zip(ct, frame["ret_72h"]))
        slope_index[symbol] = dict(zip(ct, frame["mom_slope"]))
        atr_index[symbol] = dict(zip(ct, frame["atr_ratio"]))

    cache: dict[datetime, dict[str, float]] = {}
    for t in timestamps:
        pos_ret = 0
        total_ret = 0
        pos_slope = 0
        total_slope = 0
        atr_vals: list[float] = []

        for sym in ret_index:
            v = ret_index[sym].get(t)
            if v is not None and not pd.isna(v):
                total_ret += 1
                if v > 0:
                    pos_ret += 1
            s = slope_index.get(sym, {}).get(t)
            if s is not None and not pd.isna(s):
                total_slope += 1
                if s > 0:
                    pos_slope += 1
            a = atr_index.get(sym, {}).get(t)
            if a is not None and not pd.isna(a):
                atr_vals.append(a)

        breadth = pos_ret / total_ret if total_ret > 0 else 0.0
        convergence = pos_slope / total_slope if total_slope > 0 else 0.0
        med_atr = float(np.median(atr_vals)) if atr_vals else 1.0

        cache[t] = {
            "breadth": breadth,
            "convergence": convergence,
            "med_atr": med_atr,
        }
    return cache


def _check_regime(
    gate: str,
    state: dict[str, float],
) -> bool:
    """Check whether the regime gate passes for the current cross-asset state."""
    if gate == "breadth":
        return state["breadth"] >= 0.75
    elif gate == "trend":
        return state["convergence"] >= 0.65
    elif gate == "vol_regime":
        return state["med_atr"] < 1.0
    return False


# ══════════════════════════════════════════════════════════════════════
# Entry pattern checks
# ══════════════════════════════════════════════════════════════════════

def _check_dipbuy(
    row: pd.Series,
    prev: pd.Series,
    strictness: dict[str, float],
) -> tuple[bool, dict[str, object]]:
    """Dip-buy recovery: shallow pullback + bullish bar in uptrend."""
    if pd.isna(row.get("ema_20")) or pd.isna(row.get("atr_14")):
        return False, {}

    ret_72h = _safe(row, "ret_72h")
    if ret_72h < 5.0 or ret_72h > 35.0:
        return False, {}

    ret_24h = _safe(row, "ret_24h")
    if ret_24h > strictness["ret24_hi"] or ret_24h < strictness["ret24_lo"]:
        return False, {}

    rsi = _safe(row, "rsi_14", 50.0)
    if rsi > strictness["rsi_max"] or rsi < 30.0:
        return False, {}

    atr_ratio = _safe(row, "atr_ratio", 1.0)
    if atr_ratio > 1.3:
        return False, {}

    close = float(row["close"])
    if close <= float(row["ema_20"]):
        return False, {}

    high, low, open_price = float(row["high"]), float(row["low"]), float(row["open"])
    bar_range = high - low
    if bar_range <= 0:
        return False, {}
    body = close - open_price
    if body <= 0 or body / bar_range < 0.3:
        return False, {}

    vol_ratio = _safe(row, "vol_ratio", 1.0)
    if vol_ratio < 0.8:
        return False, {}

    # Previous bar should be bearish or neutral
    prev_range = float(prev["high"]) - float(prev["low"])
    if prev_range > 0:
        prev_body = float(prev["close"]) - float(prev["open"])
        if prev_body > 0.3 * prev_range:
            return False, {}

    return True, {
        "pattern": "dipbuy",
        "ret_72h": round(ret_72h, 2),
        "ret_24h": round(ret_24h, 2),
        "rsi": round(rsi, 2),
        "atr_ratio": round(atr_ratio, 3),
    }


def _check_trend_continuation(
    row: pd.Series,
) -> tuple[bool, dict[str, object]]:
    """Trend continuation: strong bar in smooth uptrend."""
    if pd.isna(row.get("ema_20")) or pd.isna(row.get("atr_14")):
        return False, {}

    ret_72h = _safe(row, "ret_72h")
    if ret_72h < 12.0 or ret_72h > 45.0:
        return False, {}

    ret_24h = _safe(row, "ret_24h")
    if ret_24h < 0.5 or ret_24h > 5.0:
        return False, {}

    rsi = _safe(row, "rsi_14", 50.0)
    if rsi < 45.0 or rsi > 65.0:
        return False, {}

    mom_slope = _safe(row, "mom_slope")
    if mom_slope <= 0:
        return False, {}

    close = float(row["close"])
    if close <= float(row["ema_20"]):
        return False, {}

    atr_ratio = _safe(row, "atr_ratio", 1.0)
    if atr_ratio > 1.0:
        return False, {}

    atr = _safe(row, "atr_14", 0.0)
    if atr <= 0:
        return False, {}
    body = _safe(row, "body")
    if body <= 1.5 * atr:
        return False, {}

    return True, {
        "pattern": "trend_cont",
        "ret_72h": round(ret_72h, 2),
        "ret_24h": round(ret_24h, 2),
        "rsi": round(rsi, 2),
        "mom_slope": round(mom_slope, 6),
    }


# ══════════════════════════════════════════════════════════════════════
# Strategy class
# ══════════════════════════════════════════════════════════════════════

class AdaptiveRegimeStrategy(SignalGenerator):
    """Adaptive calibrated LONG-only strategy with regime gating."""

    analysis_interval = "1h"
    needs_calibration = True
    calibration_interval_hours = 168   # weekly
    calibration_lookback_hours = 720   # 30 days

    def __init__(self, leverage: float = 1.0, **kwargs: Any) -> None:
        self.leverage = leverage

    @property
    def symbols(self) -> list[str]:
        return list(SYMBOLS)

    @property
    def required_warmup_bars(self) -> int:
        return _WARMUP

    @property
    def cooldown_hours(self) -> float:
        return _COOLDOWN_HOURS

    def indicator_request(self) -> tuple[str, ...]:
        return _INDICATORS

    def market_data_request(self) -> MarketDataRequest:
        return MarketDataRequest.ohlcv_only(interval=self.analysis_interval)

    def __str__(self) -> str:
        p = self.active_params or _DEFAULTS
        return (
            f"AdaptiveRegime(gate={p.get('regime_gate', '?')} "
            f"tp_m={p.get('tp_mult', '?')} sl_m={p.get('sl_mult', '?')} "
            f"strict={p.get('entry_strictness', '?')} "
            f"{len(SYMBOLS)}sym)"
        )

    # ── Calibration API ───────────────────────────────────────────────

    def param_space(self) -> dict[str, list]:
        return {
            "regime_gate": ["breadth", "trend", "vol_regime"],
            "tp_mult": [0.8, 1.0, 1.2, 1.4],
            "sl_mult": [0.8, 1.0, 1.2],
            "entry_strictness": ["loose", "normal", "strict"],
        }

    def build_calibration_frame(
        self, frame: pd.DataFrame, t: datetime,
    ) -> pd.DataFrame:
        enriched: list[pd.DataFrame] = []
        for symbol in frame["symbol"].unique():
            sym_data = frame[frame["symbol"] == symbol].sort_values("close_time").copy()
            if len(sym_data) < _WARMUP:
                continue
            sym_data = compute_indicator_frame(sym_data, _INDICATORS)
            enriched.append(sym_data)
        if not enriched:
            return pd.DataFrame()
        return pd.concat(enriched, ignore_index=True)

    def score_params(self, params: dict[str, Any], frame: pd.DataFrame) -> float:
        """Score a candidate parameter set by simulating trades on lookback data."""
        if frame.empty or "symbol" not in frame.columns:
            return -100.0
        gate = params["regime_gate"]
        tp_mult = params["tp_mult"]
        sl_mult = params["sl_mult"]
        strictness = _STRICTNESS[params["entry_strictness"]]

        base_tp_dip = 4.5 * tp_mult / 100.0
        base_sl_dip = 2.0 * sl_mult / 100.0
        base_tp_trend = 4.0 * tp_mult / 100.0
        base_sl_trend = 2.0 * sl_mult / 100.0

        # Group by symbol
        symbols_in_frame = frame["symbol"].unique()
        sym_frames: dict[str, pd.DataFrame] = {}
        for s in symbols_in_frame:
            sf = frame[frame["symbol"] == s].sort_values("close_time").reset_index(drop=True)
            if len(sf) > _WARMUP:
                sym_frames[s] = sf

        if not sym_frames:
            return -100.0

        # Get all timestamps in signal evaluation zone (leave 72h buffer at end)
        all_times: set[datetime] = set()
        for sf in sym_frames.values():
            all_times.update(sf["close_time"].tolist())
        sorted_times = sorted(all_times)
        if not sorted_times:
            return -100.0

        max_signal_time = sorted_times[-1] - timedelta(hours=_MAX_HOLDING_HOURS)

        # Build cross-asset state
        cross_state = _build_cross_asset_cache(sym_frames, sorted_times)

        # Simulate trades
        returns: list[float] = []
        for sym, sf in sym_frames.items():
            last_sig_time: datetime | None = None
            for i in range(max(_WARMUP, 3), len(sf)):
                row = sf.iloc[i]
                ct = row["close_time"]
                if ct > max_signal_time:
                    break

                if last_sig_time is not None:
                    if (ct - last_sig_time).total_seconds() / 3600 < _COOLDOWN_HOURS:
                        continue

                state = cross_state.get(ct)
                if state is None or not _check_regime(gate, state):
                    continue

                prev = sf.iloc[i - 1]
                ok_dip, meta_dip = _check_dipbuy(row, prev, strictness)
                ok_trend, meta_trend = _check_trend_continuation(row)

                if ok_dip:
                    tp_frac = base_tp_dip
                    sl_frac = base_sl_dip
                elif ok_trend:
                    tp_frac = base_tp_trend
                    sl_frac = base_sl_trend
                else:
                    continue

                # Approximate exit: scan forward bars for TP/SL hit
                entry_price = float(row["close"])
                pnl = self._approximate_exit(
                    sf, i, entry_price, tp_frac, sl_frac,
                )
                returns.append(pnl)
                last_sig_time = ct

        if len(returns) < 3:
            return -100.0

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.001

        profit_factor = gross_win / max(gross_loss, 0.001)
        trade_penalty = min(1.0, len(returns) / 10.0)
        consistency = 1.0 / (1.0 + float(np.std(returns)))

        return profit_factor * trade_penalty * consistency

    @staticmethod
    def _approximate_exit(
        sf: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        tp_frac: float,
        sl_frac: float,
    ) -> float:
        """Scan forward up to 72 bars; check if TP or SL is hit first."""
        tp_price = entry_price * (1.0 + tp_frac)
        sl_price = entry_price * (1.0 - sl_frac)
        max_bars = min(_MAX_HOLDING_HOURS, len(sf) - entry_idx - 1)

        for j in range(1, max_bars + 1):
            bar = sf.iloc[entry_idx + j]
            low = float(bar["low"])
            high = float(bar["high"])

            # SL checked first (conservative)
            if low <= sl_price:
                return -sl_frac - 0.001  # fee approximation
            if high >= tp_price:
                return tp_frac - 0.001

        # Timeout: use last close
        if max_bars > 0:
            exit_price = float(sf.iloc[entry_idx + max_bars]["close"])
            return (exit_price - entry_price) / entry_price - 0.001
        return -0.001

    # ── Signal generation ─────────────────────────────────────────────

    def poll(self) -> Signal | list[Signal] | None:
        return None

    def generate_backtest_signals(
        self,
        prepared_context: PreparedMarketContext,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[Signal]:
        params = self.active_params if self.active_params else _DEFAULTS
        gate = params.get("regime_gate", "breadth")
        tp_mult = params.get("tp_mult", 1.0)
        sl_mult = params.get("sl_mult", 1.0)
        strictness = _STRICTNESS[params.get("entry_strictness", "normal")]

        # Build indicator frames
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                frames[symbol] = prepared_context.indicator_frame(symbol)
            except KeyError:
                continue
        if not frames:
            return []

        # Collect timestamps in [start, end)
        all_times: set[datetime] = set()
        for frame in frames.values():
            if frame.empty or "close_time" not in frame.columns:
                continue
            mask = (frame["close_time"] >= start) & (frame["close_time"] < end)
            all_times.update(frame.loc[mask, "close_time"].tolist())

        cross_state = _build_cross_asset_cache(frames, sorted(all_times))

        all_signals: list[Signal] = []
        for symbol, frame in frames.items():
            if frame.empty or len(frame) < _WARMUP + 3:
                continue

            last_signal_time: datetime | None = None

            for i in range(max(_WARMUP, 3), len(frame)):
                ct = frame.iloc[i]["close_time"]
                if ct < start or ct >= end:
                    continue

                # Cooldown
                if last_signal_time is not None:
                    hours = (ct - last_signal_time).total_seconds() / 3600.0
                    if hours < _COOLDOWN_HOURS:
                        continue

                # Regime gate
                state = cross_state.get(ct)
                if state is None or not _check_regime(gate, state):
                    continue

                row = frame.iloc[i]
                prev = frame.iloc[i - 1]

                ok, metadata = _check_dipbuy(row, prev, strictness)
                if not ok:
                    ok, metadata = _check_trend_continuation(row)

                if not ok:
                    continue

                metadata["gate"] = gate
                if state:
                    metadata["breadth"] = round(state["breadth"], 2)
                    metadata["convergence"] = round(state["convergence"], 2)
                    metadata["med_atr"] = round(state["med_atr"], 3)

                is_trend = metadata.get("pattern") == "trend_cont"
                base_tp = 4.0 if is_trend else 4.5
                base_sl = 2.0

                signal = Signal(
                    signal_date=ct,
                    position_type=PositionType.LONG,
                    ticker=symbol,
                    tp_pct=base_tp * tp_mult,
                    sl_pct=base_sl * sl_mult,
                    leverage=self.leverage,
                    market_type=MarketType.FUTURES,
                    taker_fee_rate=0.0005,
                    max_holding_hours=_MAX_HOLDING_HOURS,
                    metadata=metadata,
                )
                all_signals.append(signal)
                last_signal_time = ct

        all_signals.sort(key=lambda s: s.signal_date)
        return all_signals
