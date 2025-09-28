import argparse
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        dtype=float,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(timezone.utc)
    return df.set_index("timestamp")


def compute_indicators(df: pd.DataFrame,
                       sma_fast: int, sma_slow: int,
                       rsi_period: int,
                       macd_fast: int, macd_slow: int, macd_signal: int) -> pd.DataFrame:
    df = df.copy()

    df[f"sma_{sma_fast}"] = SMAIndicator(df["close"], window=sma_fast).sma_indicator()
    df[f"sma_{sma_slow}"] = SMAIndicator(df["close"], window=sma_slow).sma_indicator()

    rsi = RSIIndicator(df["close"], window=rsi_period)
    df["rsi"] = rsi.rsi()

    macd = MACD(
        close=df["close"],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    return df


def generate_signal(latest: pd.Series,
                    sma_fast: int, sma_slow: int,
                    rsi_upper: float, rsi_lower: float) -> dict:
    signal_votes = []

    sma_fast_val = latest[f"sma_{sma_fast}"]
    sma_slow_val = latest[f"sma_{sma_slow}"]

    if np.isnan(sma_fast_val) or np.isnan(sma_slow_val):
        sma_bias = "neutral"
    else:
        if sma_fast_val > sma_slow_val:
            sma_bias = "bullish"
            signal_votes.append(1)
        elif sma_fast_val < sma_slow_val:
            sma_bias = "bearish"
            signal_votes.append(-1)
        else:
            sma_bias = "neutral"

    rsi_val = latest["rsi"]
    if np.isnan(rsi_val):
        rsi_bias = "neutral"
    else:
        if rsi_val > rsi_upper:
            rsi_bias = "overbought"
            signal_votes.append(-1)
        elif rsi_val < rsi_lower:
            rsi_bias = "oversold"
            signal_votes.append(1)
        else:
            rsi_bias = "neutral"

    macd_val = latest["macd"]
    macd_signal_val = latest["macd_signal"]
    if np.isnan(macd_val) or np.isnan(macd_signal_val):
        macd_bias = "neutral"
    else:
        if macd_val > macd_signal_val:
            macd_bias = "bullish"
            signal_votes.append(1)
        elif macd_val < macd_signal_val:
            macd_bias = "bearish"
            signal_votes.append(-1)
        else:
            macd_bias = "neutral"

    if not signal_votes:
        overall = "neutral"
    else:
        score = sum(signal_votes)
        if score > 0:
            overall = "strong_buy" if score >= 2 else "buy"
        elif score < 0:
            overall = "strong_sell" if score <= -2 else "sell"
        else:
            overall = "neutral"

    return {
        "overall_signal": overall,
        "details": {
            "sma_bias": sma_bias,
            "rsi_bias": rsi_bias,
            "macd_bias": macd_bias,
            "price": latest["close"],
            "rsi": rsi_val,
            "macd": macd_val,
            "macd_signal": macd_signal_val,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate crypto trading signals using technical indicators.")
    parser.add_argument("--exchange", default="binance", help="Exchange ID supported by ccxt (default: binance)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair, e.g., BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="OHLCV timeframe, e.g., 1m, 5m, 1h, 4h, 1d")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles to fetch (max per exchange rules)")
    parser.add_argument("--sma_fast", type=int, default=20)
    parser.add_argument("--sma_slow", type=int, default=50)
    parser.add_argument("--rsi_period", type=int, default=14)
    parser.add_argument("--rsi_upper", type=float, default=70)
    parser.add_argument("--rsi_lower", type=float, default=30)
    parser.add_argument("--macd_fast", type=int, default=12)
    parser.add_argument("--macd_slow", type=int, default=26)
    parser.add_argument("--macd_signal", type=int, default=9)

    args = parser.parse_args()

    df = fetch_ohlcv(args.exchange, args.symbol, args.timeframe, args.limit)
    df = compute_indicators(
        df,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        rsi_period=args.rsi_period,
        macd_fast=args.macd_fast,
        macd_slow=args.macd_slow,
        macd_signal=args.macd_signal,
    )

    latest = df.dropna().iloc[-1]
    signal = generate_signal(
        latest,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        rsi_upper=args.rsi_upper,
        rsi_lower=args.rsi_lower,
    )

    timestamp = df.dropna().index[-1].strftime("%Y-%m-%d %H:%M:%S %Z")
    print("=" * 60)
    print(f"Exchange: {args.exchange} | Symbol: {args.symbol} | Timeframe: {args.timeframe}")
    print(f"Last candle: {timestamp}")
    print("=" * 60)
    print(f"Overall signal: {signal['overall_signal']}")
    print("Details:")
    for key, value in signal["details"].items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()