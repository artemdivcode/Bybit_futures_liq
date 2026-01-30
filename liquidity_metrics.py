#!/usr/bin/env python3
"""Estimate real liquidity metrics for Bybit USDT linear futures.

This script pulls Bybit v5 public market data and derives metrics that
penalize spoofed depth by comparing order book depth with recent trade flow.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError

BASE_URL = "https://api.bybit.com"
CATEGORY = "linear"


@dataclass
class OrderBook:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


@dataclass
class Trade:
    price: float
    size: float
    timestamp_ms: int


@dataclass
class SymbolMetrics:
    symbol: str
    mid_price: float
    spread_bps: float
    depth_01_bps: float
    depth_05_bps: float
    depth_10_bps: float
    trade_notional_1m: float
    trade_notional_5m: float
    effective_depth_01_bps: float
    effective_depth_05_bps: float
    effective_depth_10_bps: float


def request_json(path: str, params: dict[str, str | int]) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while requesting {url}: {exc}") from exc
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {payload}")
    return payload


def fetch_symbols() -> list[str]:
    payload = request_json(
        "/v5/market/instruments-info",
        {"category": CATEGORY, "status": "Trading"},
    )
    symbols = []
    for item in payload["result"]["list"]:
        symbol = item["symbol"]
        if "USDT" in symbol:
            symbols.append(symbol)
    return symbols


def fetch_orderbook(symbol: str, limit: int) -> OrderBook:
    payload = request_json(
        "/v5/market/orderbook",
        {"category": CATEGORY, "symbol": symbol, "limit": limit},
    )
    bids = [(float(px), float(sz)) for px, sz in payload["result"]["b"]]
    asks = [(float(px), float(sz)) for px, sz in payload["result"]["a"]]
    return OrderBook(bids=bids, asks=asks)


def fetch_trades(symbol: str, limit: int) -> list[Trade]:
    payload = request_json(
        "/v5/market/trade",
        {"category": CATEGORY, "symbol": symbol, "limit": limit},
    )
    trades = []
    for item in payload["result"]["list"]:
        trades.append(
            Trade(
                price=float(item["price"]),
                size=float(item["size"]),
                timestamp_ms=int(item["time"]),
            )
        )
    return trades


def mid_price(orderbook: OrderBook) -> float:
    if not orderbook.bids or not orderbook.asks:
        return math.nan
    return (orderbook.bids[0][0] + orderbook.asks[0][0]) / 2


def spread_bps(orderbook: OrderBook) -> float:
    if not orderbook.bids or not orderbook.asks:
        return math.nan
    mid = mid_price(orderbook)
    spread = orderbook.asks[0][0] - orderbook.bids[0][0]
    return spread / mid * 10_000


def depth_within_bps(
    side: Iterable[tuple[float, float]], mid: float, bps: float, is_bid: bool
) -> float:
    if math.isnan(mid):
        return math.nan
    limit_price = mid * (1 - bps / 10_000) if is_bid else mid * (1 + bps / 10_000)
    total = 0.0
    for price, size in side:
        if is_bid and price < limit_price:
            break
        if not is_bid and price > limit_price:
            break
        total += price * size
    return total


def trade_notional(trades: list[Trade], window_s: int, now_ms: int) -> float:
    cutoff = now_ms - window_s * 1000
    total = 0.0
    for trade in trades:
        if trade.timestamp_ms < cutoff:
            continue
        total += trade.price * trade.size
    return total


def effective_depth(depth: float, trade_flow: float, multiplier: float) -> float:
    """Cap quoted depth by recent trade flow to penalize spoofed liquidity."""
    if math.isnan(depth):
        return math.nan
    return min(depth, trade_flow * multiplier)


def compute_metrics(symbol: str, limit: int, trade_limit: int) -> SymbolMetrics:
    orderbook = fetch_orderbook(symbol, limit)
    trades = fetch_trades(symbol, trade_limit)
    now_ms = int(time.time() * 1000)

    mid = mid_price(orderbook)
    spread = spread_bps(orderbook)

    depth_01 = depth_within_bps(orderbook.bids, mid, 1, True) + depth_within_bps(
        orderbook.asks, mid, 1, False
    )
    depth_05 = depth_within_bps(orderbook.bids, mid, 5, True) + depth_within_bps(
        orderbook.asks, mid, 5, False
    )
    depth_10 = depth_within_bps(orderbook.bids, mid, 10, True) + depth_within_bps(
        orderbook.asks, mid, 10, False
    )

    trade_1m = trade_notional(trades, 60, now_ms)
    trade_5m = trade_notional(trades, 300, now_ms)

    effective_01 = effective_depth(depth_01, trade_1m, 3.0)
    effective_05 = effective_depth(depth_05, trade_1m, 3.0)
    effective_10 = effective_depth(depth_10, trade_1m, 3.0)

    return SymbolMetrics(
        symbol=symbol,
        mid_price=mid,
        spread_bps=spread,
        depth_01_bps=depth_01,
        depth_05_bps=depth_05,
        depth_10_bps=depth_10,
        trade_notional_1m=trade_1m,
        trade_notional_5m=trade_5m,
        effective_depth_01_bps=effective_01,
        effective_depth_05_bps=effective_05,
        effective_depth_10_bps=effective_10,
    )


def to_dict(metrics: SymbolMetrics) -> dict:
    return {
        "symbol": metrics.symbol,
        "mid_price": metrics.mid_price,
        "spread_bps": metrics.spread_bps,
        "depth_01_bps": metrics.depth_01_bps,
        "depth_05_bps": metrics.depth_05_bps,
        "depth_10_bps": metrics.depth_10_bps,
        "trade_notional_1m": metrics.trade_notional_1m,
        "trade_notional_5m": metrics.trade_notional_5m,
        "effective_depth_01_bps": metrics.effective_depth_01_bps,
        "effective_depth_05_bps": metrics.effective_depth_05_bps,
        "effective_depth_10_bps": metrics.effective_depth_10_bps,
    }


def write_output(metrics: list[SymbolMetrics], fmt: str) -> None:
    if not metrics:
        return
    if fmt == "json":
        data = [to_dict(item) for item in metrics]
        json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=list(to_dict(metrics[0]).keys()))
    writer.writeheader()
    for item in metrics:
        writer.writerow(to_dict(item))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate real liquidity for Bybit USDT linear futures by comparing "
            "order book depth with recent trade flow."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional list of symbols to analyze (defaults to all USDT futures).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Order book depth levels to request.",
    )
    parser.add_argument(
        "--trade-limit",
        type=int,
        default=1000,
        help="Number of recent trades to request.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = args.symbols or fetch_symbols()
    metrics = []
    for symbol in symbols:
        metrics.append(compute_metrics(symbol, args.limit, args.trade_limit))
    write_output(metrics, args.format)


if __name__ == "__main__":
    main()
