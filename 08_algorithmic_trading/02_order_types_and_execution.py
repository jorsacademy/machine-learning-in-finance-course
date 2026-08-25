"""Simple execution mechanics for market, limit, and stop orders."""

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Quote:
    bid: float
    ask: float


@dataclass
class Fill:
    side: Side
    quantity: int
    price: float


def market_order(side: Side, quantity: int, quote: Quote) -> Fill:
    price = quote.ask if side == Side.BUY else quote.bid
    return Fill(side, quantity, price)


def limit_order(side: Side, quantity: int, limit_price: float, quote: Quote) -> Fill | None:
    if side == Side.BUY and quote.ask <= limit_price:
        return Fill(side, quantity, quote.ask)
    if side == Side.SELL and quote.bid >= limit_price:
        return Fill(side, quantity, quote.bid)
    return None


def stop_order(side: Side, quantity: int, stop_price: float, last_price: float, quote: Quote) -> Fill | None:
    triggered = (side == Side.BUY and last_price >= stop_price) or (side == Side.SELL and last_price <= stop_price)
    return market_order(side, quantity, quote) if triggered else None


def main() -> None:
    quote = Quote(bid=99.95, ask=100.05)
    print("Market buy:", market_order(Side.BUY, 100, quote))
    print("Limit buy at 100.00:", limit_order(Side.BUY, 100, 100.00, quote))
    print("Limit buy at 100.10:", limit_order(Side.BUY, 100, 100.10, quote))
    print("Stop sell at 99.50 with last price 99.40:", stop_order(Side.SELL, 100, 99.50, 99.40, quote))


if __name__ == "__main__":
    main()
