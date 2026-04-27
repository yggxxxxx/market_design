from __future__ import annotations

from functools import partial
from typing import Callable, Sequence


def _validate_price_inputs(bid_price: float, ask_price: float) -> tuple[float, float]:
    bid_price = float(bid_price)
    ask_price = float(ask_price)

    if bid_price < 0 or ask_price < 0:
        raise ValueError("bid_price and ask_price must be >= 0")

    if bid_price < ask_price:
        raise ValueError("bid_price must be >= ask_price for a valid match")

    return bid_price, ask_price


# ---------------------------------------------------------------------
# CDA pricing rules
# ---------------------------------------------------------------------

def midpoint_price(bid_price: float, ask_price: float) -> float:
    """
    Smith-style midpoint pricing:
    p = (bid + ask) / 2
    """
    bid_price, ask_price = _validate_price_inputs(bid_price, ask_price)
    return (bid_price + ask_price) / 2.0


def pay_as_bid_price(bid_price: float, ask_price: float) -> float:
    """
    Buyer-price settlement.
    """
    bid_price, ask_price = _validate_price_inputs(bid_price, ask_price)
    return bid_price


def pay_as_ask_price(bid_price: float, ask_price: float) -> float:
    """
    Seller-price settlement.
    """
    bid_price, ask_price = _validate_price_inputs(bid_price, ask_price)
    return ask_price


def k_factor_price(bid_price: float, ask_price: float, k: float = 0.5) -> float:
    """
    Generalized price between ask and bid:
    p = ask + k * (bid - ask), k in [0,1]
    k=0   -> pay_as_ask
    k=0.5 -> midpoint
    k=1   -> pay_as_bid
    """
    bid_price, ask_price = _validate_price_inputs(bid_price, ask_price)

    k = float(k)
    if not (0.0 <= k <= 1.0):
        raise ValueError("k must be in [0, 1]")

    return ask_price + k * (bid_price - ask_price)


def weighted_midpoint_price(
    bid_price: float,
    ask_price: float,
    bid_quantity: float,
    ask_quantity: float,
) -> float:
    """
    Optional weighted midpoint using matched quantities.
    """
    bid_price, ask_price = _validate_price_inputs(bid_price, ask_price)

    bid_quantity = float(bid_quantity)
    ask_quantity = float(ask_quantity)

    if bid_quantity <= 0 or ask_quantity <= 0:
        raise ValueError("bid_quantity and ask_quantity must be > 0")

    total = bid_quantity + ask_quantity
    return (bid_price * bid_quantity + ask_price * ask_quantity) / total


def get_pricing_rule(name: str, **kwargs) -> Callable[[float, float], float]:
    key = str(name).strip().lower()

    if key == "midpoint":
        return midpoint_price

    if key in {"pay_as_bid", "bid", "buyer_price"}:
        return pay_as_bid_price

    if key in {"pay_as_ask", "ask", "seller_price"}:
        return pay_as_ask_price

    if key in {"k_factor", "kfactor", "k"}:
        k = kwargs.get("k", 0.5)
        return partial(k_factor_price, k=k)

    raise ValueError(
        f"Unsupported pricing rule {name!r}. "
        "Choose from: 'midpoint', 'pay_as_bid', 'pay_as_ask', 'k_factor'."
    )


# ---------------------------------------------------------------------
# CALL / uniform-price helpers
# ---------------------------------------------------------------------

def _candidate_prices_from_orders(bids: Sequence, asks: Sequence) -> list[float]:
    prices = {float(o.submitted_price) for o in bids}
    prices.update(float(o.submitted_price) for o in asks)
    return sorted(prices)


def demand_at_price(bids: Sequence, price: float) -> float:
    return float(
        sum(float(o.remaining_quantity) for o in bids if float(o.submitted_price) >= price)
    )


def supply_at_price(asks: Sequence, price: float) -> float:
    return float(
        sum(float(o.remaining_quantity) for o in asks if float(o.submitted_price) <= price)
    )


def find_call_market_clearing(bids: Sequence, asks: Sequence) -> dict:
    """
    CALL / periodic double auction:
    - evaluate candidate prices
    - cleared qty = min(demand(p), supply(p))
    - choose price(s) that maximize cleared qty
    - if tie, average tied candidate prices
    """
    if not bids or not asks:
        return {
            "clearing_price": None,
            "cleared_quantity": 0.0,
            "candidate_prices": [],
            "winning_prices": [],
        }

    candidate_prices = _candidate_prices_from_orders(bids, asks)

    best_qty = -1.0
    winning_prices: list[float] = []

    for p in candidate_prices:
        cleared = min(demand_at_price(bids, p), supply_at_price(asks, p))

        if cleared > best_qty + 1e-12:
            best_qty = cleared
            winning_prices = [p]
        elif abs(cleared - best_qty) <= 1e-12:
            winning_prices.append(p)

    if best_qty <= 0:
        return {
            "clearing_price": None,
            "cleared_quantity": 0.0,
            "candidate_prices": candidate_prices,
            "winning_prices": winning_prices,
        }

    clearing_price = sum(winning_prices) / len(winning_prices)

    return {
        "clearing_price": float(clearing_price),
        "cleared_quantity": float(best_qty),
        "candidate_prices": candidate_prices,
        "winning_prices": winning_prices,
    }


def select_orders_for_quantity(
    orders: Sequence,
    target_quantity: float,
    side: str,
) -> list[tuple[object, float]]:
    """
    Select orders by price-time priority until target quantity is filled.
    Returns list[(order, allocated_qty)].
    """
    target_quantity = float(target_quantity)

    if target_quantity <= 1e-12:
        return []

    if side == "buy":
        sorted_orders = sorted(
            orders,
            key=lambda o: (-float(o.submitted_price), int(o.submission_seq)),
        )
    elif side == "sell":
        sorted_orders = sorted(
            orders,
            key=lambda o: (float(o.submitted_price), int(o.submission_seq)),
        )
    else:
        raise ValueError("side must be 'buy' or 'sell'")

    selected: list[tuple[object, float]] = []
    remaining = target_quantity

    for order in sorted_orders:
        if remaining <= 1e-12:
            break

        qty = min(float(order.remaining_quantity), remaining)
        if qty > 1e-12:
            selected.append((order, qty))
            remaining -= qty

    return selected


def find_uniform_price_clearing(bids: Sequence, asks: Sequence) -> dict:
    """
    Uniform-price double-sided auction:
    1) determine the uniform clearing price from aggregate demand/supply
    2) determine the cleared quantity at the clearing price
    3) allocate accepted bids/asks by price-time priority
    """
    call_result = find_call_market_clearing(bids, asks)
    clearing_price = call_result["clearing_price"]

    if clearing_price is None:
        return {
            "clearing_price": None,
            "cleared_quantity": 0.0,
            "accepted_buys": [],
            "accepted_sells": [],
        }

    q_star = min(
        demand_at_price(bids, clearing_price),
        supply_at_price(asks, clearing_price),
    )

    if q_star <= 1e-12:
        return {
            "clearing_price": None,
            "cleared_quantity": 0.0,
            "accepted_buys": [],
            "accepted_sells": [],
        }

    eligible_bids = [
        o for o in bids
        if float(o.submitted_price) >= float(clearing_price)
    ]
    eligible_asks = [
        o for o in asks
        if float(o.submitted_price) <= float(clearing_price)
    ]

    accepted_buys = select_orders_for_quantity(eligible_bids, q_star, side="buy")
    accepted_sells = select_orders_for_quantity(eligible_asks, q_star, side="sell")

    return {
        "clearing_price": float(clearing_price),
        "cleared_quantity": float(q_star),
        "accepted_buys": accepted_buys,
        "accepted_sells": accepted_sells,
    }