from __future__ import annotations

from dataclasses import dataclass

from core.committed_trade import gen_committed_trades, gen_unmatched_orders
from pricing.pricing_rules import find_call_market_clearing


@dataclass
class MatchedRecord:
    match_id: str
    buyer_h_id: str
    seller_h_id: str
    buyer_order_id: str
    seller_order_id: str
    DateTime: str
    hour: int
    quantity: float
    matched_price: float
    trade_round: int


class CallMarketMechanism:
    """
    Periodic / batch double auction.

    Orders are collected for the slot, then cleared once using a call price.
    """

    def __init__(self, order_book, trader_registry=None, verbose: bool = True):
        self.order_book = order_book
        self.trader_registry = trader_registry or {}
        self.verbose = verbose
        self.matched_records = []
        self.match_counter = 0

    def run_market(self):
        bids = self.order_book.all_bids()
        asks = self.order_book.all_asks()

        clearing = find_call_market_clearing(bids, asks)
        clearing_price = clearing["clearing_price"]
        cleared_quantity = clearing["cleared_quantity"]

        if self.verbose:
            print("\n=== CALL MARKET ===")
            print(f"candidate_prices = {clearing['candidate_prices']}")
            print(f"winning_prices   = {clearing['winning_prices']}")
            print(f"call_price       = {clearing_price}")
            print(f"cleared_quantity = {cleared_quantity:.6f} kWh")

        if clearing_price is None or cleared_quantity <= 1e-12:
            unmatched_orders = self.order_book.all_bids() + self.order_book.all_asks()
            final_unmatched = gen_unmatched_orders(unmatched_orders)

            return {
                "matched_records": [],
                "committed_trades": [],
                "unmatched_orders": final_unmatched,
                "num_trades": 0,
                "num_unmatched_orders": len(final_unmatched),
                "clearing_price": None,
                "cleared_quantity": 0.0,
                "market_name": "call",
            }

        eligible_bids = [o for o in bids if float(o.submitted_price) >= float(clearing_price)]
        eligible_asks = [o for o in asks if float(o.submitted_price) <= float(clearing_price)]

        eligible_bids.sort(key=lambda o: (-float(o.submitted_price), int(o.submission_seq)))
        eligible_asks.sort(key=lambda o: (float(o.submitted_price), int(o.submission_seq)))

        target_remaining = float(cleared_quantity)
        bi = 0
        ai = 0

        while target_remaining > 1e-12 and bi < len(eligible_bids) and ai < len(eligible_asks):
            bid = eligible_bids[bi]
            ask = eligible_asks[ai]

            qty = min(
                float(bid.remaining_quantity),
                float(ask.remaining_quantity),
                target_remaining,
            )

            if qty <= 1e-12:
                break

            self.match_counter += 1
            self.matched_records.append(
                MatchedRecord(
                    match_id=f"M{self.match_counter}",
                    buyer_h_id=bid.h_id,
                    seller_h_id=ask.h_id,
                    buyer_order_id=bid.order_id,
                    seller_order_id=ask.order_id,
                    DateTime=bid.DateTime,
                    hour=bid.hour,
                    quantity=qty,
                    matched_price=float(clearing_price),
                    trade_round=1,
                )
            )

            bid.remaining_quantity -= qty
            ask.remaining_quantity -= qty
            target_remaining -= qty

            if bid.remaining_quantity <= 1e-12:
                bi += 1
            if ask.remaining_quantity <= 1e-12:
                ai += 1

        self.order_book.remove_finished_orders()

        committed_trades = gen_committed_trades(self.matched_records)
        unmatched_orders = self.order_book.all_bids() + self.order_book.all_asks()
        final_unmatched = gen_unmatched_orders(unmatched_orders)

        return {
            "matched_records": self.matched_records,
            "committed_trades": committed_trades,
            "unmatched_orders": final_unmatched,
            "num_trades": len(committed_trades),
            "num_unmatched_orders": len(final_unmatched),
            "clearing_price": float(clearing_price),
            "cleared_quantity": float(cleared_quantity),
            "market_name": "call",
        }