from __future__ import annotations

from dataclasses import dataclass

from core.committed_trade import gen_committed_trades, gen_unmatched_orders
from pricing.pricing_rules import find_uniform_price_clearing


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


class UniformPriceDoubleAuction:
    """
    Single-price double-sided auction.

    All accepted trades settle at the same clearing price.
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

        clearing = find_uniform_price_clearing(bids, asks)
        clearing_price = clearing["clearing_price"]
        cleared_quantity = clearing["cleared_quantity"]

        if self.verbose:
            print("\n=== UNIFORM-PRICE DOUBLE AUCTION ===")
            print(f"uniform_clearing_price = {clearing_price}")
            print(f"cleared_quantity       = {cleared_quantity:.6f} kWh")

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
                "market_name": "uniform_price",
            }

        accepted_buys = list(clearing["accepted_buys"])
        accepted_sells = list(clearing["accepted_sells"])

        bi = 0
        ai = 0

        while bi < len(accepted_buys) and ai < len(accepted_sells):
            bid, bid_qty = accepted_buys[bi]
            ask, ask_qty = accepted_sells[ai]

            qty = min(float(bid_qty), float(ask_qty))
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

            bid_qty -= qty
            ask_qty -= qty

            if bid_qty <= 1e-12:
                bi += 1
            else:
                accepted_buys[bi] = (bid, bid_qty)

            if ask_qty <= 1e-12:
                ai += 1
            else:
                accepted_sells[ai] = (ask, ask_qty)

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
            "market_name": "uniform_price",
        }