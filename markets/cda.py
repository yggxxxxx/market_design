from dataclasses import dataclass

from zip_strategy import MarketSignal
from committed_trade import gen_committed_trades, gen_unmatched_orders


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


class CDA_mechanism:
    def __init__(
        self,
        order_book,
        trader_registry,
        max_trade_rounds=150,
        max_no_trade_rounds=50,
        verbose=True,
    ):
        self.order_book = order_book
        self.trader_registry = trader_registry
        self.max_trade_rounds = max_trade_rounds
        self.max_no_trade_rounds = max_no_trade_rounds
        self.verbose = verbose
        self.matched_records = []
        self.match_counter = 0

    def run_cda(self):
        trade_round = 0
        no_trade_rounds = 0
        total_round = 0

        while True:
            total_round += 1

            best_bid = self.order_book.best_bid()
            best_ask = self.order_book.best_ask()

            if best_bid is None or best_ask is None:
                if self.verbose:
                    print("\n=== MARKET STOP ===")
                    print("No more orders on one side of the market.")
                break

            if self.verbose:
                self.print_header(total_round, no_trade_rounds)
                self.print_order_book("ORDER BOOK BEFORE MATCHING")

            if best_bid.submitted_price >= best_ask.submitted_price:
                if trade_round >= self.max_trade_rounds:
                    if self.verbose:
                        print("\n=== MARKET STOP ===")
                        print(f"Reached max_trade_rounds = {self.max_trade_rounds}")
                    break

                trade_round += 1
                no_trade_rounds = 0

                matched_price = self.trade_price(best_bid, best_ask)
                matched_quantity = min(
                    best_bid.remaining_quantity,
                    best_ask.remaining_quantity
                )

                if self.verbose:
                    print("\nONE TRADE MATCHED")
                    print(
                        f"buyer_order = {best_bid.order_id} (h_id={best_bid.h_id}, "
                        f"price={best_bid.submitted_price:.3f} GBP/kWh, "
                        f"remaining_quantity={best_bid.remaining_quantity:.3f} kWh)"
                    )
                    print(
                        f"seller_order = {best_ask.order_id} (h_id={best_ask.h_id}, "
                        f"price={best_ask.submitted_price:.3f} GBP/kWh, "
                        f"remaining_quantity={best_ask.remaining_quantity:.3f} kWh)"
                    )
                    print(f"matched_quantity = {matched_quantity:.3f} kWh")
                    print(f"trade_price = {matched_price:.4f} GBP/kWh")

                self.match_order(
                    best_bid=best_bid,
                    best_ask=best_ask,
                    matched_price=matched_price,
                    trade_round=trade_round,
                )

                self.order_book.remove_finished_orders()

                if self.verbose:
                    print("\nUPDATING ORDERS AFTER TRADE")

                self.update_order(matched_price)
                self.order_book.sort_orderbook()

            else:
                no_trade_rounds += 1

                if self.verbose:
                    print("\nNO TRADE MATCH")
                    print(
                        f"best_bid_price = {best_bid.submitted_price:.4f} GBP/kWh, "
                        f"best_ask_price = {best_ask.submitted_price:.4f} GBP/kWh"
                    )
                    print("Update orders using latest market shout.")

                if no_trade_rounds >= self.max_no_trade_rounds:
                    if self.verbose:
                        print("\n=== MARKET STOP ===")
                        print(f"Reached max_no_trade_rounds = {self.max_no_trade_rounds}")
                    break

                updated_any = self.update_orders_without_trade()

                self.order_book.remove_finished_orders()
                self.order_book.sort_orderbook()

                if not updated_any:
                    if self.verbose:
                        print("\n=== MARKET STOP ===")
                        print("No order was updated in the no-trade round.")
                    break

        committed_trades = gen_committed_trades(self.matched_records)
        unmatched_orders = self.order_book.all_bids() + self.order_book.all_asks()
        final_unmatched_orders = gen_unmatched_orders(unmatched_orders)

        return {
            "matched_records": self.matched_records,
            "committed_trades": committed_trades,
            "unmatched_orders": final_unmatched_orders,
            "num_trades": len(committed_trades),
            "num_unmatched_orders": len(final_unmatched_orders),
        }

    def match_order(
        self,
        best_bid,
        best_ask,
        matched_price,
        trade_round
    ):
        matched_quantity = min(
            best_bid.remaining_quantity,
            best_ask.remaining_quantity
        )

        self.match_counter += 1

        self.matched_records.append(
            MatchedRecord(
                match_id=f"M{self.match_counter}",
                buyer_h_id=best_bid.h_id,
                seller_h_id=best_ask.h_id,
                buyer_order_id=best_bid.order_id,
                seller_order_id=best_ask.order_id,
                DateTime=best_bid.DateTime,
                hour=best_bid.hour,
                quantity=matched_quantity,
                matched_price=matched_price,
                trade_round=trade_round,
            )
        )

        best_bid.remaining_quantity -= matched_quantity
        best_ask.remaining_quantity -= matched_quantity

    def trade_price(self, best_bid, best_ask):
        return (best_bid.submitted_price + best_ask.submitted_price) / 2.0

    def _signal_mode(self, strategy):
        """
        signal routing:
        - AA      -> no-trade uses opposite-side blocking signal
        - ZIP     -> keep current ZIP-style no-trade signal
        - Static  -> treated like ZIP path, but it ignores updates anyway
        """
        mode = getattr(strategy, "signal_mode", None)
        if mode is not None:
            return str(mode).strip().lower()

        class_name = strategy.__class__.__name__.lower()
        if "aa" in class_name:
            return "aa"

        return "zip"

    def _build_signal(
        self,
        strategy,
        side,
        accepted,
        trade_price=None,
        reference_bid_price=None,
        reference_ask_price=None,
    ):
        mode = self._signal_mode(strategy)

        # after-trade signal: common across strategies in this project
        if accepted:
            if side == "buy":
                return MarketSignal(
                    reference_price=trade_price,
                    accepted=True,
                    last_shout_type="ask",
                )
            return MarketSignal(
                reference_price=trade_price,
                accepted=True,
                last_shout_type="bid",
            )

        # no-trade signal: AA uses blocked opposite-side signal
        if mode == "aa":
            if side == "buy":
                return MarketSignal(
                    reference_price=reference_ask_price,
                    accepted=False,
                    last_shout_type="ask",
                )
            return MarketSignal(
                reference_price=reference_bid_price,
                accepted=False,
                last_shout_type="bid",
            )

        # no-trade signal: ZIP / Static keep current project logic
        if side == "buy":
            return MarketSignal(
                reference_price=reference_bid_price,
                accepted=False,
                last_shout_type="bid",
            )
        return MarketSignal(
            reference_price=reference_ask_price,
            accepted=False,
            last_shout_type="ask",
        )

    def update_order(self, trade_price):
        for bid_order in self.order_book.all_bids():
            if bid_order.remaining_quantity <= 0:
                continue

            strategy = self.trader_registry.get(bid_order.trader_key)
            if strategy is None:
                continue

            old_price = bid_order.submitted_price

            signal = self._build_signal(
                strategy=strategy,
                side="buy",
                accepted=True,
                trade_price=trade_price,
            )
            strategy.update_from_market_signal(signal)

            new_price = strategy.generate_shout(
                fit_price=strategy.fit_price,
                tou_price=strategy.tou_price
            )

            bid_order.submitted_price = min(new_price, bid_order.limit_price)

            if self.verbose:
                print(
                    f"[BUY UPDATE AFTER TRADE] h_id={bid_order.h_id}, order_id={bid_order.order_id}, "
                    f"remaining_quantity={bid_order.remaining_quantity:.3f} kWh, "
                    f"price: {old_price:.3f} -> {bid_order.submitted_price:.3f} GBP/kWh"
                )

        for ask_order in self.order_book.all_asks():
            if ask_order.remaining_quantity <= 0:
                continue

            strategy = self.trader_registry.get(ask_order.trader_key)
            if strategy is None:
                continue

            old_price = ask_order.submitted_price

            signal = self._build_signal(
                strategy=strategy,
                side="sell",
                accepted=True,
                trade_price=trade_price,
            )
            strategy.update_from_market_signal(signal)

            new_price = strategy.generate_shout(
                fit_price=strategy.fit_price,
                tou_price=strategy.tou_price
            )

            ask_order.submitted_price = max(new_price, ask_order.limit_price)

            if self.verbose:
                print(
                    f"[SELL UPDATE AFTER TRADE] h_id={ask_order.h_id}, order_id={ask_order.order_id}, "
                    f"remaining_quantity={ask_order.remaining_quantity:.3f} kWh, "
                    f"price: {old_price:.3f} -> {ask_order.submitted_price:.3f} GBP/kWh"
                )

    def update_orders_without_trade(self):
        best_bid = self.order_book.best_bid()
        best_ask = self.order_book.best_ask()

        if best_bid is None or best_ask is None:
            return False

        updated_any = False
        reference_bid_price = best_bid.submitted_price
        reference_ask_price = best_ask.submitted_price

        for bid_order in self.order_book.all_bids():
            if bid_order.remaining_quantity <= 0:
                continue

            strategy = self.trader_registry.get(bid_order.trader_key)
            if strategy is None:
                continue

            old_price = bid_order.submitted_price

            signal = self._build_signal(
                strategy=strategy,
                side="buy",
                accepted=False,
                reference_bid_price=reference_bid_price,
                reference_ask_price=reference_ask_price,
            )
            strategy.update_from_market_signal(signal)

            new_price = strategy.generate_shout(
                fit_price=strategy.fit_price,
                tou_price=strategy.tou_price
            )

            bid_order.submitted_price = min(new_price, bid_order.limit_price)
            updated_any = True

            if self.verbose:
                print(
                    f"[BUY UPDATE NO-TRADE] h_id={bid_order.h_id}, order_id={bid_order.order_id}, "
                    f"remaining_quantity={bid_order.remaining_quantity:.3f} kWh, "
                    f"price: {old_price:.3f} -> {bid_order.submitted_price:.3f} GBP/kWh, "
                    f"reference_best_bid={reference_bid_price:.3f} GBP/kWh, "
                    f"reference_best_ask={reference_ask_price:.3f} GBP/kWh"
                )

        for ask_order in self.order_book.all_asks():
            if ask_order.remaining_quantity <= 0:
                continue

            strategy = self.trader_registry.get(ask_order.trader_key)
            if strategy is None:
                continue

            old_price = ask_order.submitted_price

            signal = self._build_signal(
                strategy=strategy,
                side="sell",
                accepted=False,
                reference_bid_price=reference_bid_price,
                reference_ask_price=reference_ask_price,
            )
            strategy.update_from_market_signal(signal)

            new_price = strategy.generate_shout(
                fit_price=strategy.fit_price,
                tou_price=strategy.tou_price
            )

            ask_order.submitted_price = max(new_price, ask_order.limit_price)
            updated_any = True

            if self.verbose:
                print(
                    f"[SELL UPDATE NO-TRADE] h_id={ask_order.h_id}, order_id={ask_order.order_id}, "
                    f"remaining_quantity={ask_order.remaining_quantity:.3f} kWh, "
                    f"price: {old_price:.3f} -> {ask_order.submitted_price:.3f} GBP/kWh, "
                    f"reference_best_bid={reference_bid_price:.3f} GBP/kWh, "
                    f"reference_best_ask={reference_ask_price:.3f} GBP/kWh"
                )

        return updated_any

    def print_header(self, total_round, no_trade_rounds):
        print("\n" + "=" * 70)
        print(f"TOTAL ROUND {total_round}")
        print(f"no_trade_rounds = {no_trade_rounds}")
        print("=" * 70)

    def print_order_book(self, title):
        print(f"\n=== {title} ===")

        print("BIDS")
        if not self.order_book.all_bids():
            print("[empty]")

        for order in self.order_book.all_bids():
            print(
                f"order_id={order.order_id}, h_id={order.h_id}, "
                f"price={order.submitted_price:.3f} GBP/kWh, "
                f"remaining_quantity={order.remaining_quantity:.3f} kWh, "
                f"submission_seq={order.submission_seq}"
            )

        print("\nASKS")
        if not self.order_book.all_asks():
            print("[empty]")

        for order in self.order_book.all_asks():
            print(
                f"order_id={order.order_id}, h_id={order.h_id}, "
                f"price={order.submitted_price:.3f} GBP/kWh, "
                f"remaining_quantity={order.remaining_quantity:.3f} kWh, "
                f"submission_seq={order.submission_seq}"
            )