from dataclasses import dataclass, field


@dataclass
class Order:
    order_id: str
    h_id: int
    trader_key: tuple   # (h_id, side)
    DateTime: str
    hour: int
    side: str
    quantity: float
    remaining_quantity: float
    limit_price: float
    submitted_price: float
    submission_seq: int = field(default=0)

    def __post_init__(self):
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side}")

        if self.quantity < 0:
            raise ValueError("quantity_kwh must be >= 0")

        if self.remaining_quantity < 0:
            raise ValueError("remaining_quantity_kwh must be >= 0")

        if self.limit_price < 0:
            raise ValueError("limit_price_gbp_per_kwh must be >= 0")

        if self.submitted_price < 0:
            raise ValueError("submitted_price_gbp_per_kwh must be >= 0")


class OrderBook:

    def __init__(self):
        self.bids = []
        self.asks = []
        self.submission_counter = 0

    def add_order(self, order):
        self.submission_counter += 1
        order.submission_seq = self.submission_counter

        if order.side == "buy":
            self.bids.append(order)
            self.sort_bids()
        else:
            self.asks.append(order)
            self.sort_asks()

    def sort_bids(self):
        self.bids.sort(
            key=lambda o: (-o.submitted_price, o.submission_seq)
        )

    def sort_asks(self):
        self.asks.sort(
            key=lambda o: (o.submitted_price, o.submission_seq)
        )

    def sort_orderbook(self):
        self.sort_bids()
        self.sort_asks()

    def best_bid(self):
        if self.bids:
            return self.bids[0]
        return None

    def best_ask(self):
        if self.asks:
            return self.asks[0]
        return None

    def match_order(self):
        best_bid = self.best_bid()
        best_ask = self.best_ask()

        if best_bid is None or best_ask is None:
            return False

        return best_bid.submitted_price >= best_ask.submitted_price

    def remove_finished_orders(self):
        self.bids = [o for o in self.bids if o.remaining_quantity > 1e-12]
        self.asks = [o for o in self.asks if o.remaining_quantity > 1e-12]

    def all_bids(self):
        return list(self.bids)

    def all_asks(self):
        return list(self.asks)

    def summary(self):
        best_bid_price = self.bids[0].submitted_price if self.bids else None
        best_ask_price = self.asks[0].submitted_price if self.asks else None

        return {
            "num_bids": len(self.bids),
            "num_asks": len(self.asks),
            "best_bid": best_bid_price,
            "best_ask": best_ask_price,
        }

    def print_book(self):
        print("=== BIDS ===")
        if not self.bids:
            print("[empty]")
        for order in self.bids:
            print(
                f"order_id={order.order_id}, h_id={order.h_id}, "
                f"price={order.submitted_price:.4f} GBP/kWh, "
                f"remaining_quantity={order.remaining_quantity:.4f} kWh, "
                f"submission_seq={order.submission_seq}"
            )

        print("\n=== ASKS ===")
        if not self.asks:
            print("[empty]")
        for order in self.asks:
            print(
                f"order_id={order.order_id}, h_id={order.h_id}, "
                f"price={order.submitted_price:.4f} GBP/kWh, "
                f"remaining_quantity={order.remaining_quantity:.4f} kWh, "
                f"submission_seq={order.submission_seq}"
            )