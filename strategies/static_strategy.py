import random


def limited_price(price, fit_price, tou_price):
    return max(fit_price, min(price, tou_price))


class StaticStrategy:
   

    def __init__(self, h_id, side, seed=None, margin_range=(0.05, 0.35)):
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        self.h_id = h_id
        self.side = side
        self.rng = random.Random(seed)
        self.margin_range = margin_range

        self.fit_price = None
        self.tou_price = None

        self.current_slot_key = None
        self.slot_margin = None

    def set_slot_context(self, slot_key):
        
        if self.current_slot_key != slot_key:
            self.current_slot_key = slot_key
            self.slot_margin = self.rng.uniform(*self.margin_range)

    def _ensure_slot_margin(self):
        if self.slot_margin is None:
            self.slot_margin = self.rng.uniform(*self.margin_range)

    def generate_shout(self, fit_price, tou_price):
        if fit_price <= 0:
            raise ValueError("fit_price must be > 0")
        if tou_price <= 0:
            raise ValueError("tou_price must be > 0")
        if fit_price > tou_price:
            raise ValueError("fit_price must be <= tou_price")

        self.fit_price = fit_price
        self.tou_price = tou_price

        self._ensure_slot_margin()

        if self.side == "sell":
            price = fit_price * (1.0 + self.slot_margin)
        else:
            price = tou_price * (1.0 - self.slot_margin)

        return limited_price(price, fit_price, tou_price)

    def update_from_market_signal(self, signal):
        
        return False

    def state_dict(self):
        return {
            "strategy": "static",
            "h_id": self.h_id,
            "side": self.side,
            "fit_price": self.fit_price,
            "tou_price": self.tou_price,
            "current_slot_key": self.current_slot_key,
            "slot_margin": self.slot_margin,
        }