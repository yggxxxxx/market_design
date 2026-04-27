from dataclasses import dataclass
import random

beta_range = (0.1, 0.5)
gamma_range = (0.0, 0.1)
margin_range = (0.05, 0.35)


c_range = (0.0, 0.03)


@dataclass
class MarketSignal:
    reference_price: float
    accepted: bool
    last_shout_type: str  # "bid" or "ask"


def limited_margin(value, lower=0.0, upper=0.5):
    return max(lower, min(value, upper))


def limited_price(price, fit_price, tou_price):
    return max(fit_price, min(price, tou_price))


def determine_zip_action(side, p_i, q, accepted, last_shout_type):
    if side not in {"buy", "sell"}:
        raise ValueError("Cannot define 'buyer' or 'seller'")
    if last_shout_type not in {"bid", "ask"}:
        raise ValueError("Cannot define last_shout_type: 'bid' or 'ask'")

 

    if side == "sell":
        if accepted:
            if p_i <= q:
                return "raise"
            if last_shout_type == "bid" and p_i >= q:
                return "lower"
            return "none"
        else:
      
            if last_shout_type == "ask" and p_i >= q:
                return "lower"
            return "none"

    if side == "buy":
        if accepted:
            if p_i >= q:
                return "raise"
            if last_shout_type == "ask" and p_i <= q:
                return "lower"
            return "none"
        else:
         
            if last_shout_type == "bid" and p_i <= q:
                return "lower"
            return "none"


class ZIPStrategy:
    def __init__(self, h_id, side, seed=None):
        if side not in {"buy", "sell"}:
            raise ValueError("Cannot define 'buyer' or 'seller'")

        random_num = random.Random(seed)

        self.h_id = h_id
        self.household_id = h_id
        self.side = side

        self.beta = random_num.uniform(*beta_range)
        self.gamma = random_num.uniform(*gamma_range)
        self.margin = random_num.uniform(*margin_range)
        self.c = random_num.uniform(*c_range)

        self.last_delta = 0.0
        self.fit_price = None
        self.tou_price = None

    def check_tariffs(self):
        if self.fit_price is None:
            raise ValueError("fit_price_gbp_per_kwh is not set")
        if self.tou_price is None:
            raise ValueError("tou_price_gbp_per_kwh is not set")
        if self.fit_price <= 0:
            raise ValueError("fit_price_gbp_per_kwh must be > 0")
        if self.tou_price <= 0:
            raise ValueError("tou_price_gbp_per_kwh must be > 0")
        if self.fit_price > self.tou_price:
            raise ValueError("fit_price_gbp_per_kwh must be <= tou_price_gbp_per_kwh")

    def price_from_margin(self, margin=None):
        self.check_tariffs()
        m = self.margin if margin is None else margin
        if self.side == "sell":
            p = self.fit_price * (1.0 + m)
        else:
            p = self.tou_price * (1.0 - m)
        return limited_price(p, self.fit_price, self.tou_price)

    def margin_from_price(self, price):
        self.check_tariffs()
        price = limited_price(price, self.fit_price, self.tou_price)
        if self.side == "sell":
            margin = (price / self.fit_price) - 1.0
        else:
            margin = 1.0 - (price / self.tou_price)
        return limited_margin(margin, 0.0, 0.5)

    def target_price(self, current_price, signal, action):
        q = signal.reference_price
        q = limited_price(q, self.fit_price, self.tou_price)

        if action == "none":
            return current_price

        if self.side == "sell":
            if action == "raise":
                base = max(current_price, q)
                tau = base * (1.0 + self.c)
            else:
                base = min(current_price, q)
                tau = base * (1.0 - self.c)
        else:
            # buy
            if action == "raise":
                base = min(current_price, q)
                tau = base * (1.0 - self.c)
            else:
                base = max(current_price, q)
                tau = base * (1.0 + self.c)

        return limited_price(tau, self.fit_price, self.tou_price)

    def generate_shout(self, fit_price, tou_price):
        if fit_price <= 0:
            raise ValueError("fit_price > 0")
        if tou_price <= 0:
            raise ValueError("tou_price > 0")
        if fit_price > tou_price:
            raise ValueError("fit_price must be <= tou_price")

        self.fit_price = fit_price
        self.tou_price = tou_price
        return self.price_from_margin()

    def update_from_market_signal(self, signal):
        self.check_tariffs()

        current_price = self.price_from_margin()

        action = determine_zip_action(
            side=self.side,
            p_i=current_price,
            q=signal.reference_price,
            accepted=signal.accepted,
            last_shout_type=signal.last_shout_type,
        )

        tau = self.target_price(
            current_price=current_price,
            signal=signal,
            action=action,
        )

        raw_delta = self.beta * (tau - current_price)
        delta = self.gamma * self.last_delta + (1.0 - self.gamma) * raw_delta

        new_price = current_price + delta
        new_price = limited_price(new_price, self.fit_price, self.tou_price)

        self.margin = self.margin_from_price(new_price)
        self.last_delta = delta
        return True

    def state_dict(self):
        return {
            "h_id": self.h_id,
            "side": self.side,
            "beta": self.beta,
            "gamma": self.gamma,
            "margin": self.margin,
            "c": self.c,
            "last_delta": self.last_delta,
            "fit_price": self.fit_price,
            "tou_price": self.tou_price,
        }