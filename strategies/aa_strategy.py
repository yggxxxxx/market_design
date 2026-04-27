from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Optional


def limited_price(price: float, fit_price: float, tou_price: float) -> float:
    return max(fit_price, min(price, tou_price))


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


@dataclass
class MarketSignal:
    reference_price: float
    accepted: bool
    last_shout_type: str


class AAStrategy:
    """
    Tariff-bounded AA-style strategy for your microgrid CDA.

    Notes
    -----
    - accepted=True:
        buyer sees transaction price with last_shout_type="ask"
        seller sees transaction price with last_shout_type="bid"

    - accepted=False:
        this AA implementation expects blocked opposite-side information:
        buyer  -> best ask  + "ask"
        seller -> best bid  + "bid"

    cda.py will use `signal_mode="aa"` to send the correct no-trade signal.
    """

    signal_mode = "aa"

    def __init__(
        self,
        h_id,
        side: str,
        seed=None,  # kept for interface compatibility
        beta_r: float = 0.4,
        beta_theta: float = 0.4,
        lambda_rel: float = 0.01,
        lambda_abs: float = 0.02,
        theta_init: float = -4.0,
        theta_min: float = -8.0,
        theta_max: float = 2.0,
        history_window: int = 8,
        rho: float = 0.9,
        gamma: float = 2.0,
        alpha_min: float = 0.0,
        alpha_max: float = 0.15,
        eta: float = 3.0,
        warmup_trades_for_theta: int = 3,
    ):
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if beta_r <= 0 or beta_r >= 1:
            raise ValueError("beta_r must be in (0, 1)")
        if beta_theta <= 0 or beta_theta >= 1:
            raise ValueError("beta_theta must be in (0, 1)")
        if lambda_rel < 0 or lambda_abs < 0:
            raise ValueError("lambda_rel and lambda_abs must be >= 0")
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        if not (0 < rho <= 1):
            raise ValueError("rho must be in (0, 1]")
        if eta < 1:
            raise ValueError("eta must be >= 1")
        if alpha_max <= alpha_min:
            raise ValueError("alpha_max must be > alpha_min")

        self.h_id = h_id
        self.side = side

        self.beta_r = float(beta_r)
        self.beta_theta = float(beta_theta)
        self.lambda_rel = float(lambda_rel)
        self.lambda_abs = float(lambda_abs)

        self.theta = float(theta_init)
        self.theta_init = float(theta_init)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        self.history_window = int(history_window)
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.eta = float(eta)
        self.warmup_trades_for_theta = int(warmup_trades_for_theta)

        # r < 0 aggressive, r = 0 active, r > 0 passive
        self.r = 0.0

        self.fit_price: Optional[float] = None
        self.tou_price: Optional[float] = None

        self.limit_price: Optional[float] = None

        self.trade_history = deque(maxlen=self.history_window)
        self.p_hat: Optional[float] = None

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------

    def _price_floor(self) -> float:
        if self.fit_price is None:
            raise ValueError("fit_price not set")
        return self.fit_price

    def _price_ceiling(self) -> float:
        if self.tou_price is None:
            raise ValueError("tou_price not set")
        return self.tou_price

    def _default_limit_price(self) -> float:
        if self.side == "buy":
            return self._price_ceiling()
        return self._price_floor()

    def _current_limit_price(self) -> float:
        if self.limit_price is not None:
            return self.limit_price
        return self._default_limit_price()

    def _get_p_hat(self) -> float:
        if self.p_hat is not None:
            return self.p_hat
        return (self._price_floor() + self._price_ceiling()) / 2.0

    def _set_market_bounds(
        self,
        fit_price: float,
        tou_price: float,
        limit_price: Optional[float] = None,
    ) -> None:
        if fit_price <= 0:
            raise ValueError("fit_price must be > 0")
        if tou_price <= 0:
            raise ValueError("tou_price must be > 0")
        if fit_price > tou_price:
            raise ValueError("fit_price must be <= tou_price")

        self.fit_price = float(fit_price)
        self.tou_price = float(tou_price)

        if limit_price is None:
            self.limit_price = None
        else:
            self.limit_price = limited_price(float(limit_price), self.fit_price, self.tou_price)

    # ------------------------------------------------------------------
    # equilibrium estimator
    # ------------------------------------------------------------------

    def _update_p_hat(self, trade_price: float) -> None:
        trade_price = limited_price(float(trade_price), self._price_floor(), self._price_ceiling())
        self.trade_history.append(trade_price)

        values = list(self.trade_history)
        n = len(values)

        weights = []
        for i in range(n):
            age = (n - 1) - i
            weights.append(self.rho ** age)

        weight_sum = sum(weights)
        if weight_sum <= 0:
            return

        self.p_hat = sum(w * p for w, p in zip(weights, values)) / weight_sum
        self.p_hat = limited_price(self.p_hat, self._price_floor(), self._price_ceiling())

    # ------------------------------------------------------------------
    # shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _f_theta(x: float, theta: float) -> float:
        x = clip(float(x), 0.0, 1.0)

        if abs(theta) < 1e-8:
            return x

        denom = math.exp(theta) - 1.0
        if abs(denom) < 1e-12:
            return x

        value = (math.exp(theta * x) - 1.0) / denom
        return clip(value, 0.0, 1.0)

    @staticmethod
    def _f_theta_inv(y: float, theta: float) -> float:
        y = clip(float(y), 0.0, 1.0)

        if abs(theta) < 1e-8:
            return y

        base = 1.0 + y * (math.exp(theta) - 1.0)
        base = max(base, 1e-12)

        value = math.log(base) / theta
        return clip(value, 0.0, 1.0)

    # ------------------------------------------------------------------
    # trader type
    # ------------------------------------------------------------------

    def _trader_type(self) -> str:
        p_hat = self._get_p_hat()
        limit_price = self._current_limit_price()

        if self.side == "buy":
            return "intra" if limit_price >= p_hat else "extra"
        return "intra" if limit_price <= p_hat else "extra"

    # ------------------------------------------------------------------
    # r -> tau
    # ------------------------------------------------------------------

    def _target_price_from_r(self) -> float:
        p_hat = self._get_p_hat()
        floor_ = self._price_floor()
        ceil_ = self._price_ceiling()
        limit_price = self._current_limit_price()
        trader_type = self._trader_type()

        f_neg = self._f_theta(-self.r, self.theta) if self.r < 0 else 0.0
        f_pos = self._f_theta(self.r, self.theta) if self.r > 0 else 0.0

        if self.side == "buy":
            if trader_type == "intra":
                if self.r <= 0:
                    tau = p_hat + (limit_price - p_hat) * f_neg
                else:
                    tau = floor_ + (p_hat - floor_) * (1.0 - f_pos)
            else:
                if self.r <= 0:
                    tau = limit_price
                else:
                    tau = floor_ + (limit_price - floor_) * (1.0 - f_pos)

        else:
            if trader_type == "intra":
                if self.r <= 0:
                    tau = limit_price + (p_hat - limit_price) * (1.0 - f_neg)
                else:
                    tau = p_hat + (ceil_ - p_hat) * f_pos
            else:
                if self.r <= 0:
                    tau = limit_price
                else:
                    tau = limit_price + (ceil_ - limit_price) * f_pos

        return limited_price(tau, floor_, ceil_)

    # ------------------------------------------------------------------
    # price -> r
    # ------------------------------------------------------------------

    def _r_from_price(self, price: float) -> float:
        price = limited_price(float(price), self._price_floor(), self._price_ceiling())

        p_hat = self._get_p_hat()
        floor_ = self._price_floor()
        ceil_ = self._price_ceiling()
        limit_price = self._current_limit_price()
        trader_type = self._trader_type()

        eps = 1e-12

        if self.side == "buy":
            if trader_type == "intra":
                if price >= p_hat:
                    denom = max(limit_price - p_hat, eps)
                    y = (price - p_hat) / denom
                    return -self._f_theta_inv(y, self.theta)

                denom = max(p_hat - floor_, eps)
                y = (p_hat - price) / denom
                return self._f_theta_inv(y, self.theta)

            if price >= limit_price - eps:
                return 0.0

            denom = max(limit_price - floor_, eps)
            y = (limit_price - price) / denom
            return self._f_theta_inv(y, self.theta)

        if trader_type == "intra":
            if price <= p_hat:
                denom = max(p_hat - limit_price, eps)
                y = (p_hat - price) / denom
                return -self._f_theta_inv(y, self.theta)

            denom = max(ceil_ - p_hat, eps)
            y = (price - p_hat) / denom
            return self._f_theta_inv(y, self.theta)

        if price <= limit_price + eps:
            return 0.0

        denom = max(ceil_ - limit_price, eps)
        y = (price - limit_price) / denom
        return self._f_theta_inv(y, self.theta)

    # ------------------------------------------------------------------
    # short-term learning
    # ------------------------------------------------------------------

    def _desired_delta(self, r_shout: float, direction: str) -> float:
        if direction == "more_aggressive":
            delta = (1.0 - self.lambda_rel) * r_shout - self.lambda_abs
        elif direction == "less_aggressive":
            delta = (1.0 + self.lambda_rel) * r_shout + self.lambda_abs
        else:
            raise ValueError("direction must be 'more_aggressive' or 'less_aggressive'")

        return clip(delta, -1.0, 1.0)

    def _update_r(self, delta: float) -> None:
        self.r = self.r + self.beta_r * (delta - self.r)
        self.r = clip(self.r, -1.0, 1.0)

    # ------------------------------------------------------------------
    # long-term learning
    # ------------------------------------------------------------------

    def _alpha_from_trade_history(self) -> Optional[float]:
        if len(self.trade_history) < 2:
            return None

        p_hat = self._get_p_hat()
        if p_hat <= 0:
            return None

        values = list(self.trade_history)
        mean_sq = sum((p - p_hat) ** 2 for p in values) / len(values)
        alpha = math.sqrt(mean_sq) / p_hat
        return max(0.0, alpha)

    def _theta_star(self, alpha: float) -> float:
        alpha_bar = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
        alpha_bar = clip(alpha_bar, 0.0, 1.0)

        theta_star = (
            (self.theta_max - self.theta_min)
            * (1.0 - alpha_bar * math.exp(self.gamma * (alpha_bar - 1.0)))
            + self.theta_min
        )
        return clip(theta_star, self.theta_min, self.theta_max)

    def _update_theta_from_volatility(self) -> None:
        if len(self.trade_history) < self.warmup_trades_for_theta:
            return

        alpha = self._alpha_from_trade_history()
        if alpha is None:
            return

        theta_target = self._theta_star(alpha)
        self.theta = self.theta + self.beta_theta * (theta_target - self.theta)
        self.theta = clip(self.theta, self.theta_min, self.theta_max)

    # ------------------------------------------------------------------
    # optional bidding layer
    # ------------------------------------------------------------------

    def _paper_style_bidding_layer(
        self,
        tau: float,
        outstanding_bid: float,
        outstanding_ask: float,
        is_first_round: bool,
    ) -> Optional[float]:
        floor_ = self._price_floor()
        ceil_ = self._price_ceiling()
        limit_price = self._current_limit_price()

        obid = limited_price(float(outstanding_bid), floor_, ceil_)
        oask = limited_price(float(outstanding_ask), floor_, ceil_)

        if self.side == "buy":
            if limit_price <= obid:
                return None

            if is_first_round:
                oask_plus = (1.0 + self.lambda_rel) * oask + self.lambda_abs
                bid = obid + (min(limit_price, oask_plus) - obid) / self.eta
                return limited_price(bid, floor_, ceil_)

            if oask <= tau:
                return limited_price(min(oask, limit_price), floor_, ceil_)

            bid = obid + (tau - obid) / self.eta
            return limited_price(bid, floor_, ceil_)

        if limit_price >= oask:
            return None

        if is_first_round:
            obid_minus = (1.0 - self.lambda_rel) * obid - self.lambda_abs
            ask = oask - (oask - max(limit_price, obid_minus)) / self.eta
            return limited_price(ask, floor_, ceil_)

        if obid >= tau:
            return limited_price(max(obid, limit_price), floor_, ceil_)

        ask = oask - (oask - tau) / self.eta
        return limited_price(ask, floor_, ceil_)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate_shout(
        self,
        fit_price: float,
        tou_price: float,
        limit_price: Optional[float] = None,
        outstanding_bid: Optional[float] = None,
        outstanding_ask: Optional[float] = None,
        is_first_round: bool = False,
    ) -> Optional[float]:
        self._set_market_bounds(fit_price, tou_price, limit_price=limit_price)

        tau = self._target_price_from_r()

        if outstanding_bid is None or outstanding_ask is None:
            return tau

        return self._paper_style_bidding_layer(
            tau=tau,
            outstanding_bid=outstanding_bid,
            outstanding_ask=outstanding_ask,
            is_first_round=is_first_round,
        )

    def update_from_market_signal(self, signal: MarketSignal) -> bool:
        if self.fit_price is None or self.tou_price is None:
            return False

        q = limited_price(float(signal.reference_price), self._price_floor(), self._price_ceiling())
        current_tau = self._target_price_from_r()

        direction: Optional[str] = None

        if self.side == "buy":
            if signal.accepted:
                if current_tau >= q:
                    direction = "less_aggressive"
                else:
                    direction = "more_aggressive"
            else:
                if signal.last_shout_type == "ask" and current_tau <= q:
                    direction = "more_aggressive"

        else:
            if signal.accepted:
                if current_tau <= q:
                    direction = "less_aggressive"
                else:
                    direction = "more_aggressive"
            else:
                if signal.last_shout_type == "bid" and current_tau >= q:
                    direction = "more_aggressive"

        if direction is not None:
            r_shout = self._r_from_price(q)
            delta = self._desired_delta(r_shout, direction=direction)
            self._update_r(delta)

        if signal.accepted:
            self._update_p_hat(q)
            self._update_theta_from_volatility()

        return True

    def state_dict(self) -> dict:
        return {
            "strategy": "aa",
            "h_id": self.h_id,
            "side": self.side,
            "r": self.r,
            "theta": self.theta,
            "p_hat": self.p_hat,
            "limit_price": self.limit_price,
            "fit_price": self.fit_price,
            "tou_price": self.tou_price,
            "trade_history": list(self.trade_history),
            "trader_type": self._trader_type() if self.fit_price is not None else None,
        }