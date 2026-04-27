from dataclasses import dataclass


@dataclass
class CommittedTrade:
    trade_id: str
    buyer_h_id: int
    seller_h_id: int
    buyer_order_id: str
    seller_order_id: str
    DateTime: str
    hour: int
    quantity: float
    trade_price: float
    trade_value: float
    trade_round: int


@dataclass
class UnmatchedOrder:
    unmatched_order_id: str
    order_id: str
    h_id: int
    DateTime: str
    hour: int
    side: str
    original_quantity: float
    remaining_quantity: float
    limit_price: float
    submitted_price: float


def gen_committed_trades(matched_records):
    committed_trades = []

    for i, matched in enumerate(matched_records, start=1):
        trade = CommittedTrade(
            trade_id=f"T{i}",
            buyer_h_id=matched.buyer_h_id,
            seller_h_id=matched.seller_h_id,
            buyer_order_id=matched.buyer_order_id,
            seller_order_id=matched.seller_order_id,
            DateTime=matched.DateTime,
            hour=matched.hour,
            quantity=matched.quantity,
            trade_price=matched.matched_price,
            trade_value=matched.quantity * matched.matched_price,
            trade_round=matched.trade_round,
        )
        committed_trades.append(trade)

    return committed_trades


def gen_unmatched_orders(unmatched_orders):
    final_unmatched_orders = []

    for i, order in enumerate(unmatched_orders, start=1):
        u = UnmatchedOrder(
            unmatched_order_id=f"U{i}",
            order_id=order.order_id,
            h_id=order.h_id,
            DateTime=order.DateTime,
            hour=order.hour,
            side=order.side,
            original_quantity=order.quantity,
            remaining_quantity=order.remaining_quantity,
            limit_price=order.limit_price,
            submitted_price=order.submitted_price,
        )
        final_unmatched_orders.append(u)

    return final_unmatched_orders


def committed_trades_to_dicts(committed_trades):
    result = []

    for trade in committed_trades:
        d = {
            "trade_id": trade.trade_id,
            "buyer_h_id": trade.buyer_h_id,
            "seller_h_id": trade.seller_h_id,
            "buyer_order_id": trade.buyer_order_id,
            "seller_order_id": trade.seller_order_id,
            "DateTime": trade.DateTime,
            "hour": trade.hour,
            "quantity": trade.quantity,
            "trade_price": trade.trade_price,
            "trade_value": trade.trade_value,
            "trade_round": trade.trade_round,
        }
        result.append(d)

    return result


def unmatched_orders_to_dicts(unmatched_orders):
    result = []

    for order in unmatched_orders:
        d = {
            "unmatched_order_id": order.unmatched_order_id,
            "order_id": order.order_id,
            "h_id": order.h_id,
            "DateTime": order.DateTime,
            "hour": order.hour,
            "side": order.side,
            "original_quantity": order.original_quantity,
            "remaining_quantity": order.remaining_quantity,
            "limit_price_gbp_per": order.limit_price,
            "submitted_price": order.submitted_price,
        }
        result.append(d)

    return result