def apply_cost(gross_return: float, transaction_cost_bps: float) -> float:
    return gross_return - transaction_cost_bps / 10000.0
