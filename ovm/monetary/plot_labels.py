DEVIATION_LABEL_START = 'd-'
SPOT_PRICE_LABEL_START = 's-'
FUTURES_PRICE_LABEL_START = 'f-'
SKEW_LABEL_START = 'Skew'
RESERVE_SKEW_LABEL_START = 'Reserve Skew'
NOTIONAL_SKEW_LABEL_START = 'Notional Skew'
OPEN_POSITIONS_LABEL_START = 'Number of Open Positions on'
FUNDING_LABEL_START = 'Accumulated Funding'
AVG_COST_LABEL_START = 'Average Cost'
UNREALIZED_PNL_LABEL_START = 'Unrealized PnL'


def price_deviation_label(ticker: str) -> str:
    return f"{DEVIATION_LABEL_START}{ticker}"


def spot_price_label(ticker: str) -> str:
    return f"{SPOT_PRICE_LABEL_START}{ticker}"


def futures_price_label(ticker: str) -> str:
    return f"{FUTURES_PRICE_LABEL_START}{ticker}"


def skew_label(ticker: str) -> str:
    return f"{SKEW_LABEL_START} {ticker}"


def skew_relative_label(ticker: str) -> str:
    return f"{SKEW_LABEL_START} {ticker} (% of Total)"


def notional_skew_label(ticker: str) -> str:
    return f"{NOTIONAL_SKEW_LABEL_START} {ticker}"


def notional_skew_relative_label(ticker: str) -> str:
    return f"{NOTIONAL_SKEW_LABEL_START} {ticker} (% of Total)"


def notional_skew_relative_supply_label(ticker: str) -> str:
    return f"{NOTIONAL_SKEW_LABEL_START} {ticker} (% of Supply)"


def reserve_skew_label(ticker: str) -> str:
    return f"{RESERVE_SKEW_LABEL_START} {ticker}"


def reserve_skew_relative_label(ticker: str) -> str:
    return f"{RESERVE_SKEW_LABEL_START} {ticker} (% Difference)"


def avg_cost_label(ticker: str, long: bool) -> str:
    side = "Longs" if long else "Shorts"
    return f"{AVG_COST_LABEL_START} for {side} on {ticker}"


def unrealized_pnl_label(ticker: str, long: bool) -> str:
    side = "Longs" if long else "Shorts"
    return f"{UNREALIZED_PNL_LABEL_START} for {side} on {ticker}"


def open_positions_label(ticker: str) -> str:
    return f"{OPEN_POSITIONS_LABEL_START} {ticker}"


def funding_supply_change_label(ticker: str) -> str:
    return f"{FUNDING_LABEL_START} Supply Change on {FUTURES_PRICE_LABEL_START}{ticker}"


def funding_pay_long_label(ticker: str) -> str:
    return f"{FUNDING_LABEL_START} Payments Long on {FUTURES_PRICE_LABEL_START}{ticker}"


def funding_pay_short_label(ticker: str) -> str:
    return f"{FUNDING_LABEL_START} Payments Short on {FUTURES_PRICE_LABEL_START}{ticker}"


def funding_fees_label(ticker: str) -> str:
    return f"{FUNDING_LABEL_START} Fees on {FUTURES_PRICE_LABEL_START}{ticker}"


def inventory_wealth_ovl_label(agent_type_name: str) -> str:
    return f"{agent_type_name} Inventory (OVL)"


def inventory_wealth_usd_label(agent_type_name: str) -> str:
    return f"{agent_type_name} Inventory (USD)"


def inventory_wealth_quote_label(agent_type_name: str, quote_ticker: str) -> str:
    return f"{agent_type_name} Inventory ({quote_ticker})"


def agent_wealth_ovl_label(agent_type_name: str):
    return f'{agent_type_name} Wealth (OVL)'


GINI_LABEL = 'Gini'
GINI_ARBITRAGEURS_LABEL = "Gini (Arbitrageurs)"

SUPPLY_LABEL = "Supply"
TREASURY_LABEL = "Treasury"
LIQUIDITY_LABEL = "Liquidity"
