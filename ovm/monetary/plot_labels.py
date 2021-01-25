DEVIATION_LABEL_START = 'd-'
SPOT_PRICE_LABEL_START = 's-'
FUTURES_PRICE_LABEL_START = 'f-'
SKEW_LABEL_START = 'Skew'


def price_deviation_label(ticker: str) -> str:
    return f"{DEVIATION_LABEL_START}{ticker}"


def spot_price_label(ticker: str) -> str:
    return f"{SPOT_PRICE_LABEL_START}{ticker}"


def futures_price_label(ticker: str) -> str:
    return f"{FUTURES_PRICE_LABEL_START}{ticker}"


def skew_label(ticker: str) -> str:
    return f"{SKEW_LABEL_START} {ticker}"


def position_count_label(ticker: str) -> str:
    return f"{futures_price_label(ticker)} Position Count"


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
