def price_deviation_label(ticker: str) -> str:
    return f"d-{ticker}"


def spot_price_label(ticker: str) -> str:
    return f"s-{ticker}"


def futures_price_label(ticker: str) -> str:
    return f"f-{ticker}"


def skew_label(ticker: str) -> str:
    return f"Skew {ticker}"


def inventory_wealth_ovl_label(agent_type_name: str) -> str:
    return f"{agent_type_name} Inventory (OVL)"


def inventory_wealth_usd_label(agent_type_name: str) -> str:
    return f"{agent_type_name} Inventory (USD)"


GINI_LABEL = 'Gini'
GINI_ARBITRAGEURS_LABEL = "Gini (Arbitrageurs)"

SUPPLY_LABEL = "Supply"
TREASURY_LABEL = "Treasury"
LIQUIDITY_LABEL="Liquidity"
