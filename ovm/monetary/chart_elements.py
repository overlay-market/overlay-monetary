import typing as tp

from mesa.visualization.modules import ChartModule

from ovm.monetary.data_collection import DATA_COLLECTOR_NAME, DataCollectionOptions
from ovm.monetary.plot_labels import (
    SUPPLY_LABEL, TREASURY_LABEL, price_deviation_label,
    skew_label, reserve_skew_relative_label, open_positions_label,
    agent_wealth_ovl_label, LIQUIDITY_LABEL, GINI_LABEL,
    GINI_ARBITRAGEURS_LABEL, spot_price_label, futures_price_label,
    funding_fees_label, skew_relative_label, avg_cost_label,
    funding_pay_long_label, funding_pay_short_label, funding_supply_change_label,
    unrealized_pnl_label, notional_skew_label, notional_skew_relative_label,
    notional_skew_relative_supply_label,
)
from ovm.monetary.plots import random_color


def construct_chart_elements(tickers, data_collection_options: DataCollectionOptions) -> tp.List[ChartModule]:
    # TODO: Have separate lines for each bot along with the aggregate!

    chart_elements = [
        ChartModule([{"Label": SUPPLY_LABEL, "Color": "Black"}],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": TREASURY_LABEL, "Color": "Green"}],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": price_deviation_label(ticker), "Color": random_color()}
                     for ticker
                     in tickers],
                    data_collector_name=DATA_COLLECTOR_NAME),

        ChartModule([{"Label": notional_skew_label(ticker), "Color": random_color()}
                     for ticker
                     in tickers],
                    data_collector_name=DATA_COLLECTOR_NAME),
        ChartModule([{"Label": notional_skew_relative_label(ticker), "Color": random_color()}
                     for ticker
                     in tickers],
                    data_collector_name=DATA_COLLECTOR_NAME),
        ChartModule([{"Label": notional_skew_relative_supply_label(ticker), "Color": random_color()}
                     for ticker
                     in tickers],
                    data_collector_name=DATA_COLLECTOR_NAME),
    ]

    if data_collection_options.compute_inventory_wealth:
        for agent_type_name in ["Arbitrageurs", "Traders", "Holders", "Liquidators", "Keepers", "Snipers", "Apes", "Chimps"]:
            chart_elements += [
                ChartModule([{"Label": agent_wealth_ovl_label(agent_type_name), "Color": random_color()}],
                            data_collector_name=DATA_COLLECTOR_NAME),
                #ChartModule([{"Label": inventory_wealth_ovl_label(agent_type_name), "Color": random_color()}],
                #            data_collector_name=DATA_COLLECTOR_NAME),
                #ChartModule([{"Label": inventory_wealth_usd_label(agent_type_name), "Color": random_color()}],
                #            data_collector_name=DATA_COLLECTOR_NAME),
            ]

    chart_elements += [
        ChartModule([{"Label": LIQUIDITY_LABEL, "Color": "Blue"}],
                    data_collector_name=DATA_COLLECTOR_NAME),
    ]

    if data_collection_options.compute_gini_coefficient:
        chart_elements += [
            ChartModule([{"Label": GINI_LABEL, "Color": "Black"}],
                        data_collector_name=DATA_COLLECTOR_NAME),
    ]

    for ticker in tickers:
        chart_elements.append(
            ChartModule([
                {"Label": spot_price_label(ticker), "Color": "Black"},
                {"Label": futures_price_label(ticker), "Color": "Red"},
            ], data_collector_name='data_collector')
        )

        chart_elements.append(
            ChartModule([
                {"Label": funding_pay_long_label(ticker), "Color": "Black"},
                {"Label": funding_pay_short_label(ticker), "Color": "Red"},
                {"Label": funding_supply_change_label(ticker), "Color": "Blue"},
            ], data_collector_name='data_collector')
        )

        chart_elements.append(
            ChartModule([
                {"Label": funding_fees_label(ticker), "Color": "Purple"},
            ], data_collector_name='data_collector')
        )

    return chart_elements
