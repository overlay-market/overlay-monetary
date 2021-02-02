import typing as tp

from mesa.visualization.modules import ChartModule

from ovm.monetary.data_collection import DATA_COLLECTOR_NAME, DataCollectionOptions
from ovm.monetary.plot_labels import SUPPLY_LABEL, TREASURY_LABEL, price_deviation_label, \
    skew_label, reserve_skew_relative_label, open_positions_label, agent_wealth_ovl_label, LIQUIDITY_LABEL, GINI_LABEL, \
    GINI_ARBITRAGEURS_LABEL, spot_price_label, futures_price_label
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

        ChartModule([{"Label": skew_label(ticker), "Color": random_color()}
                     for ticker
                     in tickers],
                    data_collector_name=DATA_COLLECTOR_NAME),
        ChartModule([{"Label": reserve_skew_relative_label(ticker), "Color": random_color()}
                     for ticker
                     in tickers],
                    data_collector_name=DATA_COLLECTOR_NAME),
        ChartModule([{"Label": open_positions_label(ticker), "Color": random_color()}
                     for ticker
                     in tickers],
                    data_collector_name=DATA_COLLECTOR_NAME),
    ]

    if data_collection_options.compute_inventory_wealth:
        for agent_type_name in ["Arbitrageurs", "Traders", "Holders", "Liquidators", "Snipers", "Apes"]:
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
            ChartModule([{"Label": GINI_ARBITRAGEURS_LABEL, "Color": "Blue"}],
                        data_collector_name=DATA_COLLECTOR_NAME),
    ]

    for ticker in tickers:
        chart_elements.append(
            ChartModule([
                {"Label": spot_price_label(ticker), "Color": "Black"},
                {"Label": futures_price_label(ticker), "Color": "Red"},
            ], data_collector_name='data_collector')
        )

    return chart_elements
