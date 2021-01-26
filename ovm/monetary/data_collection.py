from dataclasses import dataclass

DATA_COLLECTOR_NAME = 'data_collector'


@dataclass
class DataCollectionOptions:
    # if perform_data_collection is set to False, all data collection is turned off,
    # even if individual flags are turned on below
    perform_data_collection: bool = True
    compute_gini_coefficient: bool = True
    compute_wealth: bool = True
    compute_inventory_wealth: bool = True
    data_collection_interval: int = 1