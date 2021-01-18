from dataclasses import dataclass


@dataclass
class DataCollectionOptions:
    # if False, all data collection is turned off, even if individual flags are turned on below
    perform_data_collection: bool = True
    compute_gini_coefficient: bool = True
    compute_wealth: bool = True
    compute_inventory_wealth: bool = True
