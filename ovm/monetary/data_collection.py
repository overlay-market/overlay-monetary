from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import os
import typing as tp

import git
import h5py
from mesa.agent import Agent
from mesa.model import Model
import numpy as np
import pandas as pd

from ovm.paths import DATA_OUTPUT_BASE_PATH

REPORT_RESULT_TYPE = tp.Union[int, float]
DATA_COLLECTOR_NAME = 'data_collector'
STEP_COLUMN_NAME = 'step'

AgentType = tp.TypeVar('AgentType', bound=Agent)
ModelType = tp.TypeVar('ModelType', bound=Model)


@dataclass
class DataCollectionOptions:
    # if perform_data_collection is set to False, all data collection is turned off,
    # even if individual flags are turned on below
    perform_data_collection: bool = True
    compute_gini_coefficient: bool = True
    compute_wealth: bool = True
    compute_inventory_wealth: bool = True
    data_collection_interval: int = 1
    use_hdf5: bool = False


################################################################################
# Model Level Reporter Base Classes
################################################################################
class AbstractModelReporter(tp.Generic[ModelType], ABC):
    @property
    def dtype(self) -> np.number:
        return np.float64

    @abstractmethod
    def report(self, model: ModelType) -> REPORT_RESULT_TYPE:
        pass

    def __call__(self, model: ModelType) -> REPORT_RESULT_TYPE:
        return self.report(model)


class AbstractMarketLevelReporter(tp.Generic[ModelType], AbstractModelReporter[ModelType], ABC):
    def __init__(self, ticker: str):
        self.ticker = ticker


class AbstractAgentTypeLevelReporter(tp.Generic[ModelType, AgentType], AbstractModelReporter[ModelType], ABC):
    def __init__(self,
                 agent_type: tp.Optional[tp.Type[AgentType]] = None):
        self.agent_type = agent_type


################################################################################
# Agent Level Reporter Base Classes
################################################################################
class AbstractAgentReporter(tp.Generic[AgentType], ABC):
    @property
    def dtype(self) -> np.generic:
        return np.float64

    @abstractmethod
    def report(self, agent: AgentType) -> REPORT_RESULT_TYPE:
        pass

    def __call__(self, agent: AgentType) -> REPORT_RESULT_TYPE:
        return self.report(agent)


class HDF5DataCollector:
    def __init__(self,
                 model_name: str,
                 save_interval: int,
                 existing_filename_to_append_to: tp.Optional[str] = None,
                 model_reporters: tp.Dict[str, AbstractModelReporter] = None,
                 agent_reporters: tp.Dict[str, AbstractModelReporter] = None,
                 output_base_path: tp.Optional[str] = None,
                 # tables=None
                 ):
        print(f'save_interval={save_interval}')
        if not model_reporters:
            model_reporters = {}

        if not agent_reporters:
            agent_reporters = {}

        if not output_base_path:
            output_base_path = DATA_OUTPUT_BASE_PATH

        if not os.path.exists(output_base_path):
            os.makedirs(output_base_path)

        assert not any(name == STEP_COLUMN_NAME for name in model_reporters.keys())
        assert not any(name == STEP_COLUMN_NAME for name in agent_reporters.keys())

        self.save_interval = save_interval
        self.current_buffer_index = 0
        self.model_name = model_name
        self.model_reporters = model_reporters
        # self.agent_reporters = agent_reporters

        # step
        self.step_buffer = np.zeros((save_interval, ), dtype=np.int32)
        self.step_dataset = None

        # model reporters and datasets
        self.model_reporters_buffers = \
            {name: np.zeros((save_interval, ), dtype=reporter.dtype)
             for name, reporter in model_reporters.items()}
        self.name_to_model_dataset_map: tp.Dict[str, h5py.Dataset] = {}

        if existing_filename_to_append_to:
            self.hdf5_filename = existing_filename_to_append_to
            self.hdf5_file = h5py.File(self.hdf5_filename, 'a')

            # verify that an hdf5 dataset exists for each reporter
            for name, reporter in model_reporters.items():
                dataset = self.hdf5_file.get(name)
                assert dataset is not None
                self.name_to_model_dataset_map[name] = dataset
                # assert dataset.dtype == reporter.dtype

            assert self.hdf5_file[STEP_COLUMN_NAME] is not None
        else:
            git_commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha

            dt_string = \
                datetime.now().strftime("%d-%m-%Y-%H-%M-%S(day-month-year-hour-minute-second)")
            self.hdf5_filename = model_name + "_" + git_commit_hash + "_" + dt_string
            self.hdf5_file: h5py.File = h5py.File(self.hdf5_filename, 'w')

            # create a hdf5 dataset for each reporter
            for name, reporter in model_reporters.items():
                self.name_to_model_dataset_map[name] = \
                    self.hdf5_file.create_dataset(name=name,
                                                  shape=(0,),
                                                  dtype=reporter.dtype,
                                                  maxshape=(None,),
                                                  chunks=(save_interval,))

            self.step_dataset = self.hdf5_file.create_dataset(name=STEP_COLUMN_NAME,
                                                              shape=(0,),
                                                              dtype=np.int32,
                                                              maxshape=(None,),
                                                              chunks=(save_interval,))

            self.name_to_model_dataset_map[STEP_COLUMN_NAME] = self.step_dataset

    def _purge_buffer(self):
        if self.current_buffer_index == 0:
            return

        step = self.steps
        end_index = step + 1
        begin_index = step - self.current_buffer_index + 1
        # assert begin_index == step - self.save_interval + 1

        # write buffers for model reporters to hdf5 file
        for i, (name, buffer) in enumerate(self.model_reporters_buffers.items()):
            data = self.name_to_model_dataset_map[name]
            data.resize(size=(end_index,))
            data[begin_index:end_index] = buffer[:self.current_buffer_index]

        # write time steps to HDF5 file
        self.step_dataset.resize(size=(end_index,))
        self.step_dataset[begin_index:end_index] = self.step_buffer[:self.current_buffer_index]

        self.current_buffer_index = 0

    def collect(self, model):
        self.steps = model.schedule.steps
        self.step_buffer[self.current_buffer_index] = self.steps

        # collect model reports and write to buffer
        for name, reporter in self.model_reporters.items():
            report = reporter.report(model)
            self.model_reporters_buffers[name][self.current_buffer_index] = report

        # increment index in buffer to write next model results to
        self.current_buffer_index += 1

        # check if buffer is full and must be written to hdf5 file
        if self.current_buffer_index == self.save_interval:
            self._purge_buffer()


        # if self.model_reporters:
        #     for var, reporter in self.model_reporters.items():
        #         self.model_vars[var].append(reporter(model))
        #
        # if self.agent_reporters:
        #     agent_records = self._record_agents(model)
        #     self._agent_records[model.schedule.steps] = list(agent_records)

    def get_model_vars_dataframe(self,
                                 first_step: tp.Optional[int] = 0,
                                 last_step: tp.Optional[int] = -1) -> pd.DataFrame:
        self.flush()
        # name_to_dataset_map = \
        #     {STEP_COLUMN_NAME: np.array(self.step_dataset[first_step:last_step])}
        #
        # for name, dataset in self.name_to_model_dataset_map.items():
        #     # print(f'{name} {array.shape}')
        #     name_to_dataset_map[name] = np.array(dataset)

        name_to_dataset_map = \
            {name: np.array(dataset[first_step:last_step])
             for name, dataset
             in self.name_to_model_dataset_map.items()}

        name_to_dataset_map.update({STEP_COLUMN_NAME: np.array(self.step_dataset[first_step:last_step])})

        return pd.DataFrame(name_to_dataset_map)

    def flush(self):
        # write what remains in the buffer to the HDF5 file object
        self._purge_buffer()

        # write all changes to the HDF5 object to disk
        self.hdf5_file.flush()

    def __del__(self):
        self.hdf5_file.close()
