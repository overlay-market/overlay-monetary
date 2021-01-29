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

from ovm.paths import OUTPUT_DATA_DIRECTORY

REPORT_RESULT_TYPE = tp.Union[int, float]
DATA_COLLECTOR_NAME = 'data_collector'
STEP_COLUMN_NAME = 'step'
MODEL_VARIABLES_GROUP_NAME = 'MODEL_VARIABLES'
AGENT_VARIABLES_GROUP_NAME = 'AGENT_VARIABLES'

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


class ModelReporterCollection(tp.Generic[ModelType]):
    def __init__(
            self,
            save_interval: int,
            hdf5_file: h5py.File,
            name_to_model_reporter_map:
            tp.Optional[tp.Dict[str, AbstractModelReporter[ModelType]]] = None,
    ):
        if not name_to_model_reporter_map:
            name_to_model_reporter_map = {}

        assert not any(name == STEP_COLUMN_NAME for name in name_to_model_reporter_map.keys())

        self.name_to_reporter_map = name_to_model_reporter_map
        self.reporters: tp.List[AbstractModelReporter[ModelType]] = []
        self.reporter_names: tp.List[str] = []
        self.buffers: tp.List[np.ndarray] = []
        self.datasets = []
        self.save_interval = save_interval
        self.current_buffer_index = 0

        for name, reporter in self.name_to_reporter_map.items():
            self.reporter_names.append(name)
            self.reporters.append(reporter)
            self.buffers.append(np.zeros((save_interval,), dtype=reporter.dtype))

        self.model_group = hdf5_file.get(MODEL_VARIABLES_GROUP_NAME)
        if not self.model_group:
            # create a model group
            self.model_group = hdf5_file.create_group(MODEL_VARIABLES_GROUP_NAME)

            # create a hdf5 dataset for each reporter
            for name, reporter in zip(self.reporter_names, self.reporters):

                self.datasets.append(self.model_group.create_dataset(name=name,
                                                                     shape=(0,),
                                                                     dtype=reporter.dtype,
                                                                     maxshape=(None,),
                                                                      chunks=(save_interval,)))
        else:
            for name in self.reporter_names:
                dataset = self.model_group.get(name)
                assert dataset is not None
                self.datasets.append(dataset)
                # assert dataset.dtype == reporter.dtype

    def _purge_buffer(self):
        if self.current_buffer_index == 0:
            return

        end_index = self.steps + 1
        begin_index = self.steps - self.current_buffer_index + 1

        # write buffers for model reporters to hdf5 file
        for dataset, buffer in zip(self.datasets, self.buffers):
            dataset.resize(size=(end_index,))
            dataset[begin_index:end_index] = buffer[:self.current_buffer_index]

        self.current_buffer_index = 0

    def collect(self, model):
        self.steps = model.schedule.steps

        # collect model reports and write to buffer
        for name, reporter, buffer in zip(self.reporter_names,
                                          self.reporters,
                                          self.buffers):
            buffer[self.current_buffer_index] = reporter.report(model)

        # increment index in buffer to write next model results to
        self.current_buffer_index += 1

        # check if buffer is full and must be written to hdf5 file
        if self.current_buffer_index == self.save_interval:
            self._purge_buffer()

    def get_dataframe(self,
                      step_dataset: np.ndarray,
                      first_step: tp.Optional[int] = 0,
                      last_step: tp.Optional[int] = -1,
                      variable_selection: tp.Optional[tp.Sequence[str]] = None) \
            -> pd.DataFrame:
        self._purge_buffer()

        if not variable_selection:
            variable_selection = self.reporter_names

        name_to_dataset_map = \
            {name: np.array(dataset[first_step:last_step])
             for name, dataset
             in zip(self.reporter_names, self.datasets)
             if name in variable_selection}

        name_to_dataset_map.update({STEP_COLUMN_NAME: step_dataset})

        return pd.DataFrame(name_to_dataset_map)


class HDF5DataCollector(tp.Generic[ModelType, AgentType]):
    def __init__(
            self,
            model_name: str,
            agents: tp.Sequence[AgentType],
            save_interval: int,
            existing_filename_to_append_to: tp.Optional[str] = None,
            model_reporters: tp.Optional[tp.Dict[str, AbstractModelReporter[ModelType]]] = None,
            agent_reporters: tp.Optional[tp.Dict[str, AbstractAgentReporter[ModelType]]] = None,
            output_data_directory: tp.Optional[str] = None,
            # tables=None
    ):
        if not agent_reporters:
            agent_reporters = {}

        if not output_data_directory:
            output_data_directory = OUTPUT_DATA_DIRECTORY

        if not os.path.exists(output_data_directory):
            os.makedirs(output_data_directory)

        assert not any(name == STEP_COLUMN_NAME for name in agent_reporters.keys())

        self.save_interval = save_interval
        self.current_buffer_index = 0
        self.model_name = model_name

        # step
        self.step_buffer = np.zeros((save_interval, ), dtype=np.int32)
        self.step_dataset = None

        if existing_filename_to_append_to:
            self.hdf5_filename = existing_filename_to_append_to
            self.hdf5_file = h5py.File(self.hdf5_filename, 'a')

            assert self.hdf5_file[STEP_COLUMN_NAME] is not None
        else:
            git_commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha

            dt_string = \
                datetime.now().strftime("%d-%m-%Y-%H-%M-%S(day-month-year-hour-minute-second)")
            self.hdf5_filename = model_name + "_" + git_commit_hash + "_" + dt_string
            self.hdf5_path = os.path.join(output_data_directory, self.hdf5_filename)

            self.hdf5_file: h5py.File = h5py.File(self.hdf5_path, 'w')

            self.step_dataset = \
                self.hdf5_file.create_dataset(name=STEP_COLUMN_NAME,
                                              shape=(0,),
                                              dtype=np.int32,
                                              maxshape=(None,),
                                              chunks=(save_interval,))

        # model reporter collection
        self._model_reporter_collection = \
            ModelReporterCollection(save_interval=save_interval,
                                    name_to_model_reporter_map=model_reporters,
                                    hdf5_file=self.hdf5_file)

    def _purge_buffer(self):
        if self.current_buffer_index == 0:
            return

        step = self.steps
        end_index = step + 1
        begin_index = step - self.current_buffer_index + 1

        # write time steps to HDF5 file
        self.step_dataset.resize(size=(end_index,))
        self.step_dataset[begin_index:end_index] = self.step_buffer[:self.current_buffer_index]

        # write model reports to HDF5 file
        self._model_reporter_collection._purge_buffer()

        self.current_buffer_index = 0

    def collect(self, model):
        self.steps = model.schedule.steps
        self.step_buffer[self.current_buffer_index] = self.steps

        # collect model reports and write to buffer
        self._model_reporter_collection.collect(model)

        # increment index in buffer to write next model results to
        self.current_buffer_index += 1

        # check if buffer is full and must be written to hdf5 file
        if self.current_buffer_index == self.save_interval:
            self._purge_buffer()

    def get_model_vars_dataframe(self,
                                 first_step: tp.Optional[int] = 0,
                                 last_step: tp.Optional[int] = -1,
                                 model_variable_selection: tp.Optional[tp.Sequence[str]] = None) \
            -> pd.DataFrame:
        self.flush()

        return (self
                ._model_reporter_collection
                .get_dataframe(
                    step_dataset=self.step_dataset[first_step:last_step],
                    first_step=first_step,
                    last_step=last_step,
                    variable_selection=model_variable_selection))

    def flush(self):
        # write what remains in the buffer to the HDF5 file object
        self._purge_buffer()
        self._model_reporter_collection._purge_buffer()

        # write all changes to the HDF5 object to disk
        self.hdf5_file.flush()

    def __del__(self):
        self.hdf5_file.close()
