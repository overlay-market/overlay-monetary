from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import glob
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
AGENT_ID_TO_TYPE_GROUP_NAME = 'AGENT_ID_TO_TYPE'
GIT_COMMIT_HASH_NAME = 'GIT_COMMIT_HASH'
GIT_BRANCH_NAME = 'GIT_BRANCH'

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


################################################################################
# Model Level Reporter Collection
################################################################################
class ModelReporterCollection(tp.Generic[ModelType]):
    def __init__(
            self,
            save_interval: int,
            hdf5_file: h5py.File,
            name_to_model_reporter_map:
            tp.Optional[tp.Dict[str, AbstractModelReporter[ModelType]]] = None
    ):
        if not name_to_model_reporter_map:
            name_to_model_reporter_map = {}

        assert not any(name == STEP_COLUMN_NAME for name in name_to_model_reporter_map.keys())

        self._reporters: tp.List[AbstractModelReporter[ModelType]] = []
        self._reporter_names: tp.List[str] = []
        self._buffers: tp.List[np.ndarray] = []
        self._datasets = []
        self._save_interval = save_interval
        self._current_buffer_index = 0

        for name, reporter in name_to_model_reporter_map.items():
            self._reporter_names.append(name)
            self._reporters.append(reporter)
            self._buffers.append(np.zeros((save_interval,), dtype=reporter.dtype))

        self._reporters = tuple(self._reporters)
        self._reporter_names = tuple(self._reporter_names)
        self._buffers = tuple(self._buffers)
        self._hdf5_file = hdf5_file

        model_group = hdf5_file.get(MODEL_VARIABLES_GROUP_NAME)
        if not model_group:
            # create a model group
            model_group = hdf5_file.create_group(MODEL_VARIABLES_GROUP_NAME, track_order=True)

            # create a hdf5 dataset for each reporter
            for name, reporter in zip(self._reporter_names, self._reporters):
                self._datasets.append(model_group.create_dataset(name=name,
                                                                 shape=(0,),
                                                                 dtype=reporter.dtype,
                                                                 maxshape=(None,),
                                                                 chunks=(save_interval,)))
        else:
            for name in self._reporter_names:
                dataset = model_group.get(name)
                assert dataset is not None
                self._datasets.append(dataset)

        self._datasets = tuple(self._datasets)

    @property
    def reporter_names(self) -> tp.Sequence[str]:
        return self._reporter_names

    def _purge_buffer(self):
        if self._current_buffer_index == 0:
            return

        end_index = self.steps + 1
        begin_index = self.steps - self._current_buffer_index + 1

        # write buffers for model reporters to hdf5 file
        for dataset, buffer in zip(self._datasets, self._buffers):
            dataset.resize(size=(end_index,))
            dataset[begin_index:end_index] = buffer[:self._current_buffer_index]

        self._current_buffer_index = 0

    def collect(self, model):
        self.steps = model.schedule.steps

        # collect model reports and write to buffer
        for name, reporter, buffer in zip(self._reporter_names,
                                          self._reporters,
                                          self._buffers):
            buffer[self._current_buffer_index] = reporter.report(model)

        # increment index in buffer to write next model results to
        self._current_buffer_index += 1

        # check if buffer is full and must be written to hdf5 file
        if self._current_buffer_index == self._save_interval:
            self._purge_buffer()

    def get_dataframe(self,
                      unsliced_step_dataset: np.ndarray,
                      first_step: tp.Optional[int] = 0,
                      last_step: tp.Optional[int] = -1,
                      stride: int = 1,
                      variable_selection: tp.Optional[tp.Sequence[str]] = None) \
            -> pd.DataFrame:
        self._purge_buffer()

        if not variable_selection:
            variable_selection = self._reporter_names

        model_group = self._hdf5_file.get(MODEL_VARIABLES_GROUP_NAME)

        return _get_model_df_from_hdf5_file(model_group=model_group,
                                            unsliced_step_dataset=unsliced_step_dataset,
                                            first_step=first_step,
                                            last_step=last_step,
                                            stride=stride,
                                            variable_selection=variable_selection)


def _get_model_df_from_hdf5_file(model_group: h5py.Group,
                                 unsliced_step_dataset: np.ndarray,
                                 variable_selection: tp.Sequence[str],
                                 first_step: tp.Optional[int] = 0,
                                 last_step: tp.Optional[int] = -1,
                                 stride: int = 1) -> pd.DataFrame:
    name_to_dataset_map = \
        {name: np.array(model_group[name][first_step:last_step:stride])
         for name
         in variable_selection}

    name_to_dataset_map.update({STEP_COLUMN_NAME:
                                    unsliced_step_dataset[first_step:last_step:stride]})

    return pd.DataFrame(name_to_dataset_map)


################################################################################
# Agent Level Reporter Collection
################################################################################
def _get_unqualified_class_name_from_object(obj) -> str:
    return str(type(obj)).split("'")[1].split(".")[-1]


def filter_agents_by_type(agents: tp.Iterable[AgentType],
                          agent_type: tp.Optional[tp.Type[AgentType]] = None) \
        -> tp.Iterable[AgentType]:
    if agent_type:
        agents = filter(lambda a: type(a) == agent_type, agents)

    return agents


class AgentReporterCollection(tp.Generic[ModelType, AgentType]):
    def __init__(
            self,
            agents: tp.Sequence[AgentType],
            save_interval: int,
            hdf5_file: h5py.File,
            name_to_agent_reporter_map:
            tp.Optional[tp.Dict[str, AbstractAgentReporter[AgentType]]] = None,
    ):
        if not name_to_agent_reporter_map:
            name_to_agent_reporter_map = {}

        assert not any(name == STEP_COLUMN_NAME for name in name_to_agent_reporter_map.keys())

        self._agents = tuple(agents)
        self._reporters: tp.List[AbstractAgentReporter[AgentType]] = []
        self._reporter_names: tp.List[str] = []
        self._buffers: tp.List[np.ndarray] = []
        self._datasets = []
        self._save_interval = save_interval
        self._current_buffer_index = 0

        for name, reporter in name_to_agent_reporter_map.items():
            self._reporter_names.append(name)
            self._reporters.append(reporter)
            self._buffers.append(np.zeros((save_interval, self.number_of_agents),
                                          dtype=reporter.dtype))

        self._reporters = tuple(self._reporters)
        self._reporter_names = tuple(self._reporter_names)
        self._buffers = tuple(self._buffers)

        agent_group = hdf5_file.get(AGENT_VARIABLES_GROUP_NAME)
        if not agent_group:
            # create a model group
            agent_group = hdf5_file.create_group(AGENT_VARIABLES_GROUP_NAME, track_order=True)

            # create a hdf5 dataset for each reporter
            for name, reporter in zip(self._reporter_names, self._reporters):
                agent_dataset = \
                    agent_group.create_dataset(name=name,
                                               shape=(0, self.number_of_agents),
                                               dtype=reporter.dtype,
                                               maxshape=(None, self.number_of_agents),
                                               chunks=(save_interval, self.number_of_agents))

                self._datasets.append(agent_dataset)

            # create agent type group
            agent_type_group = hdf5_file.create_group(AGENT_ID_TO_TYPE_GROUP_NAME, track_order=True)
            for agent in self._agents:
                agent_type_group.attrs[str(agent.unique_id)] = \
                    _get_unqualified_class_name_from_object(agent)
        else:
            for name in self._reporter_names:
                dataset = agent_group.get(name)
                assert dataset is not None
                self._datasets.append(dataset)

        self._datasets = tuple(self._datasets)

    @property
    def number_of_agents(self) -> int:
        return len(self._agents)

    @property
    def agents(self) -> tp.Sequence[AgentType]:
        return self._agents

    @property
    def reporter_names(self) -> tp.Sequence[str]:
        return self._reporter_names

    def _get_agent_type_indicator(self, agent_type: tp.Optional[tp.Type[AgentType]] = None) \
            -> np.ndarray:
        return np.array([type(a) == agent_type for a in self._agents])

    @property
    def agent_type_set(self) -> tp.Set[tp.Type[AgentType]]:
        return set(type(agent) for agent in self._agents)

    @property
    def agent_type_strings(self) -> tp.Sequence[str]:
        return [_get_unqualified_class_name_from_object(agent) for agent in self._agents]

    def _agent_ids(self, agent_type: tp.Optional[tp.Type[AgentType]] = None) -> tp.Sequence[int]:
        return [agent.unique_id
                for agent
                in filter_agents_by_type(self._agents, agent_type=agent_type)]

    def _agent_type_and_id_combined(self, agent_type: tp.Optional[tp.Type[AgentType]] = None) \
            -> tp.Sequence[str]:
        return [f"{_get_unqualified_class_name_from_object(agent)}-{agent.unique_id}"
                for agent
                in filter_agents_by_type(self._agents, agent_type=agent_type)]

    def _purge_buffer(self):
        if self._current_buffer_index == 0:
            return

        end_index = self.steps + 1
        begin_index = self.steps - self._current_buffer_index + 1

        # write buffers for model reporters to hdf5 file
        for dataset, buffer in zip(self._datasets, self._buffers):
            dataset.resize(size=(end_index, self.number_of_agents))
            dataset[begin_index:end_index] = buffer[:self._current_buffer_index, :]

        self._current_buffer_index = 0

    def collect(self, model: ModelType):
        self.steps = model.schedule.steps

        # collect model reports and write to buffer
        for name, reporter, buffer in zip(self._reporter_names,
                                          self._reporters,
                                          self._buffers):
            for i, agent in enumerate(model.schedule.agents):
                # print(f'{i}: {agent}')
                buffer[self._current_buffer_index, i] = reporter.report(agent)

        # increment index in buffer to write next model results to
        self._current_buffer_index += 1

        # check if buffer is full and must be written to hdf5 file
        if self._current_buffer_index == self._save_interval:
            self._purge_buffer()

    def get_dataframe_for_reporter(
            self,
            reporter_name: str,
            unsliced_step_dataset: np.ndarray,
            first_step: tp.Optional[int] = 0,
            last_step: tp.Optional[int] = -1,
            stride: int = 1,
            use_agent_types_in_header: bool = False,
            agent_type: tp.Optional[tp.Type[AgentType]] = None) -> pd.DataFrame:
        self._purge_buffer()

        reporter_index = self._reporter_names.index(reporter_name)

        dataset = self._datasets[reporter_index]
        if agent_type:
            agent_type_indicator = self._get_agent_type_indicator(agent_type)
            array = np.array(dataset[first_step:last_step:stride, agent_type_indicator])
        else:
            array = np.array(dataset[first_step:last_step:stride, :])

        if use_agent_types_in_header:
            agent_header = self._agent_type_and_id_combined(agent_type)
        else:
            agent_header = self._agent_ids(agent_type)

        return pd.DataFrame(data=array,
                            index=unsliced_step_dataset[first_step:last_step:stride],
                            columns=agent_header)


class HDF5DataCollector(tp.Generic[ModelType, AgentType]):
    def __init__(
            self,
            model: ModelType,
            save_interval: int,
            existing_filename_to_append_to: tp.Optional[str] = None,
            model_reporters: tp.Optional[tp.Dict[str, AbstractModelReporter[ModelType]]] = None,
            agent_reporters: tp.Optional[tp.Dict[str, AbstractAgentReporter[ModelType]]] = None,
            output_data_directory: tp.Optional[str] = None,
            model_parameter_name_to_parameter_value_map: tp.Optional[tp.Dict[str, tp.Any]] = None,  # metadata to attach to model group
    ):
        if not model_parameter_name_to_parameter_value_map:
            model_parameter_name_to_parameter_value_map = {}

        if not output_data_directory:
            output_data_directory = OUTPUT_DATA_DIRECTORY

        if not os.path.exists(output_data_directory):
            os.makedirs(output_data_directory)

        self._save_interval = save_interval
        self._current_buffer_index = 0

        # step
        self._step_buffer = np.zeros((save_interval,), dtype=np.int32)
        self._step_dataset = None

        if existing_filename_to_append_to:
            self.hdf5_filename = existing_filename_to_append_to
            self.hdf5_file = h5py.File(self.hdf5_filename, 'a')
            self._step_dataset = self.hdf5_file[STEP_COLUMN_NAME]

            assert self._step_dataset is not None
        else:
            git_commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha
            git_branch_name = git.Repo(search_parent_directories=True).active_branch.name

            # dt_string = \
            #     datetime.now().strftime("%d-%m-%Y-%H-%M-%S(day-month-year-hour-minute-second)")
            # self.hdf5_filename = model.name + "_" + git_commit_hash + "_" + dt_string + '.h5'
            dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            self.hdf5_filename = model.name + "_" + dt_string + '.h5'
            self.hdf5_path = os.path.join(output_data_directory, self.hdf5_filename)

            self.hdf5_file: h5py.File = h5py.File(self.hdf5_path, 'w')
            self.hdf5_file.attrs[GIT_COMMIT_HASH_NAME] = git_commit_hash
            self.hdf5_file.attrs[GIT_BRANCH_NAME] = git_branch_name

            for parameter_name, parameter_value in model_parameter_name_to_parameter_value_map.items():
                print(f"{parameter_name}={parameter_value}")
                if not parameter_value:
                    parameter_value = 'None'
                self.hdf5_file.attrs[parameter_name] = parameter_value

            self._step_dataset = \
                self.hdf5_file.create_dataset(name=STEP_COLUMN_NAME,
                                              shape=(0,),
                                              dtype=np.int32,
                                              maxshape=(None,),
                                              chunks=(save_interval,))

        # model reporter collection
        self._model_reporter_collection = \
            ModelReporterCollection(
                save_interval=save_interval,
                name_to_model_reporter_map=model_reporters,
                hdf5_file=self.hdf5_file)

        # agent reporter collection
        self._agent_reporter_collection = \
            AgentReporterCollection(agents=model.schedule.agents,
                                    save_interval=save_interval,
                                    name_to_agent_reporter_map=agent_reporters,
                                    hdf5_file=self.hdf5_file)

    @property
    def model_reporter_names(self) -> tp.Sequence[str]:
        return self._model_reporter_collection.reporter_names

    @property
    def agent_reporter_names(self) -> tp.Sequence[str]:
        return self._agent_reporter_collection.reporter_names

    def _purge_buffer(self):
        if self._current_buffer_index == 0:
            return

        step = self.steps
        end_index = step + 1
        begin_index = step - self._current_buffer_index + 1

        # write time steps to HDF5 file
        self._step_dataset.resize(size=(end_index,))
        self._step_dataset[begin_index:end_index] = self._step_buffer[:self._current_buffer_index]

        # write model reports to HDF5 file
        self._model_reporter_collection._purge_buffer()

        # write agent reports to HDF5 file
        self._agent_reporter_collection._purge_buffer()

        self._current_buffer_index = 0

    def collect(self, model):
        self.steps = model.schedule.steps
        self._step_buffer[self._current_buffer_index] = self.steps

        # collect model reports and write to buffer
        self._model_reporter_collection.collect(model)

        # collect agent reports and write to buffer
        self._agent_reporter_collection.collect(model)

        # increment index in buffer to write next model results to
        self._current_buffer_index += 1

        # check if buffer is full and must be written to hdf5 file
        if self._current_buffer_index == self._save_interval:
            self._purge_buffer()

    @property
    def model_reporter_names(self) -> tp.Sequence[str]:
        return self._model_reporter_collection.reporter_names

    def get_model_vars_dataframe(self,
                                 first_step: tp.Optional[int] = 0,
                                 last_step: tp.Optional[int] = -1,
                                 stride: int = 1,
                                 model_variable_selection: tp.Optional[tp.Sequence[str]] = None) \
            -> pd.DataFrame:
        self.flush()

        return (self
                ._model_reporter_collection
                .get_dataframe(
                    unsliced_step_dataset=self._step_dataset,
                    first_step=first_step,
                    last_step=last_step,
                    stride=stride,
                    variable_selection=model_variable_selection))

    @property
    def agent_type_set(self) -> tp.Set[tp.Type[AgentType]]:
        return self._agent_reporter_collection.agent_type_set

    @property
    def agent_reporter_names(self) -> tp.Sequence[str]:
        return self._agent_reporter_collection.reporter_names

    def get_agent_report_dataframe(self,
                                   reporter_name: str,
                                   first_step: tp.Optional[int] = 0,
                                   last_step: tp.Optional[int] = -1,
                                   stride: int = 1,
                                   use_agent_types_in_header: bool = False,
                                   agent_type: tp.Optional[tp.Type[AgentType]] = None) \
            -> pd.DataFrame:
        self.flush()

        return (self
                ._agent_reporter_collection
                .get_dataframe_for_reporter(
                    reporter_name=reporter_name,
                    unsliced_step_dataset=self._step_dataset,
                    first_step=first_step,
                    last_step=last_step,
                    stride=stride,
                    use_agent_types_in_header=use_agent_types_in_header,
                    agent_type=agent_type))

    def flush(self):
        # write what remains in the buffer to the HDF5 file object
        self._purge_buffer()
        self._model_reporter_collection._purge_buffer()
        self._agent_reporter_collection._purge_buffer()

        # write all changes to the HDF5 object to disk
        self.hdf5_file.flush()

    def __del__(self):
        self.hdf5_file.close()


class HDF5DataCollectionFile:
    def __init__(self, filepath: str):
        self._filepath = filepath
        self._hdf5_file: h5py.File = h5py.File(self._filepath, 'r')
        self._model_group = self._hdf5_file.get(MODEL_VARIABLES_GROUP_NAME)
        self._agent_group = self._hdf5_file.get(AGENT_VARIABLES_GROUP_NAME)
        self._agent_type_group = self._hdf5_file.get(AGENT_ID_TO_TYPE_GROUP_NAME)

    @property
    def step_dataset(self) -> np.ndarray:
        return np.array(self._hdf5_file[STEP_COLUMN_NAME])

    @property
    def commit_hash(self) -> str:
        return self._hdf5_file.attrs[GIT_COMMIT_HASH_NAME]

    @property
    def git_branch_name(self) -> str:
        return self._hdf5_file.attrs[GIT_BRANCH_NAME]

    def get_model_dataframe(self,
                            first_step: tp.Optional[int] = 0,
                            last_step: tp.Optional[int] = -1,
                            stride: int = 1,
                            variable_selection: tp.Optional[tp.Sequence[str]] = None) \
            -> pd.DataFrame:

        if not variable_selection:
            variable_selection = tuple(self._model_group.keys())

        return _get_model_df_from_hdf5_file(model_group=self._model_group,
                                            unsliced_step_dataset=self.step_dataset,
                                            first_step=first_step,
                                            last_step=last_step,
                                            stride=stride,
                                            variable_selection=variable_selection)

    @property
    def agent_type_strings(self) -> tp.Sequence[str]:
        return tuple(self._agent_type_group.attrs.values())

    @property
    def agent_type_string_set(self) -> tp.Set[str]:
        return set(self.agent_type_strings)

    @property
    def agent_id_to_type_map(self) -> tp.Dict[str, str]:
        return dict(self._agent_type_group.attrs)

    def _agent_type_and_id_combined(self, agent_type_string: tp.Optional[str] = None) \
            -> tp.Tuple[str, ...]:
        return tuple(f"{type_string}-{id}"
                     for id, type_string
                     in self.agent_id_to_type_map.items()
                     if type_string == agent_type_string)

    def _agent_ids(self, agent_type_string: tp.Optional[str] = None) -> tp.Tuple[int, ...]:
        return tuple(int(id)
                     for id, type_string
                     in self.agent_id_to_type_map.items()
                     if type_string == agent_type_string)

    def _get_agent_type_indicator(self, agent_type_string: tp.Optional[str] = None) \
            -> np.ndarray:
        if not agent_type_string:
            return np.full(shape=(len(self.agent_type_strings),), fill_value=True, dtype=np.bool)
        else:
            return np.array([a == agent_type_string for a in self.agent_type_strings])

    def get_agent_report_dataframe(
            self,
            reporter_name: str,
            first_step: tp.Optional[int] = 0,
            last_step: tp.Optional[int] = -1,
            stride: int = 1,
            use_agent_types_in_header: bool = False,
            agent_type_string: tp.Optional[str] = None) -> pd.DataFrame:
        if agent_type_string:
            agent_type_indicator = self._get_agent_type_indicator(agent_type_string)
            dataset = self._agent_group[reporter_name]
            array = np.array(dataset[first_step:last_step:stride, agent_type_indicator])
        else:
            array = np.array(self._agent_group[reporter_name][first_step:last_step:stride, :])

        if use_agent_types_in_header:
            agent_header = self._agent_type_and_id_combined(agent_type_string)
        else:
            agent_header = self._agent_ids(agent_type_string)

        return pd.DataFrame(data=array,
                            index=self.step_dataset[first_step:last_step:stride],
                            columns=agent_header)

    def get_model_level_parameter(self, name: str):
        return self._hdf5_file.attrs[name]

    @property
    def model_level_parameters(self) -> tp.Dict[str, tp.Any]:
        return dict(self._hdf5_file.attrs)

    def describe(self):
        print(f'HDF5 Agent Based Simulation Result')
        print(f'HDF5 File Path = {self._filepath}')
        print(f'Number of simulation steps = {len(self._hdf5_file[STEP_COLUMN_NAME])}')

        for name, value in self._hdf5_file.attrs.items():
            print(f'{name} = {value}')

    def __del__(self):
        self._hdf5_file.close()

    @classmethod
    def available_hdf5_files(cls, hdf5_base_path: str = OUTPUT_DATA_DIRECTORY) -> tp.Tuple[str]:
        files = glob.glob(os.path.join(hdf5_base_path, '*.h5'))
        files.sort(key=os.path.getmtime)
        return tuple(files)
