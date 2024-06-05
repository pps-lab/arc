"""This module defines the Configuration model.
It uses pydantic to parse the config.json file.

The module defines the following functionality:

- enum ProtocolChoices:
    Defines the name mapping between the Protocol type and the string representation in the config.json
- class TaskConfig:
    Defines the Configuration for a single Experiment Run
- class ArgumentLineConfig:
    Defines the model for the configuration received via the command line
- class JsonMpcConfing:
    Defines the model for the MPC-specific configuration received via the config.json file
- class JsonConfigModel:
    Defines the relevant model for the configuration received via the config.json file

- function parse_json_config(config_path):
    Parses the config.json file found under the given config_path and returns a JsonConfigModel object

- function build_task_config(json_cofig_obj, player_number, sleep_time, result_dir):#
    Builds the TaskConfig object that contains all configuration information. It builds this object from the JsonConfigModel stored in json_config_obj, the player_number, the sleep_time and the result_dir arguments.
"""
import pydantic
import enum
import typing

class ProtocolChoices(enum.Enum):
    """Defines the name mapping between the Protocol type and the string representation in the config.json"""
    EMULATE_X = "emulate_env"
    REPLICATED_RING_PARTY_X = "semi_honest_3"
    REP4_RING_PARTY_X = "rep4-ring-party"
    BRAIN_PARTY_X = "malicious_3_party"
    REPLICATED_BIN_PARTY_X = "semi_honest_bin_3"
    PS_REP_BIN_PARTY_X = "malicious_bin_3"
    SHAMIR_PARTY_X = "shamir_semi_honest_n"
    MALICIOUS_SHAMIR_PARTY_X = "shamir_malicious_n"
    ATLAS_PARTY_X = "atlas_n"
    MAL_ATLAS_PARTY_X = "mal_atlas_n"
    REP_FIELD_PARTY = "rep-field-party"
    MAL_REP_FIELD_PARTY = "mal-rep-field-party"
    MAL_REP_RING_PARTY = "mal-rep-ring-party"

    SY_REP_RING_PARTY = "sy-rep-ring-party"
    SY_REP_FIELD_PARTY = "sy-rep-field-party"
    PS_REP_FIELD_PARTY = "ps-rep-field-party"
    SPDZ2K_PARTY = "spdz2k-party"
    SEMI2K_PARTY = "semi2k-party"
    SEMI_PARTY = "semi-party"
    MASCOT_PARTY = "mascot-party"
    MASCOT_OFFLINE = "mascot-offline"
    LOWGEAR_PARTY = "lowgear-party"
    HIGHGEAR_PARTY = "lowgear-party"


class ArgumentLineConfig(pydantic.BaseModel):
    """Defines the model for the configuration received via the command line

    Attributes
    ----------
    - player_id : int
        The id of the player
    - sleep_time : float
        The number of seconds to sleep between the compilation step and running step (Not used)
    """
    player_id: int
    sleep_time: float

# class CommitInfConfig(pydantic.BaseModel,extra=pydantic.Extra.forbid):
#     size_x: int
#     size_y: int
#
#     executable_id: str = "inference"
#
# class CommitTrainConfig(pydantic.BaseModel,extra=pydantic.Extra.forbid):
#     size_model: int
#
#     executable_id: str = "train"
#
# class CommitConfig(pydantic.BaseModel,extra=pydantic.Extra.forbid):
#     # the type of the commit, either inference or training (?) is specified in the program_args
#     commit: bool

class JsoncMpcConfig(pydantic.BaseModel,extra=pydantic.Extra.forbid):
    """Defines the model for the MPC-specific configuration received via the config.json file

    Attributes
    ----------
    - player_count : int
        The number of players involved in the experiment
    - player_0_hostname : str
        The hostname of player 0 to which all MPC protocol virtual machines should connect to
    - abs_path_to_code_dir : str
        The absolute path to the experiment code directory where the code for the evaluation framework is stored
    - script_name : str
        The name of the script that will be executed as part of the experiment
    - script_args : list[str]
        The list of 0-based positional arguments under which the given script should be compiled with
    - protocol_setup : ProtocolChoices
        The kind of MPC protocol that should be used in the Experiment
    """
    player_count: int
    player_0_hostname: str
    abs_path_to_code_dir: str
    script_name: str
    script_args: typing.Dict[str, object]
    protocol_setup: ProtocolChoices
    stage: typing.Union[typing.Literal['compile', 'run'], typing.List[typing.Literal['compile', 'run']]] # TODO:
    custom_prime: typing.Optional[str] = None
    custom_prime_length: typing.Optional[str] = None

    compiler_args: list[str] = None
    program_args: typing.Dict[str, str] = None

    domain: typing.Optional[str] = None # convenience parameter

class JsonConsistencyConfig(pydantic.BaseModel,extra=pydantic.Extra.forbid):
    hosts_file: str
    type: typing.Literal['pc', 'cerebro', 'sha3', 'sha3s']
    pc: typing.Literal['kzg', 'ipa', 'ped']
    abs_path_to_code_dir: str
    pp_args: int
    prover_party: typing.Optional[int] = None
    eval_point: typing.Optional[str] = None # Optional point to eval at for debugging
    single_random_eval_point: bool = True # whether to use a single point for all inputs or not. If eval_point is set, it will be used
    gen_pp: bool = False # whether to generate the public parameters
    use_split: bool = False # whether to use share splitting in conversion


class JsonConfigModel(pydantic.BaseModel,extra=pydantic.Extra.ignore):
    """Defines the relevant model for the configuration received via the config.json file"""
    mpc: JsoncMpcConfig
    consistency_args: typing.Optional[JsonConsistencyConfig] = None
    commit_output: typing.Optional[bool] = None
    convert_ring_bits: int = 34
    convert_n_threads: int = 18 # should work?
    convert_chunk_size: int = 500000
    convert_debug: bool = False
    sleep_time: float = 5.0 # time to sleep between conversion and next steps in seconds
    remove_input_files: bool = True

def parse_json_config(config_path):
    """Parses the config.json file found under the given config_path and returns a JsonConfigModel object

    Parameters
    ----------
    - config_path : str
        The path to the config.json file
    """
    config_obj = JsonConfigModel.parse_file(config_path)
    return config_obj

def build_task_config(json_config_obj: JsonConfigModel, player_number: int,
                    result_dir: str):
    """Builds the TaskConfig object that contains all configuration information. It builds this object from the JsonConfigModel stored in json_config_obj, the player_number, the sleep_time and the result_dir arguments.

    Parameters
    ----------
    - json_cofig_obj : JsonConfigModel
        The model containing all the content of the parsed config.json file
    - player_number : int
        The id of the player
    - sleep_time : float
        The number of seconds to sleep between the compilation share conversion step
    - result_dir : str
        The path to the directory which contains the results folder
    """
    conf_obj = TaskConfig(
        player_id=player_number,
        sleep_time=json_config_obj.sleep_time,
        player_count=json_config_obj.mpc.player_count,
        player_0_hostname=json_config_obj.mpc.player_0_hostname,
        abs_path_to_code_dir=json_config_obj.mpc.abs_path_to_code_dir,
        protocol_setup=json_config_obj.mpc.protocol_setup,
        script_args=json_config_obj.mpc.script_args,
        script_name=json_config_obj.mpc.script_name,
        custom_prime=json_config_obj.mpc.custom_prime,
        custom_prime_length=json_config_obj.mpc.custom_prime_length,
        result_dir=result_dir,
        stage=json_config_obj.mpc.stage,
        compiler_args=json_config_obj.mpc.compiler_args,
        program_args=json_config_obj.mpc.program_args,
        consistency_args=json_config_obj.consistency_args,
        commit_output=json_config_obj.commit_output,
        convert_ring_bits=json_config_obj.convert_ring_bits,
        convert_n_threads=json_config_obj.convert_n_threads,
        convert_chunk_size=json_config_obj.convert_chunk_size,
        convert_debug=json_config_obj.convert_debug,
        remove_input_files=json_config_obj.remove_input_files
    )
    return conf_obj


class TaskConfig(pydantic.BaseModel):
    """Defines the Configuration for a single Experiment Run

    Attributes
    ----------
    - player_id : int
        The id of the player
    - sleep_time : float
        The number of seconds to sleep between the compilation step and running step (Not used)
    - player_count : int
        The number of players involved in the experiment
    - player_0_hostname : str
        The hostname of player 0 to which all MPC protocol virtual machines should connect to
    - abs_path_to_code_dir : str
        The absolute path to the experiment code directory where the code for the evaluation framework is stored
    - script_name : str
        The name of the script that will be executed as part of the experiment
    - script_args : list[str]
        The list of 0-based positional arguments under which the given script should be compiled with
    - protocol_setup : ProtocolChoices
        The kind of MPC protocol that should be used in the Experiment
    - result_dir : str
        The path to the directory which contains the results folder
    """
    player_id: int
    sleep_time: float
    player_count: int
    player_0_hostname: str
    abs_path_to_code_dir: str
    # MPC specific options
    script_name: str
    script_args: typing.Dict[str, object]
    protocol_setup: ProtocolChoices
    result_dir: str
    stage: typing.Union[typing.Literal['compile', 'run'], typing.List[typing.Literal['compile', 'run']]] # TODO:
    program_args: dict = None
    custom_prime: typing.Optional[str] = None
    custom_prime_length: typing.Optional[str] = None

    convert_ring_if_needed: bool = True
    convert_ring_bits: int
    convert_n_threads: int
    convert_chunk_size: int
    convert_debug: bool

    compiler_args: list = None
    consistency_args: typing.Optional[JsonConsistencyConfig] = None
    commit_output: typing.Optional[bool] = False

    remove_input_files: bool

    @pydantic.validator('stage')
    def convert_to_list(cls, v):
        if not isinstance(v, list):
            v = [v]
        return v