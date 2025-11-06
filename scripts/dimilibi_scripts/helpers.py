import os, sys
from pathlib import Path
from joblib import Memory
from torch import float32 as torch_float32
import numpy as np
from scipy.stats import norm
import torch

# create a memory object from joblib to store the results of function calls throughout dimilibi scripts
memory = Memory("./cachedir", verbose=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import from dimilibi package
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from _old_vrAnalysis import database
from _old_vrAnalysis import tracking
from _old_vrAnalysis import analysis
from _old_vrAnalysis import session
from dimilibi import Population

SPEEDTHRESHOLD = 1  # speed threshold for filtering out ITIs -- I'm not saving this value so it's hard-coded here to be standardized whenever called.


@memory.cache
def get_sessions():
    # first get list of mice that I tend to use for analyses
    mousedb = database.vrDatabase("vrMice")
    df = mousedb.getTable(trackerExists=True)
    mouse_names = df["mouseName"].unique()

    # for each mouse, get the session identifiers and add them to a list
    all_sessions = {mouse_name: [] for mouse_name in mouse_names}
    for mouse_name in mouse_names:
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False)  # open up place cell multi session analysis object (don't autoload!!!)
        for pcss in pcm.pcss:
            datestr, sessionid = pcss.vrexp.dateString, pcss.vrexp.sessionid
            all_sessions[mouse_name].append((datestr, sessionid))

    return all_sessions


def load_session_data(mouse_name, datestr, sessionid, keep_planes=[1, 2, 3, 4], get_behavior=False, speedThreshold=1):
    """from session identifiers, load the spiking data from requested planes"""
    ses = session.vrExperiment(mouse_name, datestr, sessionid)
    ospks = ses.loadone("mpci.roiActivityDeconvolvedOasis")
    keep_idx = ses.idxToPlanes(keep_planes=keep_planes)
    if not get_behavior:
        return ospks[:, keep_idx]

    # if here, also get the behavior data
    frame_position, frame_environment, environments = ses.get_frame_behavior(speedThreshold=speedThreshold)
    ospks, frame_position, frame_environment = filter_timepoints(ospks, frame_position, frame_environment)
    behavior_data = dict(
        position=frame_position,
        environment=frame_environment,
        environments=environments,
    )
    return ospks[:, keep_idx], behavior_data


def filter_timepoints(ospks, frame_position, frame_environment):
    """filter out timepoints where the position is nan -- which correspond to ITIs"""
    idx_valid = ~np.isnan(frame_position)
    valid_position = frame_position[idx_valid]
    valid_environment = frame_environment[idx_valid]
    valid_spks = ospks[idx_valid]
    return valid_spks, valid_position, valid_environment


def create_population(mouse_name, datestr, sessionid, keep_planes=[1, 2, 3, 4], get_behavior=False):
    """from session identifiers, load the spiking data from requested planes and create a population object"""
    time_split_prms = dict(
        num_groups=3,
        relative_size=[5, 1, 1],
        chunks_per_group=25,
    )
    # if get_behavior is True, also get the behavior data
    if get_behavior:
        ospks, behavior_data = load_session_data(
            mouse_name, datestr, sessionid, keep_planes=keep_planes, get_behavior=get_behavior, speedThreshold=SPEEDTHRESHOLD
        )
    else:
        ospks = load_session_data(mouse_name, datestr, sessionid, keep_planes=keep_planes)

    # Make the population object
    npop = Population(ospks.T, generate_splits=True, time_split_prms=time_split_prms)

    if get_behavior:
        return npop, behavior_data

    # otherwise just return population
    return npop


def get_population_name(vrexp, population_name=None, get_behavior=False):
    """get the name of the population object"""
    name = f"population_{str(vrexp)}"
    if get_behavior:
        name += "_behavior"
    if population_name is not None:
        name += f"_{population_name}"
    return name


def save_population(population, mouse_name, datestr, sessionid, population_name=None, get_behavior=False):
    """save population object to cache"""
    pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
    indices_dict = population.get_indices_dict()
    pcss.save_temp_file(indices_dict, get_population_name(pcss.vrexp, population_name=population_name, get_behavior=get_behavior))


def load_population(mouse_name, datestr, sessionid, keep_planes=[1, 2, 3, 4], get_behavior=False, population_name=None):
    """load population object from cache"""
    pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
    indices_dict = pcss.load_temp_file(get_population_name(pcss.vrexp, population_name=population_name, get_behavior=get_behavior))
    if get_behavior:
        ospks, behavior_data = load_session_data(mouse_name, datestr, sessionid, keep_planes=keep_planes, get_behavior=get_behavior)
    else:
        ospks = load_session_data(mouse_name, datestr, sessionid, keep_planes=keep_planes)
    npop = Population.make_from_indices(indices_dict, ospks.T)
    npop.dtype = torch_float32
    if get_behavior:
        return npop, behavior_data
    return npop


def make_and_save_populations(all_sessions, keep_planes, get_behavior=False, population_name=None):
    """
    Make and save population objects for all sessions in all_sessions.

    All sessions is expected to be a dictionary of mouse_name: [(datestr, sessionid)])
    from the get_sessions function in this module.
    """
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            print(f"Creating population for: {mouse_name}, {datestr}, {sessionid}")
            if get_behavior:
                # we don't need to load the behavior data, but when it is required the npop will have a different structure (as long as speed is remembered)
                npop = create_population(mouse_name, datestr, sessionid, keep_planes=keep_planes, get_behavior=get_behavior)[0]
            else:
                npop = create_population(mouse_name, datestr, sessionid, keep_planes=keep_planes)
            save_population(npop, mouse_name, datestr, sessionid, population_name=population_name, get_behavior=get_behavior)


def get_ranks():
    """A subset of ranks to use for optimization and testing."""
    return (1, 2, 3, 5, 8, 15, 50, 100, 200)


def make_position_basis(position, environment, num_basis=10, basis_width=None, min_position=None, max_position=None):
    min_position = min_position or np.floor(np.min(position))
    max_position = max_position or np.ceil(np.max(position))
    basis_centers = np.linspace(min_position, max_position, num_basis + 2)[1:-1]
    basis_width = basis_width or (max_position - min_position) / (num_basis + 2)
    basis = norm.pdf(position[:, None], basis_centers, basis_width)
    environments = np.unique(environment)
    basis_by_env = np.zeros((len(position), len(environments), num_basis))
    for i, env in enumerate(environments):
        idx = np.where(environment == env)[0]
        basis_by_env[idx, i] = basis[idx]
    return torch.tensor(basis_by_env.reshape(len(position), num_basis * len(environments)))


def figure_folder():
    folder = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    if not folder.exists():
        folder.mkdir(parents=True)
    return folder
