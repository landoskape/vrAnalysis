import os, sys
from joblib import Memory
from torch import float32 as torch_float32
import torch

# create a memory object from joblib to store the results of function calls throughout dimilibi scripts
memory = Memory("./cachedir", verbose=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import from dimilibi package
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from vrAnalysis import database
from vrAnalysis import tracking
from vrAnalysis import analysis
from vrAnalysis import session
from dimilibi import Population


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


def load_session_data(mouse_name, datestr, sessionid, keep_planes=[1, 2, 3, 4]):
    """from session identifiers, load the spiking data from requested planes"""
    ses = session.vrExperiment(mouse_name, datestr, sessionid)
    ospks = ses.loadone("mpci.roiActivityDeconvolvedOasis")
    keep_idx = ses.idxToPlanes(keep_planes=keep_planes)
    return ospks[:, keep_idx]


def create_population(mouse_name, datestr, sessionid, keep_planes=[1, 2, 3, 4]):
    """from session identifiers, load the spiking data from requested planes and create a population object"""
    ospks = load_session_data(mouse_name, datestr, sessionid, keep_planes=keep_planes)
    time_split_prms = dict(
        num_groups=3,
        relative_size=[5, 1, 1],
        chunks_per_group=25,
    )
    return Population(ospks.T, generate_splits=True, time_split_prms=time_split_prms)


def get_population_name(vrexp, population_name=None):
    """get the name of the population object"""
    return f"population_{str(vrexp)}_{population_name}" if population_name is not None else f"population_{str(vrexp)}"

def save_population(population, mouse_name, datestr, sessionid, population_name=None):
    """save population object to cache"""
    pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
    indices_dict = population.get_indices_dict()
    pcss.save_temp_file(indices_dict, get_population_name(pcss.vrexp, population_name=population_name))


def load_population(mouse_name, datestr, sessionid, population_name=None):
    """load population object from cache"""
    pcss = analysis.placeCellSingleSession(session.vrExperiment(mouse_name, datestr, sessionid), autoload=False)
    indices_dict = pcss.load_temp_file(get_population_name(pcss.vrexp, population_name=population_name))
    ospks = load_session_data(mouse_name, datestr, sessionid)
    npop = Population.make_from_indices(indices_dict, ospks.T)
    npop.dtype = torch_float32
    return npop


def make_and_save_populations(all_sessions, population_name=None):
    """
    Make and save population objects for all sessions in all_sessions.

    All sessions is expected to be a dictionary of mouse_name: [(datestr, sessionid)])
    from the get_sessions function in this module.
    """
    for mouse_name, sessions in all_sessions.items():
        for datestr, sessionid in sessions:
            print(f"Creating population for: {mouse_name}, {datestr}, {sessionid}")
            npop = create_population(mouse_name, datestr, sessionid)
            save_population(npop, mouse_name, datestr, sessionid, population_name=population_name)


def get_ranks():
    """A subset of ranks to use for optimization and testing."""
    return (1, 2, 3, 5, 8, 15, 50, 100, 200)