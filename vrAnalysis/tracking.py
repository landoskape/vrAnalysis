# inclusions
import re
import time
from tqdm import tqdm
import pickle
from pathlib import Path 
import numpy as np

# import package
from . import session
from . import functions
from . import helpers
from . import database
from . import fileManagement as fm

vrdb = database.vrDatabase()

# Variables that might need to be changed for different users
# if anyone other than me uses this, let me know and I can make it smarter by using a user dictionary or storing a file somewhere else...
data_path = fm.localDataPath()

class tracker():
    def __init__(self, mouse_name, tracking_string='ROICaT.tracking'):
        """create tracker object for a particular mouse"""
        self.mouse_name = mouse_name
        self.tracking_string = tracking_string
        self.results_string = '.results'
        self.rundata_string = '.rundata'
        self.num_parts_data_path = len(self.data_path().parts) 

        # identify tracking files
        self.identify_tracking_files()
        self.num_planes = len(self.results_files)
        assert len(self.results_files)>0, "no tracking files found"
        assert len(self.results_files)==len(self.rundata_files), "results and rundata have different numbers of files"
        
        
        # load tracking data
        self.load_tracking_files()

        # identify sessions that were tracked and create session objects for them
        self.session_names = self.identify_tracked_sessions()
        self.sessions = [session.vrExperiment(*session_name) for session_name in self.session_names]
        self.num_sessions = len(self.session_names)

        # check that number of ROIs is as expected
        self.check_num_rois_per_plane()
        
    # path methods
    def data_path(self):
        """returns data path where data is saved"""
        return data_path

    def mouse_path(self): 
        """path to mouse folder (assuming Alyx format)"""
        return self.data_path() / self.mouse_name

    # methods for loading tracking data
    def process_file_name(self, filename, filetype):
        assert (filetype=='results') or (filetype=='rundata'), f"did not recognize filetype ({filetype}), should be either 'results' or 'rundata'"
        pattern = fr"(\w+)\.(plane\d+)\.{self.tracking_string}\.{filetype}"
        match = re.search(pattern, filename)
        assert match, f"{filename} is not a valid filename"
        
        # check mouse name
        mouseName = match.group(1)
        assert mouseName==self.mouse_name, f"mouse name found in file ({mouseName}) doesn't match mouse name of this tracker object ({self.mouse_name})!"

        # return plane name
        return match.group(2) 
        
    def list_tracking_files(self, filetype):
        return list(self.mouse_path().rglob(f"*{self.tracking_string+filetype}*"))
                    
    def identify_tracking_files(self):
        """identify files where tracking data is stored"""
        self.results_files = [p.stem for p in self.list_tracking_files(self.results_string)]
        self.rundata_files = [p.stem for p in self.list_tracking_files(self.rundata_string)]
        self.plane_names = [self.process_file_name(f, 'results') for f in self.results_files]
        assert all([pn == self.process_file_name(f, 'rundata') for pn, f in zip(self.plane_names, self.rundata_files)]), "plane names don't match in results and rundata"
        
        suffices = np.unique([p.suffix for p in self.list_tracking_files(self.results_string)] + [p.suffix for p in self.list_tracking_files(self.rundata_string)])
        assert len(suffices)==1, f"suffices are multifarious... rename files so there's only 1! suffices found: {suffices}"
        self.suffix = suffices[0]
        
    def load_tracking_files(self):
        """load files where tracking data is stored"""
        self.results = []
        self.rundata = []
        for file in self.results_files:
            with open(self.mouse_path()/(file+self.suffix), 'rb') as f:
                self.results.append(pickle.load(f))
        for file in self.rundata_files:
            with open(self.mouse_path()/(file+self.suffix), 'rb') as f:
                self.rundata.append(pickle.load(f))

    def identify_tracked_sessions(self):
        """identify which sessions were tracked by filename"""

        # idiosyncratic helper function only used here
        def check_across_planes(values_each_plane):
            """helper function for getting single unique value per session from a list across planes"""
            # get list of lists where each inner list contains all unique values for that element within the input
            value_each_session = list(map(list, map(set, zip(*values_each_plane))))
            assert all([len(val)==1 for val in value_each_session]), "some planes had different values!!! (inspect tracked session indicators)"
            value = [v[0] for v in value_each_session] # convert to list of values now that we know each len(v)==1
            return value

        # store tracked session indicators here
        tracked_mouse_name = [[] for _ in range(self.num_planes)]
        tracked_date_string = [[] for _ in range(self.num_planes)]
        tracked_session_id = [[] for _ in range(self.num_planes)]

        # for each plane...
        for planeidx, results in enumerate(self.results):
            # get paths to stat files and ops files
            stat_paths = [Path(sp) for sp in results['input_data']['paths_stat']]
            ops_paths = [Path(sp) for sp in results['input_data']['paths_ops']]
            for sessionidx, (stat_path, ops_path) in enumerate(zip(stat_paths, ops_paths)):
                # then for each file path, make sure it comes from the same storage location
                assert all([sp == dp for (sp,dp) in zip(stat_path.parts[:self.num_parts_data_path], self.data_path().parts)]), f"stat_path ({stat_path}) does not match data_path!"
                assert all([op == dp for (op,dp) in zip(ops_path.parts[:self.num_parts_data_path], self.data_path().parts)]), f"ops_path ({ops_path}) does not match data_path!"

                # the mouse name, date string, and session id are the next "groups" in the path according to Alyx database convention
                tracked_mouse_name[planeidx].append(stat_path.parts[self.num_parts_data_path])
                tracked_date_string[planeidx].append(stat_path.parts[self.num_parts_data_path+1])
                tracked_session_id[planeidx].append(stat_path.parts[self.num_parts_data_path+2])

                # check that session indicators are the same between the stat paths and ops paths
                assert stat_path.parts[self.num_parts_data_path]==ops_path.parts[self.num_parts_data_path], f"mouse name doesn't match for stat and ops: ({stat_path}), ({ops_path})"
                assert stat_path.parts[self.num_parts_data_path+1]==ops_path.parts[self.num_parts_data_path+1], f"date string doesn't match for stat and ops: ({stat_path}), ({ops_path})"
                assert stat_path.parts[self.num_parts_data_path+2]==ops_path.parts[self.num_parts_data_path+2], f"session id doesn't match for stat and ops: ({stat_path}), ({ops_path})"

        # make sure value is consistent across planes and return single value per session if it is
        tracked_mouse_name = check_across_planes(tracked_mouse_name)
        tracked_date_string = check_across_planes(tracked_date_string)
        tracked_session_id = check_across_planes(tracked_session_id)

        # return tuple of session name (mousename, datestring, sessionid)
        return [session_name for session_name in zip(tracked_mouse_name, tracked_date_string, tracked_session_id)]

    def check_num_rois_per_plane(self):
        """checks if number of ROIs in ROICaT labels matches number of ROIs in each session for each plane"""
        def assertion_message(planeidx, label, session):
            return f"For session {session.sessionPrint()} and plane {planeidx}, # ROICaT ROIs ({len(label)}) doesn't match session ({session.value['roiPerPlane'][planeidx]})"
        for planeidx, results in enumerate(self.results):
            for labels, session in zip(results['clusters']['labels_bySession'], self.sessions):
                assert len(labels)==session.value['roiPerPlane'][planeidx], assertion_message(planeidx, labels, session)

                






