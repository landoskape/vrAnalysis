# inclusions
import re
import time
from tqdm import tqdm
import pickle
from pathlib import Path 
import numpy as np
import pandas as pd

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

    # basic utilities 
    def get_keepPlanes(self, keepPlanes=None):
        keepPlanes = keepPlanes if keepPlanes is not None else np.arange(self.num_planes)
        num_planes = len(keepPlanes)
        return keepPlanes, num_planes
        
    def get_idx_session(self, idx_ses=None):
        idx_ses = idx_ses if idx_ses is not None else np.arange(self.num_sessions)
        num_ses = len(idx_ses)
        return idx_ses, num_ses
        
    # database utilities
    def session_table(self, idx_ses=None, reset_index=True):
        """return dataframe of requested sessions from database"""
        idx_ses, num_ses = self.get_idx_session(idx_ses=idx_ses)
        records = [vrdb.getRecord(*self.sessions[ii].sessionName()) for ii in idx_ses]
        df = pd.concat(records, axis=1).T
        if reset_index: 
            df = df.reset_index(drop=True)
        return df
        
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
        assert self.plane_names == sorted(self.plane_names), f"plane_names are not sorted properly.. ({self.plane_names})"
        
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
        """
        get number of rois per plane
        and also checks if number of ROIs in ROICaT labels matches number of ROIs in each session for each plane
        """
        def assertion_message(planeidx, label, session):
            return f"For session {session.sessionPrint()} and plane {planeidx}, # ROICaT ROIs ({len(label)}) doesn't match session ({session.value['roiPerPlane'][planeidx]})"

        self.roi_per_plane = np.zeros((self.num_planes, self.num_sessions), dtype=int)
        for planeidx, results in enumerate(self.results):
            for sesidx, (labels, session) in enumerate(zip(results['clusters']['labels_bySession'], self.sessions)):
                assert len(labels)==session.value['roiPerPlane'][planeidx], assertion_message(planeidx, labels, session)
                self.roi_per_plane[planeidx, sesidx] = session.value['roiPerPlane'][planeidx]

    def prepare_tracking_idx(self, idx_ses=None, keepPlanes=None):
        """get index to tracked ROIs for a list of sessions"""
        # which planes to keep
        keepPlanes, num_planes = self.get_keepPlanes(keepPlanes=keepPlanes)
        
        # get session index
        idx_ses, num_ses = self.get_idx_session(idx_ses=idx_ses)
        
        # ucids in list of lists for requested sessions
        ucids = [[[] for _ in range(num_ses)] for _ in range(num_planes)]
        for planeidx, results in enumerate([self.results[p] for p in keepPlanes]):
            for sesidx, idx in enumerate(idx_ses):
                ucids[planeidx][sesidx] = results['clusters']['labels_bySession'][idx]

        # this is the number of unique IDs per plane
        num_ucids = [max([np.max(u) for u in ucid])+1 for ucid in ucids]

        # this is a boolean array of size (number unique IDs x num sessions) where there is a 1 if a unique ROI is found in each session
        roicat_index = [np.zeros((nucids, num_ses), dtype=bool) for nucids in num_ucids]
        for planeidx, ucid in enumerate(ucids):
            for sesidx, uc in enumerate(ucid):
                cindex = uc[uc >= 0] # index of ROIs found in this session
                roicat_index[planeidx][cindex, sesidx] = True # label found ROI with True

        return ucids, roicat_index
    
    def get_tracked_idx(self, idx_ses=None, keepPlanes=None):
        """retrieve indices to tracked ROIs for list of sessions"""
        # which planes to keep
        keepPlanes, num_planes = self.get_keepPlanes(keepPlanes=keepPlanes)

        # get session idx
        idx_ses, num_ses = self.get_idx_session(idx_ses=idx_ses)
        
        # get ucids and 1s index for requested sessions
        ucids, roicat_index = self.prepare_tracking_idx(idx_ses=idx_ses, keepPlanes=keepPlanes)
        
        # list of UCIDs in all requested sessions (a list of the UCIDs...)
        idx_in_ses = [np.where(np.all(rindex, axis=1))[0] for rindex in roicat_index]
        
        # For each plane & session, a sorted index to the suite2p ROI to recreate the list of UCIDs
        idx_to_ucid = [[helpers.index_in_target(iis, uc)[1] for uc in ucid] for (iis, ucid) in zip(idx_in_ses, ucids)]

        # cumulative number of ROIs before eacg plane (in numeric order of planes using sorted(self.plane_names))
        roi_per_plane = self.roi_per_plane[keepPlanes][:, idx_ses]
        roi_plane_offset = np.cumsum(np.vstack((np.zeros((1,num_ses),dtype=int), roi_per_plane[:-1])), axis=0)

        # A straightforward numpy array of (numSessions, numROIs) containing the indices to retrieve tracked and sorted ROIs
        return np.concatenate([np.stack([offset+ucid for offset, ucid in zip(offsets, ucids)], axis=1) for offsets, ucids in zip(roi_plane_offset, idx_to_ucid)], axis=0).T

    def check_red_cell_consistency(self, idx_ses=None, keepPlanes=None, use_s2p=False, s2p_cutoff=0.65):
        # which planes to keep
        keepPlanes, num_planes = self.get_keepPlanes(keepPlanes=keepPlanes)

        # get session idx
        idx_ses, num_ses = self.get_idx_session(idx_ses=idx_ses)

        # get idx of tracked ROIs
        idx_tracked = self.get_tracked_idx(idx_ses=idx_ses, keepPlanes=keepPlanes)

        # get red cell assignments
        if not(use_s2p):
            idx_red = np.stack([self.sessions[ii].getRedIdx(keepPlanes=keepPlanes)[it] for ii,it in zip(idx_ses, idx_tracked)])
        else:
            c_in_plane = [self.sessions[ii].idxToPlanes(keepPlanes=keepPlanes) for ii in idx_ses]
            idx_red = np.stack([self.sessions[ii].loadone('mpciROIs.redS2P')[cip][it] > s2p_cutoff for ii, cip, it in zip(idx_ses, c_in_plane, idx_tracked)])
        
        return idx_red
        

