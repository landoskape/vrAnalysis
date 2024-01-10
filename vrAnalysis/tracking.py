# inclusions
import re
import time
from tqdm import tqdm
import pickle
from pathlib import Path 
from functools import wraps
import numpy as np
import scipy as sp
import pandas as pd

# import package
from . import session
from . import functions
from . import helpers
from . import database
from . import fileManagement as fm

sessiondb = database.vrDatabase('vrSessions')

# Variables that might need to be changed for different users
# if anyone other than me uses this, let me know and I can make it smarter by using a user dictionary or storing a file somewhere else...
data_path = fm.localDataPath()

# ---- decorators for tracker class methods ----
def handle_keep_planes(func):
    """decorator to handle the keep_planes argument in a standard way for the tracker class"""
    @wraps(func)
    def wrapper(tracker_instance, *args, keep_planes=None, **kwargs):
        keep_planes = tracker_instance.get_keep_planes(keep_planes=keep_planes)
        return func(tracker_instance, *args, keep_planes=keep_planes, **kwargs)
    return wrapper

def handle_idx_ses(func):
    """decorator to handle the idx_ses argument in a standard way for the tracker class"""
    @wraps(func)
    def wrapper(tracker_instance, *args, idx_ses=None, **kwargs):
        idx_ses = tracker_instance.get_idx_session(idx_ses=idx_ses)
        return func(tracker_instance, *args, idx_ses=idx_ses, **kwargs)
    return wrapper


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
    def get_keep_planes(self, keep_planes=None):
        return keep_planes if keep_planes is not None else np.arange(self.num_planes)
        
    def get_idx_session(self, idx_ses=None):
        return idx_ses if idx_ses is not None else np.arange(self.num_sessions)
        
    # database utilities
    @handle_idx_ses
    def session_table(self, idx_ses=None, reset_index=True):
        """return dataframe of requested sessions from database"""
        records = [sessiondb.getRecord(*self.sessions[ii].sessionName()) for ii in idx_ses]
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
                assert all([sp.lower() == dp.lower() for (sp,dp) in zip(stat_path.parts[:self.num_parts_data_path], self.data_path().parts)]), \
                f"stat_path ({stat_path}) does not match data_path ({self.data_path()})!"
                assert all([op.lower() == dp.lower() for (op,dp) in zip(ops_path.parts[:self.num_parts_data_path], self.data_path().parts)]), \
                f"ops_path ({ops_path}) does not match data_path ({self.data_path()})!"

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

    @handle_idx_ses
    @handle_keep_planes
    def prepare_tracking_idx(self, idx_ses=None, keep_planes=None):
        """get index to tracked ROIs for a list of sessions"""
        # get number of sessions used
        num_ses = len(idx_ses)

        # ucids in list of lists for requested sessions
        ucids = [[[] for _ in range(num_ses)] for _ in range(len(keep_planes))]
        for planeidx, results in enumerate([self.results[p] for p in keep_planes]):
            for sesidx, idx in enumerate(idx_ses):
                ucids[planeidx][sesidx] = results['clusters']['labels_bySession'][idx]

        # this is the number of unique IDs per plane
        num_ucids = [max([np.max(u) for u in ucid])+1 for ucid in ucids]

        # this is a boolean array of size (number unique IDs x num sessions) where there is a 1 if a unique ROI is found in each session
        roicat_index = [np.zeros((nucids, num_ses), dtype=bool) for nucids in num_ucids]
        for planeidx, ucid in enumerate(ucids):
            for sesidx, uc in enumerate(ucid):
                cindex = uc[uc >= 0] # index of ROIs (UCIDs) found in this session in this plane (excluding -1s)
                roicat_index[planeidx][cindex, sesidx] = True # label found ROI with True

        return ucids, roicat_index
    

    @handle_idx_ses
    @handle_keep_planes
    def get_idx_to_tracked(self, with_offset=False, idx_ses=None, keep_planes=None):
        """
        retrieve indices to tracked ROIs for list of sessions
        """
        # get ucids and 1s index for requested sessions
        ucids, roicat_index = self.prepare_tracking_idx(idx_ses=idx_ses, keep_planes=keep_planes)
        
        # list of UCIDs in all requested sessions (a list of the UCIDs...)
        idx_in_ses = [np.where(np.all(rindex, axis=1))[0] for rindex in roicat_index]
        
        # For each plane & session, a sorted index to the suite2p ROI to recreate the list of UCIDs
        idx_to_ucid = [[helpers.index_in_target(iis, uc)[1] for uc in ucid] for (iis, ucid) in zip(idx_in_ses, ucids)]

        # if with_offset, add offset for number of ROIs in each plane
        if with_offset:
            # cumulative number of ROIs before each plane (in numeric order of planes using sorted(self.plane_names))
            roi_per_plane = self.roi_per_plane[keep_planes][:, idx_ses]
            roi_plane_offset = np.cumsum(np.vstack((np.zeros((1, len(idx_ses)), dtype=int), roi_per_plane[:-1])), axis=0)
            idx_to_ucid = [[offset+ucid for offset, ucid in zip(offsets, ucids)] for offsets, ucids in zip(roi_plane_offset, idx_to_ucid)]

        # return indices
        return idx_to_ucid

    @handle_idx_ses
    @handle_keep_planes
    def get_tracked_idx(self, idx_ses=None, keep_planes=None):
        """
        retrieve indices to tracked ROIs for list of sessions

        returns a (num_session, num_tracked) shape numpy array where each column contains
        integer indices to the tracked ROIs from each session. The integer indices are stacked
        indices to tracked ROIs in each plane after applying an offset for the number of ROIs 
        in each plane.
        """
        # For each plane & session, a sorted index to the suite2p ROI to recreate the list of UCIDs
        idx_to_ucid = self.get_idx_to_tracked(with_offset=True, idx_ses=idx_ses, keep_planes=keep_planes)

        # A straightforward numpy array of (numSessions, numROIs) containing the indices to retrieve tracked and sorted ROIs
        return np.concatenate([np.stack([ucid for ucid in ucids], axis=1) for ucids in idx_to_ucid], axis=0).T


    # ----- what follows is a set of methods for retrieveing similarity scores from the ROICaT pipeline -----
    @handle_keep_planes
    def similarity_lookup(self, name, keep_planes=None, make_csr=True):
        """
        retrieve the requested similarity score from the ROICaT rundata
        
        see dictionary of lookup method for explanation and possible names inside of function
        """
        lookup = {
            'sConj': lambda rundata: rundata['clusterer']['sConj'], 
            's_NN': lambda rundata: rundata['sim']['s_NN'],
            's_sf': lambda rundata: rundata['sim']['s_sf'],
            's_SWT': lambda rundata: rundata['sim']['s_SWT'],
            's_sesh': lambda rundata: rundata['sim']['s_sesh'],
        }

        similarity_data = [lookup[name](self.rundata[i]) for i in keep_planes]
        if make_csr: 
            return [sp.sparse.csr_array((scd['data'], scd['indices'], scd['indptr'])) for scd in similarity_data]
        
        return similarity_data

    @handle_idx_ses
    @handle_keep_planes
    def get_idx_roi_to_session_by_plane(self, idx_ses=None, keep_planes=None):
        """
        returns a list of indices containing the ROIs in idx_ses from keep_planes
        
        helper method for retrieving the idx to all ROIs in requested sessions
        from each plane. If there are N sessions tracked (from 0 to N-1), and x_n_p
        ROIs in session n in plane p, then ROICaT will save data with SUM_n (x_n_p)
        dimensions for each plane (containing all ROIs stacked in order of sessions).

        to pull out target sessions from each plane, we need an index that pulls out
        the relevant rows and columns (or whatever else structure, but this was 
        designed to be used with sparse matrices which require integer indexing).        
        """
        # get number of ROIs per plane from each session (for requested planes)
        first_last_roi = np.hstack((np.zeros((len(keep_planes),1)), np.cumsum(self.roi_per_plane[keep_planes], axis=1))).astype(int)

        # concatenate slices of ROI from plane indices
        idx_roi_to_session = []
        for flr in first_last_roi:
            cidx = []
            for ises in idx_ses:
                cidx += list(range(flr[ises], flr[ises+1]))
            idx_roi_to_session.append(cidx)

        return idx_roi_to_session

    def _filter_sparse_by_index(self, list_sparse, list_idx):
        """helper for retrieveing requested index in rows and columns from sparse matrices"""
        return [sm[irts][:, irts] for sm, irts in zip(list_sparse, list_idx)]
    
    def _concatenate_sparse_across_planes(self, list_sparse):
        """
        helper for concatenating sparse matrices across planes
        
        **list_sparse** is a list of sparse csr matrices
        
        this method will concatenate them such that each matrix provided
        in the **list_sparse** argument makes the main diagonal of a block
        matrix (so all off-diagonal elements between matrices in list_sparse
        are set to 0...).
        """
        
        sparse_full_rows = []
        for ii in range(len(list_sparse)):
            row = []
            for jj in range(len(list_sparse)):
                if ii==jj:
                    # if on main-diagonal of block, append the provided sparse matrix
                    row.append(list_sparse[ii])
                else:
                    # otherwise make an empty (all 0s) csr matrix with the right dimensions
                    row.append(sp.sparse.csr_array((list_sparse[ii].shape[0], list_sparse[jj].shape[1])))

            # concatenate columns within this row 
            sparse_full_rows.append(sp.sparse.hstack(row, format='csr'))
            
        # concatenate all rows to form the full square(usually) block matrix
        return sp.sparse.vstack(sparse_full_rows, format='csr')

    @handle_idx_ses
    @handle_keep_planes
    def get_similarity(self, name, tracked=True, idx_ses=None, keep_planes=None):
        """
        retrieve sparse similarity data and consolidate across planes
        
        only retrieves data from requested sessions and planes using **idx_ses** and **keep_planes**
        
        if tracked=True, will filter by whatever ROIs are officially "tracked" according to ROICaT
        and whatever criterion are defined in this class instance. Otherwise returns data for all ROIs 
        """
        # get sparse similarity data from requested planes
        sparse = self.similarity_lookup(name, keep_planes=keep_planes, make_csr=True)

        # get idx to ROIs in each plane
        idx_roi_to_sesion = self.get_idx_roi_to_session_by_plane(idx_ses=idx_ses, keep_planes=keep_planes)

        # filter sparse matrices
        sparse_filtered = self._filter_sparse_by_index(sparse, idx_roi_to_sesion)

        # concatenate 
        sparse_full = self._concatenate_sparse_across_planes(sparse_filtered)
        
        return sparse_full
    
    @handle_idx_ses
    @handle_keep_planes
    def check_red_cell_consistency(self, idx_ses=None, keep_planes=None, use_s2p=False, s2p_cutoff=0.65):
        # get idx of tracked ROIs
        idx_tracked = self.get_tracked_idx(idx_ses=idx_ses, keep_planes=keep_planes)

        # get red cell assignments
        if not(use_s2p):
            idx_red = np.stack([self.sessions[ii].getRedIdx(keep_planes=keep_planes)[it] for ii,it in zip(idx_ses, idx_tracked)])
        else:
            c_in_plane = [self.sessions[ii].idxToPlanes(keep_planes=keep_planes) for ii in idx_ses]
            idx_red = np.stack([self.sessions[ii].loadone('mpciROIs.redS2P')[cip][it] > s2p_cutoff for ii, cip, it in zip(idx_ses, c_in_plane, idx_tracked)])
        
        return idx_red
        

