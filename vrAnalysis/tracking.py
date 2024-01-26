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

from typing import List

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
        return list(self.mouse_path().glob(f"*{self.tracking_string+filetype}*"))
                    
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
            for stat_path, ops_path in zip(stat_paths, ops_paths):
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

    @handle_idx_ses
    @handle_keep_planes
    def split_by_plane(self, data: List[np.ndarray], dim=0, tracked=False, idx_ses=None, keep_planes=None):
        """
        helper method for splitting data by planes along dimension **dim** 
        
        data should be a list of numpy arrays containing some value associated with each ROI 
        concatenated across planes. this method will split each data array into a list of data
        arrays where each list corresponds to the part of the array associated with a particular
        imaging plane

        dim is used to set which dimension of the data array to index on (it's where the ROIs
        are concatenate)

        tracked=True assumes that data contains ROIs that were filtered by whether they had been
        tracked across the set of sessions in idx_ses
        """
        assert len(data)==len(idx_ses), "length of data list and idx_ses is not equal"

        # if using tracked ROIs only, get lookup of ROIs for each session
        if tracked:
            # this is a nested list with outer length = len(keep_planes) and inner length = len(idx_ses)
            # where the indices contain the indices to tracked ROIs (in order) for each plane/session combination
            # without the plane offset required for indexing into stacked data
            idx_to_ucid = self.get_idx_to_tracked(with_offset=False, idx_ses=idx_ses, keep_planes=keep_planes)
            tracked_roi_per_plane = np.stack([np.stack([len(idx) for idx in idx_ucid]) for idx_ucid in idx_to_ucid])

            # first and last roi per plane of tracked data
            first_last_roi = np.vstack((np.zeros((1, len(idx_ses))), np.cumsum(tracked_roi_per_plane, axis=0))).astype(int).T
        else:
            # get first and last roi for each plane of data
            # (num_session x num_plane) numpy array
            first_last_roi = np.vstack((np.zeros((1, len(idx_ses))), np.cumsum(self.roi_per_plane[keep_planes][:, idx_ses], axis=0))).astype(int).T

        # create index list for each plane for each session
        idx_to_plane = []
        for _, flr in enumerate(first_last_roi):
            cidx = []
            for iplane, _ in enumerate(keep_planes):
                # add a list of indices for this plane in this session
                cidx.append(list(range(flr[iplane], flr[iplane+1])))
            
            # add to list of lists of index to each plane for each session
            idx_to_plane.append(cidx)

        # check if size is correct (sum of length of index list for each plane should be equal to total number of ROIs for that session)
        expected_num_rois = [sum([len(idx) for idx in idx_plane]) for idx_plane in idx_to_plane]
        assert all([d.shape[dim]==exp_num for d, exp_num in zip(data, expected_num_rois)]), f"mismatch between number of elements on dim {dim} of data and expected number of ROIs in each plane"

        # break data down into list of arrays for each plane within each session
        data_by_plane = []
        for ses_data, idx_plane in zip(data, idx_to_plane):
            cdata = []
            for idx in idx_plane:
                cdata.append(helpers.index_on_dim(ses_data, idx, dim))
            data_by_plane.append(cdata)
        
        # return 
        return data_by_plane

    # ----- what follows this is a set of methods for retrieving aligned cell location and structure from the ROICaT pipeline -----
    @handle_idx_ses
    @handle_keep_planes
    def get_ROIs(self, idx_ses=None, keep_planes=None):
        """
        retrieve all ROIs from requested sessions and planes

        returns a list of ROI mask data from the requested sessions and planes
        where len(out)=len(idx_ses) and len(out[0])=len(keep_planes)
        
        each element is a coo_array with size (num_rois_per_plane(s), num_pixels)
        """
        return [[self._make_ROIs(plane, ses, as_coo=True) for plane in keep_planes] for ses in idx_ses]
    
    @handle_idx_ses
    @handle_keep_planes
    def get_centroids(self, method='weightedmean', combine=False, cat_planes=False, idx_ses=None, keep_planes=None):
        """
        retrieve the centroids of ROIs from requested sessions and planes

        returns a list of ROI centroids from the requested sessions and planes
        where len(out)=len(idx_ses) and len(out[0])=len(keep_planes)
        unless cat_planes=True, in which case each sublist is concatenated across planes

        method determines how to estimate the centroid from lam, ypix, xpix
        - 'weightedmean' will weigh the pixel coordinates by lam
        - 'median' will just take median of each x/y pixels

        if combine=True, will combine y/x coordinates into a 2d array for each ROI
        if combine=False, will keep y/x coordinates separated in two variables
        """
        # get list of lists of ROI mask data for each session / plane combination
        lam, ypix, xpix = self.get_roi_data(idx_ses=idx_ses, keep_planes=keep_planes)

        # convert each to centroid
        ycentroids, xcentroids = [], []
        for s_lam, s_ypix, s_xpix in zip(lam, ypix, xpix):

            # session centroids
            s_ycentroids, s_xcentroids = [], []
            for ps_lam, ps_ypix, ps_xpix in zip(s_lam, s_ypix, s_xpix):

                # get plane/session centroids by requested method
                if method == 'weightedmean':
                    ps_ycentroids = [np.sum(rlam * rypix)/np.sum(rlam) for rlam, rypix in zip(ps_lam, ps_ypix)]
                    ps_xcentroids = [np.sum(rlam * rxpix)/np.sum(rlam) for rlam, rxpix in zip(ps_lam, ps_xpix)]

                elif method == 'median':
                    ps_ycentroids = [np.median(rypix) for rypix in ps_ypix]
                    ps_xcentroids = [np.median(rxpix) for rxpix in ps_xpix]

                else:
                    raise ValueError("did not recognize centroid method")
                
                # add this planes centroids to the session
                s_ycentroids.append(np.stack(ps_ycentroids))
                s_xcentroids.append(np.stack(ps_xcentroids))
            
            if cat_planes:
                # concatenate across planes if requested
                s_ycentroids = np.concatenate(s_ycentroids)
                s_xcentroids = np.concatenate(s_xcentroids)

            # add this sessions centroids to the full list
            ycentroids.append(s_ycentroids)
            xcentroids.append(s_xcentroids)

        if combine:
            # combine into a 2d coordinate if requested
            return [[np.stack((yc, xc)).T for yc, xc in zip(ycent, xcent)] for ycent, xcent in zip(ycentroids, xcentroids)]
        
        # otherwise return centroids in separate variables
        return ycentroids, xcentroids
        

    @handle_idx_ses
    @handle_keep_planes
    def get_roi_data(self, idx_ses=None, keep_planes=None):
        """
        retrieve roi data from requested sessions and planes

        returns three lists of lists corresponding to the lam, ypix, and xpix of each ROI

        each list has:
        - len(lam)=len(idx_ses)
        - len(lam[i])=len(keep_planes)
        - len(lam[i][j])=num rois in plane j and session i
        """
        lam, ypix, xpix = [], [], []
        for ses in tqdm(idx_ses, desc='getting roi data'):
            clam, cxpix, cypix = helpers.named_transpose([self._make_lam_pix(plane, ses) for plane in keep_planes])
            lam.append(clam)
            ypix.append(cypix)
            xpix.append(cxpix)
        return lam, ypix, xpix
    
    def _make_lam_pix(self, plane, session):
        """
        makes lam, ypix, and xpix from ROIs in requested plane and session

        returns three lists corresponding to the lam, ypix & xpix of each ROI
        """
        def _get_lam_pix(single_row_sparse, num_pixels):
            # some ROIs are aligned out of frame so they have no data
            if len(single_row_sparse.data)==0:
                return np.nan, np.nan, np.nan
            
            # otherwise get their data
            lam = single_row_sparse.data
            ypix = np.fix(single_row_sparse.indices / num_pixels).astype(int)
            xpix = np.remainder(single_row_sparse.indices, num_pixels)
            return lam, ypix, xpix
        
        roi_sparse = self._make_ROIs(plane, session, as_coo=False)
        num_rois = roi_sparse.shape[0]
        num_pixels = int(np.sqrt(roi_sparse.shape[1]))
        lam, ypix, xpix = helpers.named_transpose([_get_lam_pix(roi_sparse[[i]], num_pixels) for i in range(num_rois)])
        return lam, ypix, xpix

    def _make_ROIs(self, plane, session, as_coo=True):
        """
        returns a (numROIs, numPixels) shaped sparse array containing post-aligned ROI data for requested plane and session

        defaults to coo array, but if as_coo=False then will return a csr_array
        """
        csr_data = self.rundata[plane]['aligner']['ROIs_aligned'][session]
        csr_array = sp.sparse.csr_array((csr_data['data'], csr_data['indices'], csr_data['indptr']), shape=csr_data['_shape'])
        if as_coo:
            return csr_array.tocoo()
        return csr_array
    

    # ----- what follows is a set of methods for retrieveing similarity scores from the ROICaT pipeline -----
    @handle_keep_planes
    def similarity_lookup(self, name, keep_planes=None, make_csr=True):
        """
        retrieve the requested similarity score from the ROICaT rundata
        
        see dictionary of lookup methods for explanation and possible names inside of function
        """
        lookup = {
            'sConj': lambda rundata: rundata['clusterer']['sConj'], 
            's_NN': lambda rundata: rundata['sim']['s_NN'],
            's_sf': lambda rundata: rundata['sim']['s_sf'],
            's_SWT': lambda rundata: rundata['sim']['s_SWT'],
            's_sesh': lambda rundata: rundata['sim']['s_sesh'],
            's_NN_z': lambda rundata: rundata['sim']['s_NN_z'],
            's_SWT_z': lambda rundata: rundata['sim']['s_SWT_z'],
        }

        similarity_data = [lookup[name](self.rundata[i]) for i in keep_planes]
        if make_csr: 
            return [sp.sparse.csr_array((scd['data'], scd['indices'], scd['indptr']), shape=scd['_shape']) for scd in similarity_data]
        
        return similarity_data

    @handle_idx_ses
    @handle_keep_planes
    def get_idx_roi_to_session_by_plane(self, tracked=False, split_sessions=False, idx_ses=None, keep_planes=None):
        """
        returns a list of indices containing the ROIs in idx_ses from keep_planes
        
        helper method for retrieving the idx to all ROIs in requested sessions
        from each plane. If there are N sessions tracked (from 0 to N-1), and x_n_p
        ROIs in session n in plane p, then ROICaT will save data with SUM_n (x_n_p)
        dimensions for each plane (containing all ROIs stacked in order of sessions).

        to pull out target sessions from each plane, we need an index that pulls out
        the relevant rows and columns (or whatever else structure, but this was 
        designed to be used with sparse matrices which require integer indexing).    

        if tracked=True, will filter and sort by tracked ROIs such that if there are 
        10 tracked ROIs in plane 0 and 7 sessions, the similarity matrix after indexing
        using idx_roi_to_session will be (10*7, 10*7). Can also use tracked='not' to 
        specifically look at ROIs that ~aren't~ tracked.
        """
        # if using tracked ROIs only, get lookup of ROIs for each session
        if tracked:
            # this is a nested list with outer length = len(keep_planes) and inner length = len(idx_ses)
            # where the indices contain the indices to tracked ROIs (in order) for each plane/session combination
            # without the plane offset required for indexing into stacked data
            idx_to_ucid = self.get_idx_to_tracked(with_offset=False, idx_ses=idx_ses, keep_planes=keep_planes)

        # get number of ROIs per plane from each session (for requested planes)
        first_last_roi = np.hstack((np.zeros((len(keep_planes),1)), np.cumsum(self.roi_per_plane[keep_planes], axis=1))).astype(int)

        # concatenate slices of ROI from plane indices
        idx_roi_to_session = []
        for ii, flr in enumerate(first_last_roi):
            cidx = []
            for jj, ises in enumerate(idx_ses):
                # this is the set of indices for this particular plane and session corresponding to the rows / columns
                # in the similarity matrix that correspond to those ROIs similarity comparisons
                cindices = list(range(flr[ises], flr[ises+1]))
                if tracked and tracked=='not':
                    # filter out any that are tracked if requested
                    cindices = [cind for ii, cind in enumerate(cindices) if ii not in idx_to_ucid[ii][jj]]
                elif tracked:
                    # filter for those that are tracked if requested
                    cindices = [cindices[i] for i in idx_to_ucid[ii][jj]]
                if split_sessions:
                    # If splitting across sessions, make a list of lists of indices to each session
                    cidx.append(cindices)
                else:
                    # If not splitting across sessions, make a single list for all sessions for each plane
                    cidx += cindices
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
    def get_similarity(self, name, tracked=False, cat_planes=False, idx_ses=None, keep_planes=None):
        """
        retrieve sparse similarity data and consolidate across planes
        
        only retrieves data from requested sessions and planes using **idx_ses** and **keep_planes**
        
        if tracked=True, will filter by whatever ROIs are officially "tracked" according to ROICaT
        and whatever criterion are defined in this class instance. Otherwise returns data for all ROIs 
        can also use tracked='not' to only return those that aren't tracked

        if cat_planes=True, will create a full similarity matrix stacked across planes (with all
        off-diagonal entries zero because they aren't defined in ROICaT)
        """
        # get sparse similarity data from requested planes
        sparse = self.similarity_lookup(name, keep_planes=keep_planes, make_csr=True)

        # get idx to ROIs in each plane
        idx_roi_to_session = self.get_idx_roi_to_session_by_plane(tracked=tracked, split_sessions=False, idx_ses=idx_ses, keep_planes=keep_planes)

        # filter sparse matrices
        sparse = self._filter_sparse_by_index(sparse, idx_roi_to_session)

        # concatenate across planes if requested
        if cat_planes:
            sparse = self._concatenate_sparse_across_planes(sparse)
        
        return sparse
    
    @handle_keep_planes
    def get_similarity_paired(self, name, source=None, target=None, symmetric=True, tracked=False, cat_planes=False, keep_planes=None):
        """
        retrieve sparse similarity data from a pair of sessions (source and target)

        returns the requested similarity data (by **name**) in the format of a sparse matrix
        with size (#ROIs_in_source, #ROIs_in_target). Since some similarity data is not symmetric, 
        using the symmetric=True kwarg setting will take the average of the [row, col] & [col, row]
        values. If symmetric=False, will just take [row, col] where row<source, and col<target.

        if tracked=True, will filter by whatever ROIs are officially "tracked" according to ROICaT
        and whatever criterion are defined in this class instance. Otherwise returns data for all ROIs 
        can also use tracked='not' to only return those that aren't tracked

        if cat_planes=True, will create a full similarity matrix stacked across planes (with all
        off-diagonal entries zero because they aren't defined in ROICaT)
        """
        # get sparse similarity data from requested planes
        sparse = self.similarity_lookup(name, keep_planes=keep_planes, make_csr=True)

        # get idx to ROIs in each plane
        idx_ses = [source, target]
        idx_roi_to_session = self.get_idx_roi_to_session_by_plane(tracked=tracked, split_sessions=True, idx_ses=idx_ses, keep_planes=keep_planes)

        # filter sparse matrix
        sparse_pair = [s[idx[0]][:, idx[1]] for s, idx in zip(sparse, idx_roi_to_session)]
        
        # if symmetric is requested, take the average of the data with it's transpose
        if symmetric:
            # this is the same ROI pairs but organized by (target, source) 
            # which isn't always symmetric... 
            sparse_reflection = [s[idx[1]][:, idx[0]].transpose().tocsr() for s, idx in zip(sparse, idx_roi_to_session)]
            # average the data from the [source, target] and [target, source] looks
            for idx, (sp, sr) in enumerate(zip(sparse_pair, sparse_reflection)):
                sparse_pair[idx].data = np.mean(np.stack((sp.data, sr.data)), axis=0)

        # concatenate across planes if requested
        if cat_planes:
            sparse_pair = self._concatenate_sparse_across_planes(sparse_pair)
        
        return sparse_pair

    @handle_idx_ses
    @handle_keep_planes
    def get_tracked_labels(self, cat_planes=False, nan_untracked=True, idx_ses=None, keep_planes=None):
        """
        get tracking labels across sessions
        convert untracked to nans if requested (untracked are = -1), replace with np.nan if nan_untracked=True
        """
        # helper function 
        def retrieve_labels(results_for_plane, ses, nan_untracked):
            out = results_for_plane['clusters']['labels_bySession'][ses]
            if nan_untracked:
                out = out.astype(np.float32)
                np.put(out, np.where(out==-1)[0], np.nan)
            return out
        
        # list of lists of labels per plane for each session in idx_ses
        labels = [[retrieve_labels(self.results[plane], ses, nan_untracked) for plane in keep_planes] for ses in idx_ses]

        # concatenate across planes if requested
        if cat_planes:
            labels = [np.concatenate(label_by_plane) for label_by_plane in labels]
        
        # return labels
        return labels
    
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
        

