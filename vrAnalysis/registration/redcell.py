import time
import numpy as np
from _old_vrAnalysis import helpers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .register import B2Registration


class RedCellProcessing:
    """
    The redCellProcessing object is devoted to handling red cell processing.
    It accepts as input a B2Registration object.
    """

    def __init__(self, b2_registration: "B2Registration", um_per_pixel: float = 1.3, autoload: bool = True):

        # Make sure redcell is available...
        msg = "redcell is not an available suite2p output, so you can't do redCellProcessing."
        assert "redcell" in b2_registration.get_value("available"), msg

        self.b2_registration = b2_registration

        # standard names of the features used to determine red cell criterion
        self.feature_names = ["S2P", "dotProduct", "pearson", "phaseCorrelation"]

        # load some critical values for easy readable access
        self.num_planes = len(self.b2_registration.get_value("planeNames"))
        self.um_per_pixel = um_per_pixel  # store this for generating correct axes and measuring distances

        self.data_loaded = False  # initialize to false in case data isn't loaded
        if autoload:
            self.load_reference_and_masks()  # prepare reference images and ROI mask data

    # ------------------------------
    # -- initialization functions --
    # ------------------------------
    def load_reference_and_masks(self):
        # load reference images
        ops = self.b2_registration.load_s2p("ops")
        self.reference = [op["meanImg_chan2"] for op in ops]
        self.lx, self.ly = self.reference[0].shape
        for ref in self.reference:
            msg = "reference images do not all have the same shape"
            assert (self.lx, self.ly) == ref.shape, msg

        # load masks (lam=weight of each pixel, xpix & ypix=index of each pixel in ROI mask)
        stat = self.b2_registration.load_s2p("stat")
        self.lam = [s["lam"] for s in stat]
        self.ypix = [s["ypix"] for s in stat]
        self.xpix = [s["xpix"] for s in stat]
        self.roi_plane_idx = self.b2_registration.loadone("mpciROIs.stackPosition")[:, 2]

        # load S2P red cell value
        self.red_s2p = self.b2_registration.loadone("mpciROIs.redS2P")  # (preloaded, will never change in this function)

        # create supporting variables for mapping locations and axes
        self.y_base_ref = np.arange(self.ly)
        self.x_base_ref = np.arange(self.lx)
        self.y_dist_ref = self.create_centered_axis(self.ly, self.um_per_pixel)
        self.x_dist_ref = self.create_centered_axis(self.lx, self.um_per_pixel)

        # update data_loaded field
        self.data_loaded = True

    # ---------------------------------
    # -- updating one data functions --
    # ---------------------------------
    def one_name_feature_cutoffs(self, name):
        """standard method for naming the features used to define redCellIdx cutoffs"""
        return "parameters" + "Red" + name[0].upper() + name[1:] + ".minMaxCutoff"

    def update_red_idx(self, s2p_cutoff=None, dot_product_cutoff=None, corr_coef_cutoff=None, phase_corr_cutoff=None):
        """method for updating the red index given new cutoff values"""
        # create initial all true red cell idx
        red_cell_idx = np.full(self.b2_registration.loadone("mpciROIs.redCellIdx").shape, True)

        # load feature values for each ROI
        red_s2p = self.b2_registration.loadone("mpciROIs.redS2P")
        dot_product = self.b2_registration.loadone("mpciROIs.redDotProduct")
        corr_coef = self.b2_registration.loadone("mpciROIs.redPearson")
        phase_corr = self.b2_registration.loadone("mpciROIs.redPhaseCorrelation")

        # create lists for zipping through each feature/cutoff combination
        features = [red_s2p, dot_product, corr_coef, phase_corr]
        cutoffs = [s2p_cutoff, dot_product_cutoff, corr_coef_cutoff, phase_corr_cutoff]
        usecutoff = [[False, False] for _ in range(len(cutoffs))]

        # check validity of each cutoff and identify whether it should be used
        for name, use, cutoff in zip(self.feature_names, usecutoff, cutoffs):
            if not isinstance(cutoff, np.ndarray) and not isinstance(cutoff, list):
                raise ValueError(f"Expecting a numpy array or a list for {name} cutoff, got {type(cutoff)}")
            assert len(cutoff) == 2, f"{name} cutoff does not have 2 elements"
            if not (np.isnan(cutoff[0])):
                use[0] = True
            if not (np.isnan(cutoff[1])):
                use[1] = True

        # add feature cutoffs to redCellIdx (sets any to False that don't meet the cutoff)
        for feature, use, cutoff in zip(features, usecutoff, cutoffs):
            if use[0]:
                red_cell_idx &= feature >= cutoff[0]
            if use[1]:
                red_cell_idx &= feature <= cutoff[1]

        # save new red cell index to one data
        self.b2_registration.saveone(red_cell_idx, "mpciROIs.redCellIdx")

        # save feature cutoffs to one data
        for idx, name in enumerate(self.feature_names):
            self.b2_registration.saveone(cutoffs[idx], self.one_name_feature_cutoffs(name))
        print(f"Red Cell curation choices are saved for session {self.b2_registration.session_print()}")

    def update_from_session(self, red_cell, force_update=False):
        """method for updating the red cell cutoffs from another session"""
        assert isinstance(red_cell, RedCellProcessing), "red_cell is not a RedCellProcessing object"
        if not (force_update):
            assert (
                red_cell.b2_registration.mouse_name == self.b2_registration.mouse_name
            ), "session to copy from is from a different mouse, this isn't allowed without the force_update=True input"
        cutoffs = [red_cell.b2_registration.loadone(red_cell.one_name_feature_cutoffs(name)) for name in self.feature_names]
        self.update_red_idx(s2p_cutoff=cutoffs[0], dot_product_cutoff=cutoffs[1], corr_coef_cutoff=cutoffs[2], phase_corr_cutoff=cutoffs[3])

    def cropped_phase_correlation(self, plane_idx=None, width=40, eps=1e6, winFunc=lambda x: np.hamming(x.shape[-1])):
        """
        This returns the phase correlation of each (cropped) mask with the (cropped) reference image.
        The default parameters (width=40um, eps=1e6, and a hamming window function) were tested on a few sessions and is purely subjective.
        I recommend that if you use this function to determine which of your cells are red, you do manual curation and potentially update some of these parameters.
        """
        if not (self.data_loaded):
            self.load_reference_and_masks()
        if winFunc == "hamming":
            winFunc = lambda x: np.hamming(x.shape[-1])
        refStack = self.centered_reference_stack(plane_idx=plane_idx, width=width)  # get stack of reference image centered on each ROI
        maskStack = self.centered_mask_stack(plane_idx=plane_idx, width=width)  # get stack of mask value centered on each ROI
        window = winFunc(refStack)  # create a window function
        pxcStack = np.stack(
            [helpers.phaseCorrelation(ref, mask, eps=eps, window=window) for (ref, mask) in zip(refStack, maskStack)]
        )  # measure phase correlation
        pxcCenterPixel = int((pxcStack.shape[2] - 1) / 2)
        return refStack, maskStack, pxcStack, pxcStack[:, pxcCenterPixel, pxcCenterPixel]

    def compute_dot(self, plane_idx=None, lowcut=12, highcut=250, order=3, fs=512):
        if plane_idx is None:
            plane_idx = np.arange(self.num_planes)
        if isinstance(plane_idx, (int, np.integer)):
            plane_idx = (plane_idx,)  # make plane_idx iterable
        if not (self.data_loaded):
            self.load_reference_and_masks()

        dot_prod = []
        for plane in plane_idx:
            t = time.time()
            c_roi_idx = np.where(self.roi_plane_idx == plane)[0]  # index of ROIs in this plane
            bwReference = helpers.butterworthbpf(self.reference[plane], lowcut, highcut, order=order, fs=fs)  # filtered reference image
            bwReference /= np.linalg.norm(bwReference)  # adjust to norm for straightforward cosine angle
            # compute normalized dot product for each ROI
            dot_prod.append(
                np.array([bwReference[self.ypix[roi], self.xpix[roi]] @ self.lam[roi] / np.linalg.norm(self.lam[roi]) for roi in c_roi_idx])
            )

        return np.concatenate(dot_prod)

    def compute_corr(self, plane_idx=None, width=20, lowcut=12, highcut=250, order=3, fs=512):
        if plane_idx is None:
            plane_idx = np.arange(self.num_planes)
        if isinstance(plane_idx, (int, np.integer)):
            plane_idx = (plane_idx,)  # make plane_idx iterable
        if not (self.data_loaded):
            self.load_reference_and_masks()

        corr_coef = []
        for plane in plane_idx:
            num_rois = self.b2_registration.get_value("roiPerPlane")[plane]
            c_ref_stack = np.reshape(
                self.centered_reference_stack(plane_idx=plane, width=width, fill=np.nan, filtPrms=(lowcut, highcut, order, fs)),
                (num_rois, -1),
            )
            c_mask_stack = np.reshape(self.centered_mask_stack(plane_idx=plane, width=width, fill=0), (num_rois, -1))
            c_mask_stack[np.isnan(c_ref_stack)] = np.nan

            # Measure mean and standard deviation (and number of non-nan datapoints)
            u_ref = np.nanmean(c_ref_stack, axis=1, keepdims=True)
            u_mask = np.nanmean(c_mask_stack, axis=1, keepdims=True)
            s_ref = np.nanstd(c_ref_stack, axis=1)
            s_mask = np.nanstd(c_mask_stack, axis=1)
            N = np.sum(~np.isnan(c_ref_stack), axis=1)

            # compute correlation coefficient and add to storage variable
            corr_coef.append(np.nansum((c_ref_stack - u_ref) * (c_mask_stack - u_mask), axis=1) / N / s_ref / s_mask)

        return np.concatenate(corr_coef)

    # --------------------------
    # -- supporting functions --
    # --------------------------
    def create_centered_axis(self, numElements, scale=1):
        return scale * (np.arange(numElements) - (numElements - 1) / 2)

    def getyref(self, yCenter):
        if not (self.data_loaded):
            self.load_reference_and_masks()
        return self.um_per_pixel * (self.y_base_ref - yCenter)

    def getxref(self, xCenter):
        if not (self.data_loaded):
            self.load_reference_and_masks()
        return self.um_per_pixel * (self.x_base_ref - xCenter)

    def get_roi_centroid(self, idx, mode="weightedmean"):
        if not (self.data_loaded):
            self.load_reference_and_masks()

        if mode == "weightedmean":
            yc = np.sum(self.lam[idx] * self.ypix[idx]) / np.sum(self.lam[idx])
            xc = np.sum(self.lam[idx] * self.xpix[idx]) / np.sum(self.lam[idx])
        elif mode == "median":
            yc = int(np.median(self.ypix[idx]))
            xc = int(np.median(self.xpix[idx]))

        return yc, xc

    def get_roi_range(self, idx):
        if not (self.data_loaded):
            self.load_reference_and_masks()
        # get range of x and y pixels for a particular ROI
        yr = np.ptp(self.ypix[idx])
        xr = np.ptp(self.xpix[idx])
        return yr, xr

    def get_roi_in_plane_idx(self, idx):
        if not (self.data_loaded):
            self.load_reference_and_masks()
        # return index of ROI within it's own plane
        plane_idx = self.roi_plane_idx[idx]
        return idx - np.sum(self.roi_plane_idx < plane_idx)

    def centered_reference_stack(self, plane_idx=None, width=15, fill=0.0, filtPrms=None):
        # return stack of reference images centered on each ROI (+/- width um around ROI centroid)
        # if plane_idx is none, then returns across all planes
        # fill determines what value to use as the background (should either be 0 or nan...)
        # if filterPrms=None, then just returns centered reference stack. otherwise, filterPrms requires a tuple of 4 parameters which define a butterworth filter
        if plane_idx is None:
            plane_idx = np.arange(self.num_planes)
        if isinstance(plane_idx, (int, np.integer)):
            plane_idx = (plane_idx,)  # make plane_idx iterable
        if not (self.data_loaded):
            self.load_reference_and_masks()
        num_pixels = int(np.round(width / self.um_per_pixel))  # numPixels to each side around the centroid
        ref_stack = []
        for plane in plane_idx:
            c_reference = self.reference[plane]
            if filtPrms is not None:
                # filtered reference image
                c_reference = helpers.butterworthbpf(c_reference, filtPrms[0], filtPrms[1], order=filtPrms[2], fs=filtPrms[3])
            idx_roi_in_plane = np.where(self.roi_plane_idx == plane)[0]
            ref_stack.append(np.full((len(idx_roi_in_plane), 2 * num_pixels + 1, 2 * num_pixels + 1), fill))
            # fill the reference stack with the reference image
            for idx, idx_roi in enumerate(idx_roi_in_plane):
                yc, xc = self.get_roi_centroid(idx_roi, mode="median")
                yUse = (np.maximum(yc - num_pixels, 0), np.minimum(yc + num_pixels + 1, self.ly))
                xUse = (np.maximum(xc - num_pixels, 0), np.minimum(xc + num_pixels + 1, self.lx))
                yMissing = (
                    -np.minimum(yc - num_pixels, 0),
                    -np.minimum(self.ly - (yc + num_pixels + 1), 0),
                )
                xMissing = (
                    -np.minimum(xc - num_pixels, 0),
                    -np.minimum(self.lx - (xc + num_pixels + 1), 0),
                )
                ref_stack[-1][
                    idx,
                    yMissing[0] : 2 * num_pixels + 1 - yMissing[1],
                    xMissing[0] : 2 * num_pixels + 1 - xMissing[1],
                ] = c_reference[yUse[0] : yUse[1], xUse[0] : xUse[1]]
        return np.concatenate(ref_stack, axis=0).astype(np.float32)

    def centered_mask_stack(self, plane_idx=None, width=15, fill=0.0):
        # return stack of ROI Masks centered on each ROI (+/- width um around ROI centroid)
        # if plane_idx is none, then returns across all planes
        # fill determines what value to use as the background (should either be 0 or nan)
        if plane_idx is None:
            plane_idx = np.arange(self.num_planes)
        if isinstance(plane_idx, (int, np.integer)):
            plane_idx = (plane_idx,)  # make plane_idx iterable
        if not (self.data_loaded):
            self.load_reference_and_masks()
        num_pixels = int(np.round(width / self.um_per_pixel))  # numPixels to each side around the centroid
        mask_stack = []
        for plane in plane_idx:
            idx_roi_in_plane = np.where(self.roi_plane_idx == plane)[0]
            mask_stack.append(np.full((len(idx_roi_in_plane), 2 * num_pixels + 1, 2 * num_pixels + 1), fill))
            for idx, idx_roi in enumerate(idx_roi_in_plane):
                yc, xc = self.get_roi_centroid(idx_roi, mode="median")
                # centered y&x pixels of ROI
                cyidx = self.ypix[idx_roi] - yc + num_pixels
                cxidx = self.xpix[idx_roi] - xc + num_pixels
                # index of pixels still within width of stack
                idx_use_points = (cyidx >= 0) & (cyidx < 2 * num_pixels + 1) & (cxidx >= 0) & (cxidx < 2 * num_pixels + 1)
                mask_stack[-1][idx, cyidx[idx_use_points], cxidx[idx_use_points]] = self.lam[idx_roi][idx_use_points]
        return np.concatenate(mask_stack, axis=0).astype(np.float32)

    def compute_volume(self, plane_idx=None):
        if plane_idx is None:
            plane_idx = np.arange(self.num_planes)
        if isinstance(plane_idx, (int, np.integer)):
            plane_idx = (plane_idx,)  # make plane_idx iterable
        msg = f"in session: {self.b2_registration.session_print()}, there are only {self.num_planes} planes!"
        assert all([0 <= plane < self.num_planes for plane in plane_idx]), msg
        if not (self.data_loaded):
            self.load_reference_and_masks()
        roi_mask_volume = []
        for plane in plane_idx:
            roi_mask_volume.append(np.zeros((self.b2_registration.get_value("roiPerPlane")[plane], self.ly, self.lx)))
            idx_roi_in_plane = np.where(self.roi_plane_idx == plane)[0]
            for roi in range(self.b2_registration.get_value("roiPerPlane")[plane]):
                c_roi_idx = idx_roi_in_plane[roi]
                roi_mask_volume[-1][roi, self.ypix[c_roi_idx], self.xpix[c_roi_idx]] = self.lam[c_roi_idx]
        return np.concatenate(roi_mask_volume, axis=0)
