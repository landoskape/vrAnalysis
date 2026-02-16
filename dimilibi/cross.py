from typing import Optional
import torch
from sklearn.metrics import r2_score

from .pca import PCA
from .svca import SVCA


class CrossCompare:
    def __init__(
        self,
        num_components: Optional[int] = None,
        verbose: bool = False,
        center: bool = True,
    ):
        """
        Initialize the CrossCompare model.

        Parameters
        ----------
        num_components : Optional[int]
            The number of components to use in the PCA and SVCA models (default is None).
        verbose : bool
            If True, will print updates and results as they are computed (default is False).
        center : bool
            If True, center the data before PCA. Default is True.
        """
        self.num_components = num_components
        self.verbose = verbose
        self.center = center

    def fit(self, source: torch.Tensor, target: torch.Tensor):
        """
        Fit the CrossCompare model to the provided data.

        Parameters
        ----------
        source : torch.Tensor
            The source data to be used for training (num_neurons_source, num_timepoints).
        target : torch.Tensor
            The target data to be used for training (num_neurons_target, num_timepoints).

        Returns
        -------
        self : object
            The CrossCompare object with the fitted model.
        """
        self.pca_source = PCA(num_components=self.num_components, verbose=self.verbose, center=self.center).fit(source)
        self.pca_target = PCA(num_components=self.num_components, verbose=self.verbose, center=self.center).fit(target)
        self.svca = SVCA(num_components=self.num_components, centered=self.center, verbose=self.verbose).fit(source, target)

        # build a map from the SVCA modes to the PCA components
        self.pc_to_u = torch.linalg.lstsq(self.pca_source.get_components(), self.svca.U)[0]
        self.pc_to_v = torch.linalg.lstsq(self.pca_target.get_components(), self.svca.V)[0]

        self.u_to_pc = torch.linalg.lstsq(self.svca.U, self.pca_source.get_components())[0]
        self.v_to_pc = torch.linalg.lstsq(self.svca.V, self.pca_target.get_components())[0]

        return self

    def score(self, to_pca: bool = False):
        """
        Score the CrossCompare model on the provided data by comparing the PCA components to the SVCA projections.

        Parameters
        ----------
        to_pca : bool
            If True, will compare the SVCA projections to the PCA components (default is False).

        Returns
        -------
        r2_source : float
            The R^2 score of the (SVCA/PCA) components compared to the projection from (PCA/SVCA) for the source data.
        r2_target : float
            The R^2 score of the (SVCA/PCA) components compared to the projection from (PCA/SVCA) for the target data.
        """
        if to_pca:
            u_proj = self.svca.U @ self.u_to_pc
            v_proj = self.svca.V @ self.v_to_pc
            u_r2 = r2_score(self.pca_source.get_components(), u_proj)
            v_r2 = r2_score(self.pca_target.get_components(), v_proj)
            return u_r2, v_r2
        else:
            u_proj = self.pca_source.get_components() @ self.pc_to_u
            v_proj = self.pca_target.get_components() @ self.pc_to_v
            u_r2 = r2_score(self.svca.U, u_proj)
            v_r2 = r2_score(self.svca.V, v_proj)
            return u_r2, v_r2

    def analyze(self, to_pca: bool = False):
        """
        Analyze the CrossCompare model by measuring the center of mass and entropy of the PCA to SVCA map.

        Parameters
        ----------
        to_pca : bool
            If True, will analyze the SVCA to PCA map instead of PCA -> SVCA (default is False).

        Returns
        -------
        source_com : torch.Tensor
            The center of mass of the (PCA/SVCA) to (SVCA/PCA) map for the source data.
        target_com : torch.Tensor
            The center of mass of the (PCA/SVCA) to (SVCA/PCA) map for the target data.
        source_entropy : torch.Tensor
            The entropy of the (PCA/SVCA) to (SVCA/PCA) map for the source data.
        target_entropy : torch.Tensor
            The entropy of the (PCA/SVCA) to (SVCA/PCA) map for the target data.
        """
        if to_pca:
            source_com = self._center_of_mass(torch.abs(self.u_to_pc))
            target_com = self._center_of_mass(torch.abs(self.v_to_pc))
            source_entropy = self._entropy(torch.abs(self.u_to_pc))
            target_entropy = self._entropy(torch.abs(self.v_to_pc))
        else:
            source_com = self._center_of_mass(torch.abs(self.pc_to_u))
            target_com = self._center_of_mass(torch.abs(self.pc_to_v))
            source_entropy = self._entropy(torch.abs(self.pc_to_u))
            target_entropy = self._entropy(torch.abs(self.pc_to_v))
        return source_com, target_com, source_entropy, target_entropy

    def _center_of_mass(self, data: torch.Tensor):
        """
        Calculate the center of mass of the provided data on the 0th dimension.

        Parameters
        ----------
        data : torch.Tensor
            The data to calculate the center of mass of.

        Returns
        -------
        com : torch.Tensor
            The center of mass of the data.
        """
        assert not torch.any(data < 0), "Data must be non-negative to compute the center of mass."
        nd = data.size(0)
        idx = torch.arange(nd).unsqueeze(1)
        return torch.sum(data * idx, dim=0) / torch.sum(data, dim=0)

    def _entropy(self, data: torch.Tensor, base=2, atol: float = 1e-15):
        """
        Calculate the entropy of the provided data on the 0th dimension.

        Parameters
        ----------
        data : torch.Tensor
            The data to calculate the entropy of.
        base : int
            The base of the logarithm for the entropy calculation (default is 2).
        atol : float
            The tolerance for zero values in the data.

        Returns
        -------
        entropy : torch.Tensor
            The entropy of the data.
        """
        if base == 2:
            log_method = torch.log2
        elif base == "nat":
            log_method = torch.log
        elif base == 10:
            log_method = torch.log10
        else:
            assert base != 1 and base > 0, "base must be positive and not 1"
            log_method = lambda x: torch.log(x) / torch.log(torch.tensor(base, dtype=data.dtype))

        # log will produce nans for zero values
        idx_nonzero = data > atol
        log_nonzero_data = torch.zeros(data.size(), dtype=data.dtype)
        log_nonzero_data[idx_nonzero] = log_method(data[idx_nonzero])

        return -torch.sum(data * log_nonzero_data, dim=0)
