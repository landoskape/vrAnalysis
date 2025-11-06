from typing import Union, Optional, Dict

import numpy as np
import torch


class Population:
    """
    Population is a class that holds and manipulates a dataset of neurons by timepoints for use in dimensionality analyses.

    Many analyses in this library require manipulating a dataset of neurons in groups for cross-validating purposes. The primary two operations
    are splitting cells into groups and splitting the timepoints into groups (often with special considerations for intelligent cross-validation).
    """

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        generate_splits: bool = True,
        cell_split_prms: dict = {},
        time_split_prms: dict = {},
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the Population object

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            A 2D array of shape (num_neurons, num_timepoints) representing the activity of neurons over time.
        generate_splits : bool
            If True, will generate splits for cells and timepoints using any parameters in cell_split_prms and time_split_prms, respectively.
            (default is True)
        cell_split_prms : dict
            Parameters for splitting the cells into groups.
            (default is {})
        time_split_prms : dict
            Parameters for splitting the timepoints into groups.
            (default is {})
        dtype : Optional[torch.dtype]
            The data type to cast the data to if it isn't already a torch tensor.
            (default is None)
        """
        # Converts to a torch tensor if necessary and throws an error if it can't
        data = self._check_datatype(data)

        if len(data.shape) != 2:
            raise ValueError("data must be a 2D array/tensor with shape (num_neurons, num_timepoints)")

        self.data = data
        self.num_neurons, self.num_timepoints = data.shape
        self.dtype = dtype

        if generate_splits:
            # remove return_indices from the parameters to avoid returning the indices (force storing as attributes when called in constructor method)
            cell_split_prms.pop("return_indices", None)
            time_split_prms.pop("return_indices", None)
            self.split_cells(**cell_split_prms)
            self.split_times(**time_split_prms)

    def get_split_data(
        self,
        time_idx: Optional[Union[int, list[int]]] = None,
        center: bool = False,
        scale: bool = False,
        pre_split: bool = False,
        scale_type: Optional[str] = None,
    ):
        """
        Get the source and target data for a specific set of timepoints.

        Parameters
        ----------
        time_idx : Optional[Union[int, list[int]]]
            The time group(s) to use as the target data. If a list of integers, will concatenate the data for the specified time groups.
            If None, will use all timepoints in the data.
            (default is None)
        center : bool
            If True, will center the data so each neuron has a mean of 0 across timepoints
            (default is False)
        scale : bool
            If True, will scale the data so each neuron has a standard deviation of 1 across timepoints
            (default is False)
        pre_split : bool
            If True, will center and scale the data before splitting into source and target data
        scale_type : Optional[str]
            How to scale the data.
            =[None, 'std'] -> will scale the data by the standard deviation of the source data.
            ='sqrt' -> will scale the data by the square root of the standard deviation of the source data.
            ='preserve' -> will scale the data such that the neuron with median std will end up with a std of 1,
            and the other neurons will be scaled accordingly (preserving the relative standard deviation).

        Returns
        -------
        torch.Tensor
            The source data for the specified set of timepoints.
        torch.Tensor
            The target data for the specified set of timepoints.
        """
        if not hasattr(self, "cell_split_indices"):
            raise ValueError("cell_split_indices must be set before calling get_source_target")

        if not hasattr(self, "time_split_indices"):
            raise ValueError("time_split_indices must be set before calling get_source_target")

        source = self.apply_split(self.data[self.cell_split_indices[0]], time_idx, center, scale, pre_split, scale_type)
        target = self.apply_split(self.data[self.cell_split_indices[1]], time_idx, center, scale, pre_split, scale_type)
        return source, target

    def apply_split(
        self,
        data: torch.Tensor,
        time_idx: Optional[Union[int, list[int]]] = None,
        center: bool = False,
        scale: bool = False,
        pre_split: bool = False,
        scale_type: Optional[str] = None,
    ):
        """
        Apply the time splits to a new dataset. If time_idx is a list of integers, will concatenate the data for the specified time groups.

        Parameters
        ----------
        data : torch.Tensor
            The data to apply the time splits to. Must have shape (num_features, num_timepoints), where
            num_features is unconstrained and num_timepoints must match the number of timepoints in the
            Population instance.
        time_idx : Optional[Union[int, list[int]]]
            The time group(s) to use as the target data. If a list of integers, will concatenate the data for the specified time groups.
            If None, will use all timepoints in the data.
            (default is None)
            If None, will use all timepoints in the data.
            (default is 0)
        center : bool
            If True, will center the data so each neuron has a mean of 0 across timepoints
            (default is False)
        scale : bool
            If True, will scale the data so each neuron has a standard deviation of 1 across timepoints
            (default is False)
        pre_split : bool
            If True, will center and scale the data before splitting into source and target data
        scale_type : Optional[str]
            How to scale the data.
            =[None, 'std'] -> will scale the data by the standard deviation of the source data.
            ='sqrt' -> will scale the data by the square root of the standard deviation of the source data.
            ='preserve' -> will scale the data such that the neuron with median std will end up with a std of 1,
            and the other neurons will be scaled accordingly (preserving the relative standard deviation).

        Returns
        -------
        torch.Tensor
            The data for the specified set of timepoints.
        """
        if not hasattr(self, "time_split_indices"):
            raise ValueError("time_split_indices must be set before calling get_source_target")

        if isinstance(time_idx, int):
            assert time_idx < len(self.time_split_indices), "time_idx must correspond to one of the time groups in time_split_indices"
            time_idx = [time_idx]
        elif isinstance(time_idx, list):
            assert all(isinstance(idx, int) for idx in time_idx), "time_idx must be a list of integers"
            assert all(
                idx < len(self.time_split_indices) for idx in time_idx
            ), "time_idx must correspond to one of the time groups in time_split_indices"
        else:
            if not time_idx is None:
                raise ValueError("time_idx must be an integer or None")

        # Convert data to torch tensor if necessary and throw an error if it can't
        data = self._check_datatype(data)
        assert data.size(1) == self.num_timepoints, "data must have the same number of timepoints as the Population instance"

        if pre_split:
            if center:
                mean = data.mean(dim=1, keepdim=True)
            if scale:
                std = data.std(dim=1, keepdim=True)
                std[std == 0] = 1

        # Select the timepoints for the specified group when time_idx is an integer
        if time_idx is not None:
            _split_data = [data[:, self.time_split_indices[idx]] for idx in time_idx]
            data = torch.cat(_split_data, dim=1)

        if center:
            if pre_split:
                data = data - mean
            else:
                data = data - data.mean(dim=1, keepdim=True)

        if scale:
            # prepare source / target standard deviation if not pre-computed
            if not pre_split:
                std = data.std(dim=1, keepdim=True)
                std[std == 0] = 1

            # normalize source and target appropriately
            if scale_type is None or scale_type == "std":
                data = data / std
            elif scale_type == "sqrt":
                data = data / torch.sqrt(std)
            elif scale_type == "preserve":
                data = data / std.median()
            else:
                raise ValueError(f"scale_type must be one of [None, 'std', 'sqrt', 'preserve'], got {scale_type}")

        if self.dtype is not None:
            data = data.to(self.dtype)

        return data

    def split_cells(self, force_even: bool = False, return_indices: bool = False):
        """
        Assign indices to each neurons to split into two groups.

        TODO: Add method diversion to split neurons into groups based on additional criteria (e.g. location chunks)

        Parameters
        ----------
        force_even : bool
            If True, will ensure that the number of neurons in each group is equal to each other.
            If number of neurons isn't divisible by number of groups, will clip neurons randomly.
            (default is False)
        return_indices : bool
            If True, will return the indices of the split neurons instead of storing them
            (default is False)

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_neurons, ) representing the group assignment of each neuron
            Only returned if return_indices is True.
        """
        index = torch.randperm(self.num_neurons)

        # Clip neurons to be divisible by num_groups if requested by user (use randomly clipped neuron(s))
        if force_even:
            num_per_group = self.num_neurons // 2
            total_neurons = num_per_group * 2
            index = index[:total_neurons]

        cell_split_indices = torch.tensor_split(index, 2)

        if return_indices:
            return cell_split_indices

        self.cell_split_indices = cell_split_indices

    def split_times(
        self,
        num_groups: int = 2,
        relative_size: Optional[list[int]] = None,
        chunks_per_group: int = 5,
        num_buffer: int = 10,
        force_even: bool = False,
        return_indices: bool = False,
    ):
        """
        Assign indices to each timepoint to split into groups.

        Note: if force_even is set to True, then the size of each group will be equal to each other (subject
        to integer scaling by relative size). However, if it isn't, then the groups will be split based on
        relative size according to the number of chunks going into each group and the chunks may be uneven
        so the relative size may not be exactly equal to the integer ratio in relative size.

        Parameters
        ----------
        num_groups : int
            Number of groups to split the timepoints into.
            (default is 2)
        relative_size : Optional[list[int]]
            A list of integers representing the relative size of each group.
            If provided, will split timepoints into groups based on the relative size.
            (default is None)
        chunks_per_group : int
            Number of chunks for each group (subject to scaling by relative size)
            > if positive, will make chunks as big as possible given the other parameters such that the
            total number of chunks match the request.
            > if negative, will instead make chunks have the number of samples corresponding to the number provided.
            (default is 5)
        num_buffer : int
            Number of buffer timepoints between each chunk.
            (default is 10)
        force_even : bool
            If True, will ensure that the number of timepoints in each group is equal to each other.
            If number of timepoints isn't divisible by number of groups, will clip timepoints of later groups.
            (default is False)
        return_indices : bool
            If True, will return the indices of the split timepoints instead of storing them
            (default is False)
        """
        if relative_size is None:
            relative_size = [1] * num_groups
        else:
            assert len(relative_size) == num_groups, "relative_size must have the same length as num_groups"
            assert all(isinstance(size, int) for size in relative_size), "relative_size must be a list of integers"

        indices = torch.arange(self.num_timepoints)
        time_chunks = self._chunk_indices(indices, chunks_per_group * sum(relative_size), num_buffer, force_even=force_even)

        # randomize order of chunks
        time_chunks = [time_chunks[i] for i in torch.randperm(len(time_chunks))]

        if chunks_per_group < 0:
            # if chunks_per_group is negative, then the number of chunks is fixed and the chunks are the size of the number provided
            # this can cause uneven group sizes -- so we just clip whatever chunks are leftover
            num_chunks = len(time_chunks)
            chunks_per_group = num_chunks // sum(relative_size)

        # consolidate chunks into groups
        start_stop_index = torch.cumsum(torch.tensor([0] + [rs * chunks_per_group for rs in relative_size]), dim=0)
        time_split_indices = []
        for i in range(num_groups):
            time_split_indices.append(torch.cat(time_chunks[start_stop_index[i] : start_stop_index[i + 1]]))

        if return_indices:
            return time_split_indices

        self.time_split_indices = time_split_indices

    def _chunk_indices(self, indices, num_chunks, num_buffer, force_even: bool = False):
        """
        Chunks indices into num_chunks with a buffer of num_buffer between chunks.

        Parameters
        ----------
        indices : torch.Tensor
            A tensor of indices to split into chunks.
        num_chunks : int
            Number of chunks to split the indices into.
            > If positive, will make chunks as big as possible given the other parameters such that the
            total number of chunks match the request.
            > If negative, will instead make chunks have the number of samples corresponding to the number provided.
        num_buffer : int
            Number of buffer indices between each chunk.
        force_even : bool
            If True, will ensure that the number of indices in each chunk is equal to each other.
            If number of indices isn't divisible by number of chunks, will clip indices of later chunks.
            (default is False)

        Returns
        -------
        List[torch.Tensor]
            A list of tensors representing the chunked indices.
        """
        assert num_buffer >= 0, "num_buffer must be greater than or equal to 0"
        assert num_chunks != 0 and isinstance(num_chunks, int), "num_chunks must be a non-zero integer"

        if num_chunks > 0:
            num_buffered_samples = num_buffer * (num_chunks - 1)

            # Clip indices to be divisible by num_chunks if requested by user (clip the later indices)
            # drop_buffer is the number of additional indices that have to be dropped when even chunks are requested
            if force_even:
                each_chunk_size = (len(indices) - num_buffered_samples) // num_chunks
                chunk_sizes = [each_chunk_size] * num_chunks
                drop_buffer = len(indices) - (sum(chunk_sizes) + num_buffered_samples)
            else:
                num_chunked_indices = len(indices) - num_buffered_samples
                minimum_chunk_size = num_chunked_indices // num_chunks
                num_remainder = num_chunked_indices - minimum_chunk_size * num_chunks
                chunk_sizes = [minimum_chunk_size + 1 * (i < num_remainder) for i in range(num_chunks)]
                drop_buffer = 0

        else:
            # All chunks are the same size
            number_of_chunks = len(indices) // (-num_chunks + num_buffer)
            chunk_sizes = [-num_chunks] * number_of_chunks
            samples_in_chunks = len(chunk_sizes) * -num_chunks
            drop_buffer = len(indices) - samples_in_chunks - (len(chunk_sizes) - 1) * num_buffer
            num_chunks = len(chunk_sizes)

        # Assign the size to each group (chunk / buffer / chunk / buffer / ... / chunk / drop_buffer)
        bin_size = []
        for ichunk, chunk_size in enumerate(chunk_sizes):
            bin_size.append(chunk_size)
            if ichunk < (num_chunks - 1):
                bin_size.append(num_buffer)
            else:
                bin_size.append(drop_buffer)

        bin_indices = torch.split(indices, bin_size)
        chunk_indices = bin_indices[::2]
        return chunk_indices

    def _check_datatype(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a numpy array or torch tensor")

        return data

    def size(self, dim=None):
        """Get the size of the population activity (works like torch.size on self.data)"""
        return self.data.size(dim)

    def get_indices_dict(self) -> Dict:
        """
        Get a dictionary containing the split indices and metadata.

        Returns
        -------
        Dict
            A dictionary containing cell_split_indices, time_split_indices, num_neurons, and num_timepoints.
        """
        return {
            "cell_split_indices": [indices.tolist() for indices in self.cell_split_indices] if hasattr(self, "cell_split_indices") else None,
            "time_split_indices": [indices.tolist() for indices in self.time_split_indices] if hasattr(self, "time_split_indices") else None,
            "num_neurons": self.num_neurons,
            "num_timepoints": self.num_timepoints,
            "dtype": self.dtype,
        }

    @classmethod
    def make_from_indices(cls, indices_dict: Dict, data: Union[np.ndarray, torch.Tensor]):
        """
        Make a new Population instance with the provided split indices and data.

        Parameters
        ----------
        indices_dict : Dict
            The dictionary containing the split indices and metadata.
        data : Union[np.ndarray, torch.Tensor]
            The data to use for the new Population instance.

        Returns
        -------
        Population
            A new Population instance with the loaded split indices.

        Raises
        ------
        ValueError
            If the shape of the provided data doesn't match the saved indices.
        """
        # Check if the shape of the provided data matches the saved indices
        if data.shape != (indices_dict["num_neurons"], indices_dict["num_timepoints"]):
            raise ValueError(
                f"Shape of provided data {data.shape} doesn't match the shape in saved indices "
                f"({indices_dict['num_neurons']}, {indices_dict['num_timepoints']})"
            )

        population = cls(data, generate_splits=False, dtype=indices_dict.get("dtype", None))

        if indices_dict["cell_split_indices"]:
            population.cell_split_indices = [torch.tensor(indices) for indices in indices_dict["cell_split_indices"]]

        if indices_dict["time_split_indices"]:
            population.time_split_indices = [torch.tensor(indices) for indices in indices_dict["time_split_indices"]]

        return population


class SourceTarget(Population):
    """
    SourceTarget is a class inheriting from the Population class where there is a natural division between source and target.
    The purpose is to reuse the Population class for generating time splits.
    """

    def __init__(
        self,
        source: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        generate_splits: bool = True,
        time_split_prms={},
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the SourceTarget object

        Parameters
        ----------
        source : Union[np.ndarray, torch.Tensor]
            A 2D array of shape (num_source_features, num_timepoints) representing variable used to make predictions.
        target : Union[np.ndarray, torch.Tensor]
            A 2D array of shape (num_target_features, num_timepoints) representing variable to predict.
        generate_splits : bool
            If True, will generate splits for timepoints using any parameters in time_split_prms.
            (default is True)
        time_split_prms : dict
            Parameters for splitting the timepoints into groups.
            (default is {})
        dtype : Optional[torch.dtype]
            The data type to cast the data to if it isn't already a torch tensor.
            (default is None)
        """
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source)

        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        if len(source.shape) != 2:
            raise ValueError("source must be a 2D array/tensor with shape (num_source_features, num_timepoints)")

        if len(target.shape) != 2:
            raise ValueError("target must be a 2D array/tensor with shape (num_target_features, num_timepoints)")

        data = torch.cat([source, target], dim=0)
        self.num_source_features = source.shape[0]
        self.num_target_features = target.shape[0]

        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a numpy array or torch tensor")

        self.data = data
        num_features, self.num_timepoints = data.shape
        self.dtype = dtype

        # Split the "cells" into source and target groups (this is hard-coded by this class)
        feature_index = torch.arange(num_features)
        self.cell_split_indices = [feature_index[: self.num_source_features], feature_index[self.num_source_features :]]

        if generate_splits:
            # remove return_indices from the parameters to avoid returning the indices (force storing as attributes when called in constructor method)
            time_split_prms.pop("return_indices", None)
            self.split_times(**time_split_prms)

    def size(self):
        """Get the size of the population activity (works like torch.size on self.data)"""
        raise NotImplementedError(
            "SourceTarget does not have a size method. Use num_source_features, num_target_features, or num_timepoints instead."
        )

    def get_indices_dict(self) -> Dict:
        """
        Get a dictionary containing the split indices and metadata.

        Returns
        -------
        Dict
            A dictionary containing cell_split_indices, time_split_indices, num_neurons, and num_timepoints.
        """
        return {
            "cell_split_indices": [indices.tolist() for indices in self.cell_split_indices],
            "time_split_indices": [indices.tolist() for indices in self.time_split_indices] if hasattr(self, "time_split_indices") else None,
            "num_source_features": self.num_source_features,
            "num_target_features": self.num_target_features,
            "num_timepoints": self.num_timepoints,
            "dtype": self.dtype,
        }

    @classmethod
    def make_from_indices(cls, indices_dict: Dict, source: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]):
        """
        Make a new SourceTarget instance with the provided split indices and data.

        Parameters
        ----------
        indices_dict : Dict
            The dictionary containing the split indices and metadata.
        source : Union[np.ndarray, torch.Tensor]
            The source data to use for the new SourceTarget instance.
        target : Union[np.ndarray, torch.Tensor]
            The target data to use for the new SourceTarget instance.

        Returns
        -------
        SourceTarget
            A new SourceTarget instance with the loaded split indices.

        Raises
        ------
        ValueError
            If the shape of the provided data doesn't match the saved indices.
        """
        # Check if the shape of the provided data matches the saved indices
        if source.shape != (indices_dict["num_source_features"], indices_dict["num_timepoints"]):
            raise ValueError(
                f"Shape of provided source data {source.shape} doesn't match the shape in saved indices "
                f"({indices_dict['num_source_features']}, {indices_dict['num_timepoints']})"
            )
        if target.shape != (indices_dict["num_target_features"], indices_dict["num_timepoints"]):
            raise ValueError(
                f"Shape of provided target data {target.shape} doesn't match the shape in saved indices "
                f"({indices_dict['num_target_features']}, {indices_dict['num_timepoints']})"
            )

        source_target = cls(source, target, generate_splits=False, dtype=indices_dict.get("dtype", None))

        if indices_dict["cell_split_indices"]:
            source_target.cell_split_indices = [torch.tensor(indices) for indices in indices_dict["cell_split_indices"]]

        if indices_dict["time_split_indices"]:
            source_target.time_split_indices = [torch.tensor(indices) for indices in indices_dict["time_split_indices"]]

        return source_target
