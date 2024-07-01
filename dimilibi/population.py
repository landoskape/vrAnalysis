from typing import Union, Optional, Dict

import numpy as np
import torch


class Population:
    """
    Population is a class that holds and manipulates a dataset of neurons by timepoints for use in dimensionality analyses.

    Many analyses in this library require manipulating a dataset of neurons in groups for cross-validating purposes. The primary two operations
    are splitting cells into groups and splitting the timepoints into groups (often with special considerations for intelligent cross-validation).
    """

    def __init__(self, data: Union[np.ndarray, torch.Tensor], generate_splits: bool = True, cell_split_prms={}, time_split_prms={}):
        """
        Initialize the Population object

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            A 2D array of shape (num_neurons, num_timepoints) representing the activity of neurons over time.
        generate_splits : bool
            If True, will generate splits for cells and timepoints using any parameters in cell_split_prms and time_split_prms, respectively.
            (default is True)
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a numpy array or torch tensor")

        if len(data.shape) != 2:
            raise ValueError("data must be a 2D array/tensor with shape (num_neurons, num_timepoints)")

        self.data = data
        self.num_neurons, self.num_timepoints = data.shape

        if generate_splits:
            # remove return_indices from the parameters to avoid returning the indices (force storing as attributes when called in constructor method)
            cell_split_prms.pop("return_indices", None)
            time_split_prms.pop("return_indices", None)
            self.split_cells(**cell_split_prms)
            self.split_times(**time_split_prms)

    def get_split_data(self, time_idx: int = 0, center: bool = False, scale: bool = False, pre_split: bool = False):
        """
        Get the source and target data for a specific set of timepoints.

        Parameters
        ----------
        time_idx : int
            The time group to use as the target data.
            (default is 0)
        center : bool
            If True, will center the data so each neuron has a mean of 0 across timepoints
            (default is False)
        scale : bool
            If True, will scale the data so each neuron has a standard deviation of 1 across timepoints
            (default is False)
        pre_split : bool
            If True, will center and scale the data before splitting into source and target data

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

        assert time_idx < len(self.time_split_indices), "time_idx must correspond to one of the time groups in time_split_indices"

        if pre_split:
            if center: 
                source_mean = self.data[self.cell_split_indices[0]].mean(dim=1, keepdim=True)
                target_mean = self.data[self.cell_split_indices[1]].mean(dim=1, keepdim=True)
            if scale:
                source_std = self.data[self.cell_split_indices[0]].std(dim=1, keepdim=True)
                target_std = self.data[self.cell_split_indices[1]].std(dim=1, keepdim=True)
                source_std[source_std == 0] = 1
                target_std[target_std == 0] = 1

        source = self.data[self.cell_split_indices[0]][:, self.time_split_indices[time_idx]]
        target = self.data[self.cell_split_indices[1]][:, self.time_split_indices[time_idx]]

        if center:
            if pre_split:
                source = source - source_mean
                target = target - target_mean
            else:
                source = source - source.mean(dim=1, keepdim=True)
                target = target - target.mean(dim=1, keepdim=True)

        if scale:
            if pre_split: 
                source = source / source_std
                target = target / target_std
            else:
                source_std = source.std(dim=1, keepdim=True)
                target_std = target.std(dim=1, keepdim=True)
                source_std[source_std == 0] = 1
                target_std[target_std == 0] = 1
                source = source / source_std
                target = target / target_std

        return source, target

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

        num_buffered_indices = num_buffer * (num_chunks - 1)

        # Clip indices to be divisible by num_chunks if requested by user (clip the later indices)
        # drop_buffer is the number of additional indices that have to be dropped when even chunks are requested
        if force_even:
            each_chunk_size = (len(indices) - num_buffered_indices) // num_chunks
            chunk_sizes = [each_chunk_size] * num_chunks
            drop_buffer = len(indices) - (sum(chunk_sizes) + num_buffered_indices)
        else:
            num_chunked_indices = len(indices) - num_buffered_indices
            minimum_chunk_size = num_chunked_indices // num_chunks
            num_remainder = num_chunked_indices - minimum_chunk_size * num_chunks
            chunk_sizes = [minimum_chunk_size + 1 * (i < num_remainder) for i in range(num_chunks)]
            drop_buffer = 0

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

        population = cls(data, generate_splits=False)

        if indices_dict["cell_split_indices"]:
            population.cell_split_indices = [torch.tensor(indices) for indices in indices_dict["cell_split_indices"]]

        if indices_dict["time_split_indices"]:
            population.time_split_indices = [torch.tensor(indices) for indices in indices_dict["time_split_indices"]]

        return population
