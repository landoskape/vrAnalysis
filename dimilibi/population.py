from typing import Union, Optional, Dict, Literal, List
import numpy as np
import torch


def _ensure_tensor(data: Union[np.ndarray, torch.Tensor, list, tuple]) -> torch.Tensor:
    """Convert input data to a torch tensor, handling multiple input types.

    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor, list, tuple]
        Input data to convert. Can be a numpy array, existing torch tensor,
        or any sequence-like object that can be converted to a tensor.

    Returns
    -------
    torch.Tensor
        A torch tensor containing the data. If input was already a tensor,
        returns a detached clone. If input was a numpy array, returns a tensor
        sharing memory with the numpy array (via torch.from_numpy).
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data.clone().detach()
    else:
        return torch.tensor(data)


class Population:
    """
    Population is a class that holds and manipulates a dataset of neurons by timepoints for use in dimensionality analyses.

    Many analyses in this library require manipulating a dataset of neurons in groups for cross-validating purposes. The primary two operations
    are splitting cells into groups and splitting the timepoints into groups (often with special considerations for intelligent cross-validation).
    """

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        cell_split_prms: dict = {},
        time_split_prms: dict = {},
        dtype: Optional[torch.dtype] = None,
        generate_splits: bool = True,
        idx_neurons: Optional[Union[np.ndarray, torch.Tensor]] = None,
        idx_samples: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Initialize the Population object

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            A 2D array of shape (num_neurons, num_timepoints) representing the activity of neurons over time.
        cell_split_prms : dict
            Parameters for splitting the cells into groups.
            (default is {})
        time_split_prms : dict
            Parameters for splitting the timepoints into groups.
            (default is {})
        dtype : Optional[torch.dtype]
            The data type to cast the data to if it isn't already a torch tensor.
            (default is None)
        generate_splits : bool
            If True, will generate splits for cells and timepoints using any parameters in cell_split_prms and time_split_prms, respectively.
            (default is True)
        idx_neurons: Optional[Union[np.ndarray, torch.Tensor]] = None
            The indices of the neurons to use in the population. If None, will use all neurons.
            (default is None)
        idx_samples: Optional[Union[np.ndarray, torch.Tensor]] = None
            The indices of the samples to use in the population. If None, will use all samples.
            (default is None)
        """
        # Converts to a torch tensor if necessary and throws an error if it can't
        data = self._check_datatype(data)

        if len(data.shape) != 2:
            raise ValueError("data must be a 2D array/tensor with shape (num_neurons, num_timepoints)")

        # Store the data and note the total number of neurons and timepoints
        self.data = data
        self.total_neurons, self.total_timepoints = data.shape

        # Build filtering indices to the neurons and timepoints to use
        self.idx_neurons = _validate_indices(idx_neurons, self.total_neurons)
        self.idx_samples = _validate_indices(idx_samples, self.total_timepoints)
        self.num_neurons = len(self.idx_neurons)
        self.num_timepoints = len(self.idx_samples)
        self.dtype = dtype

        # Generate splits for cells and timepoints
        if generate_splits:
            self.split_cells(**cell_split_prms)
            self.split_times(**time_split_prms)

    def get_split_data(
        self,
        time_idx: Optional[Union[int, list[int], tuple[int]]] = None,
        center: bool = False,
        scale: Union[bool, float] = False,
        pre_split: bool = False,
        scale_type: Optional[str] = None,
    ):
        """
        Get the source and target data for a specific set of timepoints.

        Parameters
        ----------
        time_idx : Optional[Union[int, list[int], tuple[int]]]
            The time group(s) to use as the target data. If a list or tuple of integers, will concatenate the data
            for the specified time groups. If None, will use all timepoints in the data.
            (default is None)
        center : bool
            If True, will center the data so each neuron has a mean of 0 across timepoints
            (default is False)
        scale : Union[bool, float]
            If scale_type is not 'percentile': If True, will scale the data. If False, no scaling is applied.
            If scale_type is 'percentile': The percentile value (0-100) to use for scaling. Each neuron will be
            scaled by its percentile value across timepoints.
            (default is False)
        pre_split : bool
            If True, will center and scale the data before splitting into source and target data
        scale_type : Optional[str]
            How to scale the data.
            =[None, 'std'] -> will scale the data by the standard deviation of each neuron.
            ='sqrt' -> will scale the data by the square root of the standard deviation of each neuron.
            ='preserve' -> will scale the data such that the neuron with median std will end up with a std of 1,
            and the other neurons will be scaled accordingly (preserving the relative standard deviation).
            ='percentile' -> will scale the data by a percentile of each neuron's values across timepoints.
            When using 'percentile', the scale parameter must be a numeric value (0-100) representing the percentile.
            ='max' -> will scale the data by the maximum value of each neuron across timepoints.

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

        source = self.apply_split(
            self.data[self.idx_neurons[self.cell_split_indices[0]]],
            time_idx,
            center,
            scale,
            pre_split,
            scale_type,
            prefiltered=False,
        )
        target = self.apply_split(
            self.data[self.idx_neurons[self.cell_split_indices[1]]],
            time_idx,
            center,
            scale,
            pre_split,
            scale_type,
            prefiltered=False,
        )
        return source, target

    def get_split_times(self, time_idx: Optional[Union[int, list[int], tuple[int]]] = None, within_idx_samples: bool = True) -> torch.Tensor:
        """Get the timepoints for a specific set of timepoints.

        Parameters
        ----------
        time_idx : Optional[Union[int, list[int], tuple[int]]]
            The time group(s) to use as the target data. If a list or tuple of integers,
            will concatenate the indices for the specified time groups. If None, will use
            all timepoints in the data. (default is None).
        within_idx_samples : bool
            If True, will return the indices to the timepoints within the idx_samples array. This is how the population
            is filtered by get_split_data, otherwise it simply provides the indices to idx_samples, which may not
            be contiguous. Default is True to match the behavior of get_split_data.

        Returns
        -------
        torch.Tensor
            The indices to timepoints for the specified time groups. Will be sorted in ascending order.
        """
        if not hasattr(self, "time_split_indices"):
            raise ValueError("time_split_indices must be set before calling get_split_times")

        if isinstance(time_idx, int):
            assert time_idx < len(self.time_split_indices), "time_idx must correspond to one of the time groups in time_split_indices"
            time_idx = [time_idx]
        elif isinstance(time_idx, list) or isinstance(time_idx, tuple):
            assert all(isinstance(idx, int) for idx in time_idx), "time_idx must be a list or tuple of integers"
            assert all(
                idx < len(self.time_split_indices) for idx in time_idx
            ), "time_idx must correspond to one of the time groups in time_split_indices"
        elif time_idx is None:
            time_idx = list(range(len(self.time_split_indices)))
        else:
            if time_idx is not None:
                raise ValueError("time_idx must be an integer or None")

        idx_times = torch.sort(torch.cat([self.time_split_indices[idx] for idx in time_idx])).values
        if within_idx_samples:
            return self.idx_samples[idx_times]
        else:
            return idx_times

    def get_split_cells(self, group: Literal["source", "target"], within_idx_neurons: bool = True) -> torch.Tensor:
        """Get the cells for a specific group.

        Parameters
        ----------
        group : Literal["source", "target"]
            The group to get the cells for.
        within_idx_neurons : bool
            If True, will return the indices to the cells within the idx_neurons array. This is how the population
            is filtered by get_split_data, otherwise it simply provides the indices to idx_neurons, which may not
            be contiguous. Default is True to match the behavior of get_split_data.

        Returns
        -------
        torch.Tensor
            The indices to the cells for the specified group.
        """
        if group == "source":
            csi = self.cell_split_indices[0]
        elif group == "target":
            csi = self.cell_split_indices[1]
        else:
            raise ValueError("group must be one of ['source', 'target']")

        if within_idx_neurons:
            return self.idx_neurons[csi]
        else:
            return csi

    def _compute_percentile_scale(self, data: torch.Tensor, scale: Union[bool, float]) -> torch.Tensor:
        """
        Compute percentile scaling values for each neuron.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor with shape (num_features, num_timepoints).
        scale : Union[bool, float]
            The percentile value (0-100) to use for scaling.

        Returns
        -------
        torch.Tensor
            A tensor with shape (num_features, 1) containing the percentile values for each neuron.
            Zero values are replaced with 1 to avoid division by zero.

        Raises
        ------
        ValueError
            If scale is not a numeric value or is outside the valid range [0, 100].
        """
        # Validate percentile usage
        if not isinstance(scale, (int, float)):
            raise ValueError("When scale_type='percentile', scale must be a numeric " f"percentile value (0-100), got {type(scale).__name__}")
        percentile = float(scale)
        if not (0 <= percentile <= 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")

        # Compute percentile per neuron across timepoints
        quantile = percentile / 100.0
        percentile_val = torch.quantile(data, quantile, dim=1, keepdim=True)
        percentile_val[percentile_val == 0] = 1
        return percentile_val

    def apply_split(
        self,
        data: torch.Tensor,
        time_idx: Optional[Union[int, list[int], tuple[int]]] = None,
        center: bool = False,
        scale: Union[bool, float] = False,
        pre_split: bool = False,
        scale_type: Optional[str] = None,
        prefiltered: bool = True,
    ):
        """
        Apply the time splits to a new dataset. If time_idx is a list or tuple of integers, will concatenate the data for the specified time groups.

        Parameters
        ----------
        data : torch.Tensor
            The data to apply the time splits to. Must have shape (num_features, num_timepoints), where
            num_features is unconstrained and num_timepoints must match the number of timepoints in the
            Population instance.
        time_idx : Optional[Union[int, list[int], tuple[int]]]
            The time group(s) to use as the target data. If a list or tuple of integers, will concatenate the data for the specified time groups.
            If None, will use all timepoints in the data.
            (default is None)
        center : bool
            If True, will center the data so each neuron has a mean of 0 across timepoints
            (default is False)
        scale : Union[bool, float]
            If scale_type is not 'percentile': If True, will scale the data. If False, no scaling is applied.
            If scale_type is 'percentile': The percentile value (0-100) to use for scaling. Each neuron will be
            scaled by its percentile value across timepoints.
            (default is False)
        pre_split : bool
            If True, will center and scale the data before splitting into source and target data
        scale_type : Optional[str]
            How to scale the data.
            =[None, 'std'] -> will scale the data by the standard deviation of each neuron.
            ='sqrt' -> will scale the data by the square root of the standard deviation of each neuron.
            ='preserve' -> will scale the data such that the neuron with median std will end up with a std of 1,
            and the other neurons will be scaled accordingly (preserving the relative standard deviation).
            ='percentile' -> will scale the data by a percentile of each neuron's values across timepoints.
            When using 'percentile', the scale parameter must be a numeric value (0-100) representing the percentile.
            ='max' -> will scale the data by the maximum value of each neuron across timepoints.
        prefiltered : bool
            If True, assumes that provided data is already filtered by idx_samples. Defaults to True.

        Returns
        -------
        torch.Tensor
            The data for the specified set of timepoints.
        """
        # Convert data to torch tensor if necessary and throw an error if it can't
        data = self._check_datatype(data)
        if prefiltered:
            if data.size(1) != self.num_timepoints:
                raise ValueError("Data must be pre-filtered to match the length of idx_samples when prefiltered is True")
        else:
            if data.size(1) != self.total_timepoints:
                raise ValueError("Data must have the same number of timepoints as the Population instance when prefiltered is False")

        if pre_split:
            if center:
                mean = data.mean(dim=1, keepdim=True)
            if scale:
                if scale_type == "percentile":
                    percentile_val = self._compute_percentile_scale(data, scale)
                elif scale_type == "max":
                    max_val = data.max(dim=1, keepdim=True)[0]
                    max_val[max_val == 0] = 1
                else:
                    # Validate boolean usage for non-percentile scaling
                    if not isinstance(scale, bool):
                        raise ValueError(f"When scale_type is not 'percentile', scale must be a boolean, " f"got {type(scale).__name__}")
                    std = data.std(dim=1, keepdim=True)
                    std[std == 0] = 1

        # Select the timepoints for the specified group
        if time_idx is not None:
            idx_times = self.get_split_times(time_idx, within_idx_samples=not prefiltered)
            data = data[:, idx_times]

        # Center the data if requested
        if center:
            if pre_split:
                data = data - mean
            else:
                data = data - data.mean(dim=1, keepdim=True)

        if scale:
            # Validate scale parameter based on scale_type
            if scale_type == "percentile":
                # Compute percentile scaling
                if pre_split:
                    # Use pre-computed percentile value
                    data = data / percentile_val
                else:
                    # Compute percentile per neuron across timepoints
                    percentile_val = self._compute_percentile_scale(data, scale)
                    data = data / percentile_val
            elif scale_type == "max":
                # Validate boolean usage for max scaling
                if not isinstance(scale, bool):
                    raise ValueError(f"When scale_type='max', scale must be a boolean, " f"got {type(scale).__name__}")

                # Compute max per neuron across timepoints if not pre-computed
                if pre_split:
                    # Use pre-computed max value
                    data = data / max_val
                else:
                    max_val = data.max(dim=1, keepdim=True)[0]
                    max_val[max_val == 0] = 1
                    data = data / max_val
            else:
                # Validate boolean usage for non-percentile scaling
                if not isinstance(scale, bool):
                    raise ValueError(f"When scale_type is not 'percentile', scale must be a boolean, " f"got {type(scale).__name__}")

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
                    raise ValueError(f"scale_type must be one of [None, 'std', 'sqrt', 'preserve', 'percentile', 'max'], got {scale_type}")

        if self.dtype is not None:
            data = data.to(self.dtype)

        return data

    def split_cells(self, force_even: bool = False):
        """
        Assign indices to each neurons to split into two groups.

        TODO: Add method diversion to split neurons into groups based on additional criteria (e.g. location chunks)

        Parameters
        ----------
        force_even : bool
            If True, will ensure that the number of neurons in each group is equal to each other.
            If number of neurons isn't divisible by number of groups, will clip neurons randomly.
            (default is False)
        """
        index = torch.randperm(self.num_neurons)

        # Clip neurons to be divisible by num_groups if requested by user (use randomly clipped neuron(s))
        if force_even:
            num_per_group = self.num_neurons // 2
            total_neurons = num_per_group * 2
            index = index[:total_neurons]

        self.cell_split_indices = [torch.sort(indices).values for indices in torch.tensor_split(index, 2)]

    def split_times(
        self,
        num_groups: int = 2,
        relative_size: Optional[tuple[int]] = None,
        chunks_per_group: int = 5,
        num_buffer: int = 10,
        force_even: bool = False,
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
        relative_size : Optional[tuple[int]]
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
        """
        if relative_size is None:
            relative_size = [1] * num_groups
        else:
            assert len(relative_size) == num_groups, "relative_size must have the same length as num_groups"
            assert all(isinstance(size, int) for size in relative_size), "relative_size must be a tuple of integers"

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
        start_stop_index = torch.cumsum(_ensure_tensor([0] + [rs * chunks_per_group for rs in relative_size]), dim=0)
        time_split_indices: list[torch.Tensor] = []
        for i in range(num_groups):
            time_split_indices.append(torch.sort(torch.cat(time_chunks[start_stop_index[i] : start_stop_index[i + 1]])).values)

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

    def to(self, device):
        """
        Move the Population to a device.

        Parameters
        ----------
        device : torch.device or str
            The device to move the Population to (e.g., 'cpu', 'cuda', torch.device('cuda:0')).

        Returns
        -------
        self : Population
            The Population object with all tensors moved to the specified device.
        """
        self.data = self.data.to(device)
        if hasattr(self, "idx_neurons"):
            self.idx_neurons = self.idx_neurons.to(device)
        if hasattr(self, "idx_samples"):
            self.idx_samples = self.idx_samples.to(device)
        if hasattr(self, "cell_split_indices"):
            self.cell_split_indices = [indices.to(device) for indices in self.cell_split_indices]
        if hasattr(self, "time_split_indices"):
            self.time_split_indices = [indices.to(device) for indices in self.time_split_indices]
        return self

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
            "total_neurons": self.total_neurons,
            "total_timepoints": self.total_timepoints,
            "num_neurons": self.num_neurons,
            "num_timepoints": self.num_timepoints,
            "idx_neurons": self.idx_neurons.tolist(),
            "idx_samples": self.idx_samples.tolist(),
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
        # Backwards compatibility with old indices_dicts
        # They didn't have the sub indices idx_neurons, idx_samples, so num_* represents total_*
        if "total_neurons" not in indices_dict:
            indices_dict["total_neurons"] = indices_dict["num_neurons"]
            indices_dict["total_timepoints"] = indices_dict["num_timepoints"]
            indices_dict["idx_neurons"] = torch.arange(indices_dict["total_neurons"])
            indices_dict["idx_samples"] = torch.arange(indices_dict["total_timepoints"])

        if "idx_neurons" not in indices_dict or "idx_samples" not in indices_dict:
            raise ValueError("idx_neurons and idx_samples must be provided in the indices_dict")
        else:
            idx_neurons = _ensure_tensor(indices_dict["idx_neurons"])
            idx_samples = _ensure_tensor(indices_dict["idx_samples"])

        # Check if the shape of the provided data matches the saved indices
        if data.shape != (indices_dict["total_neurons"], indices_dict["total_timepoints"]):
            raise ValueError(
                f"Shape of provided data {data.shape} doesn't match the shape in saved indices "
                f"({indices_dict['total_neurons']}, {indices_dict['total_timepoints']})"
            )

        # Remake population with saved indices, don't generate splits because we already have them
        population = cls(data, generate_splits=False, dtype=indices_dict.get("dtype", None), idx_neurons=idx_neurons, idx_samples=idx_samples)

        if indices_dict["cell_split_indices"]:
            population.cell_split_indices = [_ensure_tensor(indices) for indices in indices_dict["cell_split_indices"]]

        if indices_dict["time_split_indices"]:
            population.time_split_indices = [_ensure_tensor(indices) for indices in indices_dict["time_split_indices"]]

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
        time_split_prms={},
        dtype: Optional[torch.dtype] = None,
        generate_splits: bool = True,
    ):
        """
        Initialize the SourceTarget object

        Parameters
        ----------
        source : Union[np.ndarray, torch.Tensor]
            A 2D array of shape (num_source_features, num_timepoints) representing variable used to make predictions.
        target : Union[np.ndarray, torch.Tensor]
            A 2D array of shape (num_target_features, num_timepoints) representing variable to predict.
        time_split_prms : dict
            Parameters for splitting the timepoints into groups.
            (default is {})
        dtype : Optional[torch.dtype]
            The data type to cast the data to if it isn't already a torch tensor.
            (default is None)
        generate_splits : bool
            If True, will generate splits for timepoints using any parameters in time_split_prms.
            (default is True)
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
            source_target.cell_split_indices = [_ensure_tensor(indices) for indices in indices_dict["cell_split_indices"]]

        if indices_dict["time_split_indices"]:
            source_target.time_split_indices = [_ensure_tensor(indices) for indices in indices_dict["time_split_indices"]]

        return source_target


# Create indices to the neurons and timepoints to use in data splits
def _validate_indices(
    indices: Optional[Union[np.ndarray, torch.Tensor]],
    max_index: int,
) -> torch.Tensor:
    """Check and/or convert indices to integer indices within valid range and return as a torch tensor.

    Parameters
    ----------
    indices : Optional[Union[np.ndarray, torch.Tensor]]
        The indices to validate. Can be None, numpy array, or torch tensor.
    max_index : int
        The maximum valid index (exclusive). Indices must be in range [0, max_index).

    Returns
    -------
    torch.Tensor
        Validated integer indices as a torch tensor.

    Raises
    ------
    ValueError
        If indices are out of range integers or boolean masks with incorrect length.
    """
    if indices is None:
        return torch.arange(max_index)

    # Convert to torch tensor if needed
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)
    elif not isinstance(indices, torch.Tensor):
        raise TypeError(f"indices must be numpy array or torch tensor, got {type(indices)}")

    # Check if it's a boolean mask and convert if so
    if indices.dtype == torch.bool:
        if len(indices) != max_index:
            raise ValueError("boolean masks must be the same length as the maximum index")
        return torch.nonzero(indices)

    # Ensure integer type
    if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"indices must be integer type, got {indices.dtype}")
    indices = indices.to(torch.long)

    # Check range
    if len(indices) > 0:
        min_idx = indices.min().item()
        max_idx = indices.max().item()
        if min_idx < 0 or max_idx >= max_index:
            raise ValueError(f"indices must be in range [0, {max_index}), " f"got range [{min_idx}, {max_idx}]")

    return indices
