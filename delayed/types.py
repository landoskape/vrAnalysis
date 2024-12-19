from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar, Optional

T = TypeVar("T")
GenericType = Generic[T]


class NotComputed:
    """Singleton class to represent a computation that has not yet been performed."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self):
        return False

    def __repr__(self):
        return "NotComputed()"


class DelayedData(GenericType):
    """Manages the state and metadata of a cached computation result.

    Attributes
    ----------
    data : T
        The computed data.
    computed : bool
        Whether the data has been computed.
    compute_time : datetime
        The time of the last computation.
    compute_count : int
        The number of times the data has been computed.
    last_access : datetime
        The time of the last access.
    access_count : int
        The number of times the data has been accessed.

    Examples
    --------
    >>> ddata = DelayedData()
    >>> ddata.set(42)
    >>> ddata.get()
    42
    >>> if ddata:
    ...     print("Data is computed")
    Data is computed
    >>> ddata.stats
    {
        "computed": True,
        "compute_count": 1,
        "access_count": 1,
        "last_compute": datetime.datetime(2021, 8, 24, 12, 0, 0),
        "last_access": datetime.datetime(2021, 8, 24, 12, 0, 0),
        "age": 0.0,
    }
    >>> ddata.clear()
    >>> if not ddata:
    ...     print("Data is not computed")
    Data is not computed
    >>> ddata.stats
    {
        "computed": False,
        "compute_count": 0,
        "access_count": 0,
        "last_compute": None,
        "last_access": None,
        "age": None,
    }
    """

    def __init__(self) -> None:
        self.data = NotComputed()
        self.computed = False
        self.compute_time = None
        self.compute_count = 0
        self.last_access = None
        self.access_count = 0

    def set(self, data: T) -> None:
        """Set the computed data with metadata."""
        self.data = data
        self.computed = True
        self.compute_time = datetime.now()
        self.compute_count += 1

    def get(self) -> T:
        """Get the computed data, updating access metadata."""
        if not self.computed:
            raise ValueError("Data not yet computed")
        self.last_access = datetime.now()
        self.access_count += 1
        return self.data

    def clear(self, reset_stats: bool = False) -> None:
        """Clear the cached data, optionally resetting statistics."""
        self.data = NotComputed()
        self.computed = False
        if reset_stats:
            self.compute_count = 0
            self.access_count = 0
            self.compute_time = None
            self.compute_duration = None
            self.last_access = None

    def __bool__(self) -> bool:
        return self.computed

    def __call__(self) -> T:
        return self.get()

    def __repr__(self) -> str:
        if self.computed:
            return f"DelayedData(computed={self.computed}, datatype={type(self.data)})"
        else:
            return f"DelayedData(computed={self.computed})"

    @property
    def age(self) -> Optional[float]:
        """Time since last computation in seconds."""
        if self.compute_time is None:
            return None
        return (datetime.now() - self.compute_time).total_seconds()

    @property
    def stats(self) -> dict:
        """Return computation statistics."""
        return {
            "computed": self.computed,
            "compute_count": self.compute_count,
            "access_count": self.access_count,
            "last_compute": self.compute_time,
            "last_access": self.last_access,
            "age": self.age,
        }
