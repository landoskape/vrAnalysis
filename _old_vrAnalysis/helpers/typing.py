from typing import Union
from argparse import ArgumentTypeError
from datetime import datetime


def cutoff_type(cutoff):
    if cutoff is None:
        return cutoff

    if len(cutoff) != 2:
        raise TypeError("cutoff is not None or a length=2 iterable")

    return cutoff


def positive_unit_float(value):
    """
    Check if a value is a float that is greater than 0 and less than or equal to 1
    """
    # First try to convert the value to a float
    try:
        value = float(value)
    except ValueError:
        raise TypeError("value is not a float")

    # Then check if the value is in range
    if value <= 0 or value > 1:
        raise ValueError("value is not between 0 and 1")

    return value


def positive_float(value):
    """Check if a value is a float that is greater than 0"""
    # First try to convert the value to a float
    try:
        value = float(value)
    except ValueError:
        raise TypeError("value is not a float")

    # Then check if the value is positive
    if value <= 0:
        raise ValueError("value is not greater than 0")

    return value


def argbool(value):
    """Convert a string to a boolean (for use with ArgumentParser)"""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


class PrettyDatetime(datetime):
    """Subclass of datetime with a prettier __repr__"""

    def __repr__(self) -> str:
        return self.strftime("%Y-%m-%d")

    @classmethod
    def make_pretty(cls, dt: Union[datetime, str, float]) -> "PrettyDatetime":
        """Convert a datetime object to PrettyDatetime"""
        if isinstance(dt, datetime):
            return cls.fromtimestamp(dt.timestamp())
        elif isinstance(dt, str):
            return cls.strptime(dt, "%Y-%m-%d")
        elif isinstance(dt, float):
            return cls.fromtimestamp(dt)
        else:
            raise ValueError(f"Invalid input type for PrettyDatetime: {type(dt)}")

    @classmethod
    def from_datetime(cls, dt: datetime) -> "PrettyDatetime":
        """Convert a datetime object to PrettyDatetime"""
        return cls.fromtimestamp(dt.timestamp())

    @classmethod
    def from_string(cls, date_string: str, format: str = "%Y-%m-%d") -> "PrettyDatetime":
        """Convert a string to PrettyDatetime using the specified format"""
        return cls.strptime(date_string, format)
