from typing import Union, Any, Dict, Type, TypeVar, Union
from typing import Any, Protocol, Type, runtime_checkable, get_type_hints
from dataclasses import is_dataclass
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

    def __str__(self) -> str:
        return self.strftime("%Y-%m-%d")

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


T = TypeVar("T")


def resolve_dataclass(input_data: Union[Dict[str, Any], None, T], dataclass_type: Type[T]) -> T:
    """
    Resolve an input to the desired dataclass instance.

    Parameters
    ----------
    input_data : Dict[str, Any] | None | T
        The input data, which can be a dictionary of parameters, None, or
        already an instance of the target dataclass type.
    dataclass_type : Type[T]
        The dataclass type to convert the input to.

    Returns
    -------
    T
        An instance of the specified dataclass type.

    Raises
    ------
    ValueError
        If the input is not a dict, None, or an instance of the target dataclass.
    """
    if not isinstance(dataclass_type, type):
        raise ValueError(f"dataclass_type must be a type, got a {type(dataclass_type)}")

    if input_data is None:
        return dataclass_type()
    elif isinstance(input_data, dict):
        return dataclass_type(**input_data)
    elif isinstance(input_data, dataclass_type):
        return input_data
    elif isinstance(input_data, make_protocol_from_dataclass(dataclass_type)):
        return input_data
    elif type(input_data).__name__ == dataclass_type.__name__:
        # TODO: When reloading modules we need to do this
        print(f"Types don't officially match, but names do: {dataclass_type.__name__} from {type(input_data).__name__}")
        return input_data
    else:
        raise ValueError(f"Input must be a dict, None, or an instance of {dataclass_type.__name__}, " f"got {type(input_data).__name__}")


def make_protocol_from_dataclass(dataclass_type: Type) -> Type:
    """
    Generate a runtime_checkable Protocol from a dataclass.

    Parameters
    ----------
    dataclass_type : Type
        The dataclass to create a Protocol from

    Returns
    -------
    Type[Protocol]
        A runtime_checkable Protocol class with the same interface
    """
    annotations = get_type_hints(dataclass_type)
    method_names = [
        name
        for name in dir(dataclass_type)
        if callable(getattr(dataclass_type, name)) and not name.startswith("__") and name not in ("from_dict", "from_path")
    ]  # Exclude class methods

    # Create protocol attribute annotations and stub methods
    namespace = {"__annotations__": annotations}

    # Add stub methods
    for method_name in method_names:
        namespace[method_name] = lambda *args, **kwargs: ...

    # Create the Protocol class
    protocol_name = f"{dataclass_type.__name__}Protocol"
    protocol_class = type(protocol_name, (Protocol,), namespace)

    # Make it runtime_checkable
    return runtime_checkable(protocol_class)
