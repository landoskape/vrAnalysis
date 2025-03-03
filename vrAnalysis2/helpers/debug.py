from enum import Enum
from time import perf_counter, time


class TimerMethods(Enum):
    TIME = time
    PERF_COUNTER = perf_counter


class Timer:
    """A flexible timer utility that supports context manager usage.

    Preferred usage:
        with Timer("operation_name"):
            # code to time

    Access elapsed time:
        with Timer("operation_name") as t:
            # code to time
        print(f"Operation took {t.elapsed:.4f} seconds")

    The context manager pattern ensures the timer is properly stopped
    even if exceptions occur.
    """

    def __init__(self, name: str | None = None, precision: int = 4, timer_method: str = "perf_counter"):
        """Initialize a new timer.

        Args:
            name: Optional name for the timer. Used in timing report.
            timer_method: Which timer to use ("time" or "perf_counter")
        """
        self.name = name
        self.precision = precision
        self.timer_method = TimerMethods[timer_method.upper()].value
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        """Start timing when entering context"""
        self.start_time = self.timer_method()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context and print elapsed time"""
        if self.start_time is None:
            raise RuntimeError("Timer was not started: this should never happen, contact the developer on GitHub Issues!")

        self.elapsed_time = self.timer_method() - self.start_time
        if self.name is None:
            print(f"Elapsed time: {self.elapsed_time:.{self.precision}f} seconds")
        else:
            print(f"{self.name} || elapsed time: {self.elapsed_time:.{self.precision}f} seconds")
