from enum import Enum
from time import perf_counter, time
from IPython.display import display, Markdown
import cProfile
import pstats
import io


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

    Profile code execution:
        with Timer("operation_name", profile=True) as t:
            # code to profile
        t.print_stats()  # prints profiling stats

    The context manager pattern ensures the timer is properly stopped
    even if exceptions occur.
    """

    def __init__(self, name: str | None = None, precision: int = 4, timer_method: str = "perf_counter", profile: bool = False):
        """Initialize a new timer.

        Args:
            name: Optional name for the timer. Used in timing report.
            precision: Number of decimal places to show in timing output
            timer_method: Which timer to use ("time" or "perf_counter")
            profile: Whether to enable profiling
        """
        self.name = name
        self.precision = precision
        self.timer_method = TimerMethods[timer_method.upper()].value
        self.start_time = None
        self.elapsed_time = None
        self.profile = profile
        self.profiler = cProfile.Profile() if profile else None

    def __enter__(self):
        """Start timing when entering context"""
        self.start_time = self.timer_method()
        if self.profile:
            self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context and print elapsed time"""
        if self.start_time is None:
            raise RuntimeError("Timer was not started: this should never happen, contact the developer on GitHub Issues!")

        if self.profile:
            self.profiler.disable()

        self.elapsed_time = self.timer_method() - self.start_time
        if self.name is None:
            print(f"Elapsed time: {self.elapsed_time:.{self.precision}f} seconds")
        else:
            print(f"{self.name} || elapsed time: {self.elapsed_time:.{self.precision}f} seconds")

    def print_stats(self, *args, **kwargs):
        """Print profiling statistics if profiling was enabled.

        Args:
            *args: Arguments to pass to pstats.Stats.sort_stats()
            **kwargs: Keyword arguments to pass to pstats.Stats.print_stats()
        """
        if not self.profile:
            print("Profiling was not enabled. Set profile=True when creating the Timer.")
            return

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        if args:
            ps.sort_stats(*args)
        ps.print_stats(**kwargs)
        print(s.getvalue())


def error_print(text):
    # supporting function for printing error messages but continuing
    display(Markdown(f"<font color=red>{text}</font>"))
