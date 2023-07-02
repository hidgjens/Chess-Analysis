"""
:file: chess_analysis/lah_util.py
:author: hidgjens
:created: 2023/07/01
:last modified: 2023/07/02
"""

import time
from dataclasses import dataclass
from typing import Final

_SECONDS_IN_MIN: Final = 60
_SECONDS_IN_HOUR: Final = 60 * _SECONDS_IN_MIN
_SECONDS_IN_DAY: Final = 24 * _SECONDS_IN_HOUR
_SECONDS_IN_MS: Final = 0.001


@dataclass
class Timer:
    """
    Simple timer class that can be used as a context manager.

        >>> with Timer("Sleeping"):
        ...     time.sleep(10)
        Starting 'Sleeping'
        Sleeping took 10.0 sec

    Can also be used as:

        >>> timer = Timer("Sleeping")
        >>> timer.start()
        Starting 'Sleeping'
        >>> time.sleep(10)
        >>> duration = timer.stop()
        Sleeping took 10.0 sec
        >>> # Can also access duration as an attribute:
        >>> duration = timer.duration_s
    """

    proc_name: str

    _start: float = 0
    _end: float = 0
    duration_s: float = 0

    def __enter__(self):
        self.start()

    def start(self, verbose: bool = True):
        """
        Start the timer.

        :param verbose:
            Print out message saying process has started, defaults to True
        :type verbose:
            bool, optional
        """
        self._start = time.perf_counter()
        if verbose:
            print(f"Starting '{self.proc_name}'")

    def __exit__(self, _1, _2, _3):
        self.stop()

    def stop(self, verbose: bool = True) -> float:
        """
        Stop the timer. Returns the duration in seconds.

        :param verbose:
            Print out message reporting time taken, defaults to True
        :type verbose:
            bool, optional
        :return:
            The time taken in seconds.
        :rtype:
            float
        """
        self._end = time.perf_counter()
        delta = self._end - self._start
        self.duration_s = delta
        # Print out.
        if verbose:
            time_str = self._time_to_string(self.duration_s)
            print(f"{self.proc_name} took {time_str}")
        return self.duration_s

    @staticmethod
    def _time_to_string(seconds: float) -> str:
        if seconds > 2 * _SECONDS_IN_DAY:
            days = seconds / _SECONDS_IN_DAY
            return f"{days:.1f} days"
        if seconds > 2 * _SECONDS_IN_HOUR:
            hours = seconds / _SECONDS_IN_HOUR
            return f"{hours:.1f} hr"
        if seconds > 2 * _SECONDS_IN_MIN:
            mins = seconds / _SECONDS_IN_MIN
            return f"{mins:.1f} min"
        if seconds > 1:
            return f"{seconds:.1f} sec"
        ms = seconds / _SECONDS_IN_MS
        return f"{ms:.1f} ms"


def isolate_games(src_file: str, dest_file: str, n_games: int = 1):
    """
    Read a pgn file and copy the first N games to a new file.

    This works by assuming there are two blank lines per game.
    It scans the files looking for the blank lines to determine
    where game data starts and ends.

    TODO a better method for detecting games was implemented in
    piece_capture.py. This function should be changed to use that
    method instead.

    :param src_file:
        Source .pgn file.
    :type src_file:
        str
    :param dest_file:
        Destination file.
    :type dest_file:
        str
    :param n_games:
        How many games to copy over, defaults to 1.
    :type n_games:
        int, optional
    """
    # Keep track of empty lines.
    empty_lines = 0
    with open(src_file, "r") as src_file:
        with open(dest_file, "w") as dest_file:
            while True:
                line = src_file.readline()

                if line.strip("\n"):
                    dest_file.write(line)
                else:
                    empty_lines += 1
                    if empty_lines >= 2 * n_games:
                        break


def test_timer():
    time_exp_results = [
        (0.1, "100.0 ms"),
        (5, "5.0 sec"),
        (2.5 * _SECONDS_IN_MIN, "2.5 min"),
        (2.5 * _SECONDS_IN_HOUR, "2.5 hr"),
        (2.5 * _SECONDS_IN_DAY, "2.5 days"),
    ]

    for time_s, expected_result in time_exp_results:
        result = Timer._time_to_string(time_s)
        if result != expected_result:
            print(
                f"Timer._time_to_string({time_s!r}) return {result!r} (expected {expected_result!r})"
            )

    with Timer("Testing sleep 2 sec"):
        time.sleep(2)


if __name__ == "__main__":
    test_timer()
