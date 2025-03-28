import numpy as np
from .slope import find_nearest_index


class Intervals:
    def __init__(self, intervals):
        """
        Initializes the Intervals object with a set of given intervals.
        """
        self.intervals = np.array(intervals)

    def prev_interval(self, interval):
        """
        Given a value of an interval, this function returns the
        previous interval value
        """
        index = np.where(self.intervals == interval)
        if index[0][0] - 1 < len(self.intervals):
            return self.intervals[index[0][0] - 1]
        else:
            raise IndexError("Ran out of intervals!")

    def next_interval(self, interval):
        """
        Given a value of an interval, this function returns the 
        next interval value
        """
        index = np.where(self.intervals == interval)
        if index[0][0] + 1 < len(self.intervals):
            return self.intervals[index[0][0] + 1]
        else:
            raise IndexError("Ran out of intervals!")

    def nearest_interval(self, interval):
        """
        Returns the nearest defined interval to any given interval value.
        
        Instead of raising an error when the interval is outside the defined range,
        this function clamps the input to the nearest valid interval. This makes
        the function more robust when dealing with edge cases in peak detection.

        Parameters
        ----------
        interval : float
            The interval value to find the nearest match for (in cents)

        Returns
        -------
        float
            The nearest defined interval value from self.intervals
        """
        # Clamp the input value to the valid range of intervals
        interval = max(self.intervals[0], min(self.intervals[-1], interval))
        
        # Find and return the nearest interval
        index = find_nearest_index(self.intervals, interval)
        return self.intervals[index]
