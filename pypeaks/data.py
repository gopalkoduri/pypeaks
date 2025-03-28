from __future__ import division
import pickle
from warnings import warn

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from . import slope


class Data:
    """
    The Data object for peak detection has methods necessary to handle the
    histogram/time-series like data.
    """
    def __init__(self, x, y, smoothness=7, default_smooth=True):
        """
        Initializes the data object for peak detection with x, y, smoothness
        parameter and peaks (empty initially). In a histogram, x refers to bin centers (not
        edges), and y refers to the corresponding count/frequency.

        The smoothness parameter refers to the standard deviation of the
        gaussian kernel used for smoothing.

        The peaks variable is a dictionary of the following form:
        {"peaks":[[peak positions], [peak amplitudes]],
        "valleys": [[valley positions], [valley amplitudes]]}
        """

        self.x = np.array(x)
        self.y_raw = np.array(y)
        self.y = np.array(y)
        self.smoothness = smoothness
        if default_smooth:
            self.smooth()
        self.peaks = {}

    def set_smoothness(self, smoothness):
        """
        This method (re)sets the smoothness parameter.
        """
        self.smoothness = smoothness
        self.smooth()

    def smooth(self):
        """
        Smooths the data using a gaussian kernel of the given standard deviation
        (as smoothness parameter)
        """
        self.y = gaussian_filter(self.y_raw, self.smoothness)

    def normalize(self):
        """
        Normalizes the given data such that the area under the histogram/curve
        comes to 1. Also re applies smoothing once done.
        """
        median_diff = np.median(np.diff(self.x))
        bin_edges = [self.x[0] - median_diff/2.0]
        bin_edges.extend(median_diff/2.0 + self.x)
        self.y_raw = self.y_raw/(self.y_raw.sum()*np.diff(bin_edges))
        self.smooth()

    def serialize(self, path):
        """
        Saves the raw (read unsmoothed) histogram data to the given path using
        pickle python module.
        """
        pickle.dump([self.x, self.y_raw], open(path, 'w'))

    def get_peaks(self, method="slope", peak_amp_thresh=0.00005,
                  valley_thresh=0.00003, intervals=None, lookahead=20,
                  avg_interval=100):
        """Detects peaks in a smoothed histogram using one of three methods.

        This function analyzes smoothed histogram data to identify significant peaks
        and their corresponding valleys. It requires pre-smoothed data - running it
        on raw histograms may result in no peaks being detected.

        Parameters
        ----------
        method : str, optional (default="slope")
            The peak detection method to use:
            - "slope": Uses slope analysis to identify peaks. More precise for 
              irregular data but may miss peaks in regular intervals.
            - "interval": Steps through histogram at regular intervals to find local
              maxima. Better for data with known regular spacing between peaks.
            - "hybrid": Combines slope and interval methods, using slope-based valleys
              when peaks are found in the same region by both methods.

        peak_amp_thresh : float, optional (default=0.00005)
            Minimum normalized amplitude a peak must have to be considered valid.
            Higher values result in fewer, more prominent peaks.

        valley_thresh : float, optional (default=0.00003)
            Minimum normalized depth a valley must have relative to adjacent peaks.
            Higher values ensure clearer separation between peaks.

        intervals : Intervals object, optional (default=None)
            Required for interval/hybrid methods. Defines the regular intervals
            at which to look for peaks. Must be an instance of Intervals class.

        lookahead : int, optional (default=20)
            For slope/hybrid methods: number of points to look ahead when 
            analyzing slope changes. Larger values smooth out small variations.

        avg_interval : int, optional (default=100)
            For slope/hybrid methods: approximate distance between peaks, used
            for determining search windows around peaks.

        Notes
        -----
        The function applies several filters to identify genuine peaks:
        1. Peak amplitude must exceed peak_amp_thresh
        2. Valley depth must exceed valley_thresh
        3. Left and right "lobes" around peak must be reasonably balanced
           (within 15% of each other in size)

        The detected peaks are stored in self.peaks dictionary with format:
        {
            "peaks": [
                [peak_position_1, peak_position_2, ...],  # x-values
                [peak_amplitude_1, peak_amplitude_2, ...]  # y-values
            ],
            "valleys": [
                [valley_position_1, valley_position_2, ...],  # x-values
                [valley_amplitude_1, valley_amplitude_2, ...]  # y-values
            ]
        }

        Examples
        --------
        # For regular spaced peaks (e.g. musical intervals):
        data.get_peaks(method="interval", intervals=interval_obj)

        # For irregular data with clear slope changes:
        data.get_peaks(method="slope", lookahead=30)

        # For complex data that may have both regular and irregular peaks:
        data.get_peaks(method="hybrid", intervals=interval_obj, lookahead=25)
        """

        # Initialize storage for peaks and their properties
        peaks = {}          # Will store final filtered peaks
        slope_peaks = {}    # Temporary storage for slope-based peaks
        
        # Note on indexing throughout this function:
        # - All position indices (left_index, right_index, etc.) refer to array 
        #   indices in self.x/self.y arrays
        # - To get actual x-values (e.g. cents), use self.x[index]
        # - Peak dictionary format: {position_index: [amplitude, left_valley_index, right_valley_index]}
        #----------------------------------------
        # Slope-based Peak Detection
        #----------------------------------------
        if method == "slope" or method == "hybrid":
            # Step 1: Detect potential peaks using slope changes
            # - lookahead: how far to look for slope changes
            # - delta (valley_thresh): minimum depth between peak and valley
            result = slope.peaks(self.x, self.y, lookahead=lookahead,
                                 delta=valley_thresh)

            # Step 2: Find enclosing valleys for each detected peak
            peak_data = result["peaks"]      # [x_values, amplitudes]
            valley_data = result["valleys"]   # [x_values, amplitudes]

            # Process each detected peak to find its enclosing valleys
            for i in range(len(peak_data[0])):
                # Find the valley point closest to current peak
                # This could be either left or right of the peak
                nearest_index = slope.find_nearest_index(valley_data[0],
                                                         peak_data[0][i])
                
                # Case 1: Nearest valley is to the left of peak
                if valley_data[0][nearest_index] < peak_data[0][i]:
                    # Use this valley as left boundary
                    left_index = slope.find_nearest_index(
                        self.x, valley_data[0][nearest_index])
                    
                    # Look for right valley
                    if len(valley_data[0][nearest_index + 1:]) == 0:
                        # No more valleys to right, use estimated position
                        right_index = slope.find_nearest_index(
                            self.x, peak_data[0][i] + avg_interval / 2)
                    else:
                        # Find next closest valley to the right
                        offset = nearest_index + 1
                        nearest_index = offset + slope.find_nearest_index(
                            valley_data[0][offset:], peak_data[0][i])
                        right_index = slope.find_nearest_index(
                            self.x, valley_data[0][nearest_index])
                
                # Case 2: Nearest valley is to the right of peak
                else:
                    # Use this valley as right boundary
                    right_index = slope.find_nearest_index(
                        self.x, valley_data[0][nearest_index])
                    
                    # Look for left valley
                    if len(valley_data[0][:nearest_index]) == 0:
                        # No more valleys to left, use estimated position
                        left_index = slope.find_nearest_index(
                            self.x, peak_data[0][i] - avg_interval / 2)
                    else:
                        # Find next closest valley to the left
                        nearest_index = slope.find_nearest_index(
                            valley_data[0][:nearest_index], peak_data[0][i])
                        left_index = slope.find_nearest_index(
                            self.x, valley_data[0][nearest_index])

                # Store peak with its properties:
                # - Convert peak x-value to index in self.x/self.y
                pos = slope.find_nearest_index(self.x, peak_data[0][i])
                # - Store [peak_amplitude, left_valley_index, right_valley_index]
                slope_peaks[pos] = [peak_data[1][i], left_index, right_index]

        if method == "slope":
            peaks = slope_peaks

        #----------------------------------------
        # Interval-based Peak Detection
        #----------------------------------------
        interval_peaks = {}  # Temporary storage for interval-based peaks
        if method == "interval" or method == "hybrid":
            # Intervals object required for this method
            if intervals is None:
                raise ValueError('The interval argument is not passed.')
            # Step 1: Calculate search boundaries
            # Use average interval size to determine where peaks are expected
            avg_interval = np.average(intervals.intervals[1:] - intervals.intervals[:-1])
            value = (min(self.x) + 1.5 * avg_interval) / avg_interval * avg_interval
            first_center = intervals.nearest_interval(value)
            value = (max(self.x) - avg_interval) / avg_interval * avg_interval
            last_center = intervals.nearest_interval(value)
            if first_center < intervals.intervals[0]:
                first_center = intervals.intervals[0]
                warn("In the interval based approach, the first center was seen\
                    to be too low and is set to " + str(first_center))
            if last_center > intervals.intervals[-1]:
                last_center = intervals.intervals[-1]
                warn("In the interval based approach, the last center was seen\
                     to be too high and is set to " + str(last_center))

            # Step 2: Scan through intervals to find peaks
            # For each interval:
            # 1. Calculate boundaries (midpoint between current and adjacent intervals)
            # 2. Find highest point within these boundaries
            # 3. Store peak with its enclosing valley points
            interval = first_center
            while interval <= last_center:
                prev_interval = intervals.prev_interval(interval)
                next_interval = intervals.next_interval(interval)
                left_index = slope.find_nearest_index(
                    self.x, (interval + prev_interval) / 2)
                right_index = slope.find_nearest_index(
                    self.x, (interval + next_interval) / 2)
                if left_index >= right_index:
                    interval = next_interval
                    continue
                peak_pos = np.argmax(self.y[left_index:right_index])
                # add left_index to peak_pos to get the correct position in x/y
                peak_amp = self.y[left_index + peak_pos]
                interval_peaks[left_index + peak_pos] = [peak_amp, left_index,
                                                         right_index]
                interval = next_interval

        if method == "interval":
            peaks = interval_peaks

        #----------------------------------------
        # Hybrid Method: Merge Results
        #----------------------------------------
        if method == "hybrid":
            # Get peak positions from both methods
            slope_positions = list(slope_peaks.keys())      # From slope analysis
            interval_positions = list(interval_peaks.keys()) # From interval analysis
            all_peaks = {}
            
            # Step 1: Remove interval-based peaks that are too close to slope-based peaks
            # We prefer slope-based valleys as they're typically more precise
            for slope_pos in slope_positions:
                if len(interval_positions) > 0:
                    # Find closest interval-based peak
                    near_index = slope.find_nearest_index(interval_positions, slope_pos)
                    # If peaks are within half an interval, consider them the same peak
                    if abs(slope_pos - interval_positions[near_index]) < avg_interval / 2:
                        interval_positions.pop(near_index)  # Remove duplicate peak
            
            # Step 2: Combine unique peaks from both methods
            # First add all slope-based peaks
            for p in slope_positions:
                all_peaks[p] = slope_peaks[p]
            # Then add remaining interval-based peaks
            for p in interval_positions:
                all_peaks[p] = interval_peaks[p]
                
            peaks = all_peaks

        # Finally, filter the peaks and retain eligible peaks, also get
        # their valley points.

        # check 1: peak_amp_thresh
        peak_positions = list(peaks.keys())
        for pos in peak_positions:
            # pos is an index in x/y. DOES NOT refer to a cent value.
            if peaks[pos][0] < peak_amp_thresh:
                peaks.pop(pos)

        # check 2, 3: valley_thresh, proportion of size of left and right lobes
        peak_positions = list(peaks.keys())
        valleys = {}
        for pos in peak_positions:
            # Each peak must have balanced "lobes" (regions on either side)
            # peaks[pos] format: [peak_amplitude, left_index, right_index]
            left_lobe = self.y[peaks[pos][1]:pos]    # Region between left valley and peak
            right_lobe = self.y[pos:peaks[pos][2]]   # Region between peak and right valley
            
            # Skip peaks with empty lobes (this shouldn't happen in practice)
            if len(left_lobe) == 0 or len(right_lobe) == 0:
                peaks.pop(pos)
                continue
                
            # Ensure peak is reasonably symmetric (within 85% difference)
            # This helps filter out artifacts and noise
            if len(left_lobe) / len(right_lobe) < 0.15 or len(right_lobe) / len(left_lobe) < 0.15:
                peaks.pop(pos)
                continue
                
            # Find the deepest points (valleys) on each side of the peak
            left_valley_pos = np.argmin(left_lobe)    # Index of minimum in left region
            right_valley_pos = np.argmin(right_lobe)  # Index of minimum in right region
            # Validate peak using valley depths
            # Both valleys must be deep enough compared to peak height
            peak_height = self.y[pos]
            left_valley_depth = abs(left_lobe[left_valley_pos] - peak_height)
            right_valley_depth = abs(right_lobe[right_valley_pos] - peak_height)
            
            if (left_valley_depth < valley_thresh and right_valley_depth < valley_thresh):
                peaks.pop(pos)  # Remove peaks with shallow valleys
            else:
                # Store valid valley points with their amplitudes
                # Convert local indices back to global indices in self.y
                valleys[peaks[pos][1] + left_valley_pos] = left_lobe[left_valley_pos]
                valleys[pos + right_valley_pos] = right_lobe[right_valley_pos]

        #----------------------------------------
        # Format Final Results
        #----------------------------------------
        if len(peaks) > 0:
            # Extract peak amplitudes from the peaks dictionary values
            # peaks[pos] format is [amplitude, left_index, right_index]
            # so we take the first element (amplitude) from each value
            peak_amps = np.array(list(peaks.values()))
            peak_amps = peak_amps[:, 0]
            
            # Convert array indices to actual x-values for the final result
            # peaks.keys() and valleys.keys() are indices into self.x
            # self.x[index] gives the actual x-value (e.g., cents in music)
            self.peaks = {
                'peaks': [
                    self.x[list(peaks.keys())],    # Convert peak indices to x-values
                    peak_amps                       # Peak amplitudes
                ],
                'valleys': [
                    self.x[list(valleys.keys())],  # Convert valley indices to x-values
                    list(valleys.values())         # Valley amplitudes
                ]
            }
        else:
            self.peaks = {'peaks': [[], []], 'valleys': [[], []]}

    def extend_peaks(self, prop_thresh=50):
        """Each peak in the peaks of the object is checked for its presence in 
        other octaves. If it does not exist, it is created. 
        
        prop_thresh is the cent range within which the peak in the other octave 
        is expected to be present, i.e., only if there is a peak within this 
        cent range in other octaves, then the peak is considered to be present
        in that octave.

        Note that this does not change the peaks of the object. It just returns 
        the extended peaks.
        """

        # octave propagation of the reference peaks
        temp_peaks = [i + 1200 for i in self.peaks["peaks"][0]]
        temp_peaks.extend([i - 1200 for i in self.peaks["peaks"][0]])
        extended_peaks = []
        extended_peaks.extend(self.peaks["peaks"][0])
        for i in temp_peaks:
            # if a peak exists around, don't add this new one.
            nearest_ind = slope.find_nearest_index(self.peaks["peaks"][0], i)
            diff = abs(self.peaks["peaks"][0][nearest_ind] - i)
            diff = np.mod(diff, 1200)
            if diff > prop_thresh:
                extended_peaks.append(i)
        return extended_peaks

    def plot(self, intervals=None, new_fig=True):
        """This function plots histogram together with its smoothed
        version and peak information if provided. Just intonation
        intervals are plotted for a reference."""

        import pylab as p

        if new_fig:
            p.figure()

        #step 1: plot histogram
        p.plot(self.x, self.y, ls='-', c='b', lw='1.5')

        #step 2: plot peaks
        first_peak = None
        last_peak = None
        if self.peaks:
            first_peak = min(self.peaks["peaks"][0])
            last_peak = max(self.peaks["peaks"][0])
            p.plot(self.peaks["peaks"][0], self.peaks["peaks"][1], 'rD', ms=10)
            p.plot(self.peaks["valleys"][0], self.peaks["valleys"][1], 'yD', ms=5)

        #Intervals
        if intervals is not None:
            #spacing = 0.02*max(self.y)
            for interval in intervals:
                if first_peak is not None:
                    if interval <= first_peak or interval >= last_peak:
                        continue
                p.axvline(x=interval, ls='-.', c='g', lw='1.5')
                if interval-1200 >= min(self.x):
                    p.axvline(x=interval-1200, ls=':', c='b', lw='0.5')
                if interval+1200 <= max(self.x):
                    p.axvline(x=interval+1200, ls=':', c='b', lw='0.5')
                if interval+2400 <= max(self.x):
                    p.axvline(x=interval+2400, ls='-.', c='r', lw='0.5')
                #spacing *= -1

        #p.title("Tonic-aligned complete-range pitch histogram")
        #p.xlabel("Pitch value (Cents)")
        #p.ylabel("Normalized frequency of occurence")
        p.show()
