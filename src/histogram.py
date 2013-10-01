#!/usr/bin/env python

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from intervals import Intervals
import slope

class Histogram:
    def __init__(self, x, y, smoothness=7):
        self.x = x
        self.y_raw = y
        self.smoothness = smoothness
        self.y = self.smooth()

    def set_smoothness(self, smoothness):
        self.smoothness = smoothness

    def smooth(self):
        self.y = gaussian_filter(y, self.smoothness)

    def normalize(self):
        #TODO
        pass

    def serialize(self, path):
        pickle.dump([self.x, self.y_raw], file(path, 'w'))

    def get_peaks(self, method="slope", peak_amp_thresh=0.00005, 
              valley_thresh=0.00003, intervals = None, lookahead = 20,
                  avg_window = 100):
        """
        This function expects SMOOTHED histogram. If you run it on a raw histogram,
        there is a high chance that it returns no peaks.

        method can be interval/slope/hybrid.
            The interval-based method simply steps through the whole histogram
            and pick up the local maxima in each interval, from which irrelevant
            peaks are filtered out by looking at the proportion of points on 
            either side of the detected peak in each interval.
        
            Slope approach uses, of course slope information, to find peaks, 
            which are then filtered by applying peal_amp_thresh and 
            valley_thresh bounds. 
            
            Hybrid approach first finds peaks using slope method and then filters 
            them heuristically as in interval-based approach.
        
        peak_amp_thresh is the minimum amplitude/height that a peak should have
        in a normalized smoothed histogram, to be qualified as a peak. 
        valley_thresh is viceversa for valleys!

        If the method is interval/hybrid, then the intervals argument must be passed
        and it should be an instance of Intervals class.

        If the method is slope/hybrid, then the lookahead and avg_window
        arguments should be changed based on the application. 
        They have some default values though.

        The method returns:
        {"peaks":[[peak positions], [peak amplitudes]], 
        "valleys": [[valley positions], [valley amplitudes]]}
        """
        data = zip(x, y)
        x = np.array(x)

        if method == "slope" or method == "hybrid":
            peaks = {}
            result = slope.peaks(x, y, lookahead=lookahead, delta=valley_thresh)

            # find correspondences between peaks and valleys,
            # and set valleys are left and right Indices
            # see the other method(s) for clarity!

            peak_data = result["peaks"]
            valley_data = result["valleys"]

            # print len(peak_data[0]), len(peak_data[1])
            for i in xrange(len(peak_data[0])):
                nearest_index = find_nearest_index(valley_data[0], peak_data[0][i])
                if valley_data[0][nearest_index] < peak_data[0][i]:
                    left_index = find_nearest_index(
                        x, valley_data[0][nearest_index])
                    if (len(valley_data[0][nearest_index + 1:]) == 0):
                        right_index = find_nearest_index(
                            x, peak_data[0][i] + avg_window / 2.0)
                    else:
                        offset = nearest_index + 1
                        nearest_index = offset + \
                            find_nearest_index(
                                valley_data[0][offset:], peak_data[0][i])
                        right_index = find_nearest_index(
                            x, valley_data[0][nearest_index])
                else:
                    right_index = find_nearest_index(
                        x, valley_data[0][nearest_index])
                    if (len(valley_data[0][:nearest_index]) == 0):
                        left_index = find_nearest_index(
                            x, peak_data[0][i] - avg_window / 2.0)
                    else:
                        nearest_index = find_nearest_index(
                            valley_data[0][:nearest_index], peak_data[0][i])
                        left_index = find_nearest_index(
                            x, valley_data[0][nearest_index])

                pos = find_nearest_index(x, peak_data[0][i])
                # print x[pos], peak_data[1][i], x[left_index],
                # x[right_index]
                peaks[pos] = [peak_data[1][i], left_index, right_index]

                if method == "hybrid":
                    slope_peaks = peaks

        if method == "interval" or method == "hybrid":
            peaks = {}
            avg_window = np.average(intervals.intervals[1:] - intervals.intervals[:-1])
            first_center = (min(x) + 1.5 * avg_window) / avg_window * avg_window
            last_center = (max(x) - avg_window) / avg_window * avg_window
            if first_center < min(intervals.intervals[0]):
                first_center = intervals.intervals[0]
                warn("In the interval based approach, the first center was seen
                     to be too low and is set to " + str(first_center))
            if last_center > intervals.intervals[-1]:
                last_center = intervals.intervals[-1]
                warn("In the interval based approach, the last center was seen
                     to be too high and is set to " + str(last_center))

            interval = first_center
            prev_interval = first_center - avg_window
            # NOTE: All intervals are in cents. indices are of x/y
            while interval < last_center:
                if method == "ET":
                    left_index = find_nearest_index(
                        x, interval - avg_window / 2)
                    right_index = find_nearest_index(
                        x, interval + avg_window / 2)
                    interval += avg_window
                elif method == "JI" or method == "hybrid":
                    left_index = find_nearest_index(
                        x, (interval + prev_interval) / 2.0)
                    prev_interval = interval
                    interval = next_ji(interval)
                    right_index = find_nearest_index(
                        x, (interval + prev_interval) / 2.0)
                peak_pos = np.argmax(y[left_index:right_index])
                peak_amp = y[left_index + peak_pos]
                peaks[left_index + peak_pos] = [peak_amp, left_index, right_index]

                # print x[left_index], x[right_index], x[left_index+peak_pos], peak_amp
                # NOTE: All the indices (left/right_index, peak_pos) are to be changed to represent respective cent
                # value corresponding to the bin. Right now, they are indices of
                # respective x in the array.

        if method == "hybrid":
            # Mix peaks from slope method and JI method.
            p1 = slope_peaks.keys()
            p2 = peaks.keys()
            all_peaks = {}  # overwriting peaks dict
            for p in p1:
                near_index = find_nearest_index(p2, p)
                if abs(p - p2[near_index]) < avg_window / 2.0:
                    p2.pop(near_index)

            for p in p1:
                all_peaks[p] = slope_peaks[p]
            for p in p2:
                all_peaks[p] = peaks[p]
            peaks = all_peaks

        # Filter the peaks and retain eligible peaks, also get their valley points.

        # ----> peak_amp_thresh <---- : remove the peaks which are below that

        for pos in peaks.keys():
            # pos is an index in x/y. DOES NOT refer to a cent value.
            if peaks[pos][0] < peak_amp_thresh:
                # print "peak_amp: ", x[pos]
                peaks.pop(pos)

        # Check if either left or right valley is deeper than ----> valley_thresh
        # <----.
        valleys = {}
        for pos in peaks.keys():
            left_lobe = y[peaks[pos][1]:pos]
            right_lobe = y[pos:peaks[pos][2]]
            # Sanity check: Is it a genuine peak? Size of distributions on either
            # side of the peak should be comparable.
            if len(left_lobe) == 0 or len(right_lobe) == 0:
                continue
            if 1.0 * len(left_lobe) / len(right_lobe) < 0.15 or 1.0 * len(left_lobe) / len(right_lobe) > 6.67:
                # print "size: ", x[pos]
                # peaks.pop(pos)
                continue

            left_valley_pos = np.argmin(left_lobe)
            right_valley_pos = np.argmin(right_lobe)
            if (abs(left_lobe[left_valley_pos] - y[pos]) < valley_thresh and abs(right_lobe[right_valley_pos] - y[pos]) < valley_thresh):
                # print "valley: ", x[pos]
                peaks.pop(pos)
            else:
                valleys[peaks[pos][1] + left_valley_pos] = left_lobe[
                    left_valley_pos]
                valleys[pos + right_valley_pos] = right_lobe[right_valley_pos]

        if len(peaks) > 0:
            temp1 = np.array(peaks.values())
            temp1 = temp1[:, 0]

            return {'peaks': [x[peaks.keys()], temp1], 'valleys': [x[valleys.keys()], valleys.values()]}
        else:
            return {'peaks': [[], []], 'valleys': [[], []]}

    def extend_peaks(src_peaks, prop_thresh=30):
        """Each peak in src_peaks is checked for its presence in other octaves.
        If it does not exist, it is created. prop_thresh is the cent range within
        which the peak in the other octave is expected to be present, i.e., only
        if there is a peak within this cent range in other octaves, then the peak
        is considered to be present in that octave.
        """
        # octave propagation of the reference peaks
        temp_peaks = [i + 1200 for i in src_peaks["peaks"][0]]
        temp_peaks.extend([i - 1200 for i in src_peaks["peaks"][0]])
        extended_peaks = []
        extended_peaks.extend(src_peaks["peaks"][0])
        for i in temp_peaks:
            # if a peak exists around, don't add this new one.
            nearest_ind = find_nearest_index(src_peaks["peaks"][0], i)
            diff = abs(src_peaks["peaks"][0][nearest_ind] - i)
            diff = np.mod(diff, 1200)
            if diff > prop_thresh:
                extended_peaks.append(i)
        return extended_peaks
