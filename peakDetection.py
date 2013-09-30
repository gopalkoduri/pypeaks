#!/usr/bin/env python

import essentia.standard as es
import numpy as np

#the following two functions are taken from: 
#https://gist.github.com/1178136
def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise (ValueError, 
                'Input vectors y_axis and x_axis must have same length')
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis
    
def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200) 
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
    
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError, "Lookahead must be '1' or above in value"
    #NOTE: commented this to use the function with log(histogram)
    #if not (np.isscalar(delta) and delta >= 0):
    if not (np.isscalar(delta)):
        raise ValueError, "delta must be a positive number"
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]

def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value 
    in the array and returns its index."""
    arr = np.array(arr)
    index = (np.abs(arr - value)).argmin()
    return index

def next_ji(curJI):
    ji_intervals = np.array([-1200, -1089, -997, -885, -814, -702, -591, 
                   -499, -387, -316, -204, -112, 0, 111, 203, 315, 386, 
                   498, 609, 701, 813, 884, 996, 1088, 1200, 1311, 1403, 
                   1515, 1586, 1698, 1809, 1901, 2013, 2084, 2196, 2288, 
                   2400, 2511, 2603, 2715, 2786, 2898, 3009, 3101, 3213, 
                   3284, 3396, 3488, 3600])
    curIndex = np.where(ji_intervals == curJI)
    if curIndex[0][0] + 1 < len(ji_intervals):
        return ji_intervals[curIndex[0][0] + 1]
    else:
        raise IndexError("Increase number of octaves in ji_intervals!")

def nearest_ji(et_interval):
    ji_intervals = np.array([-1200, -1089, -997, -885, -814, -702, -591, 
                   -499, -387, -316, -204, -112, 0, 111, 203, 315, 386, 
                   498, 609, 701, 813, 884, 996, 1088, 1200, 1311, 1403, 
                   1515, 1586, 1698, 1809, 1901, 2013, 2084, 2196, 2288, 
                   2400, 2511, 2603, 2715, 2786, 2898, 3009, 3101, 3213, 
                   3284, 3396, 3488, 3600])
    if et_intervals < ji_intervals[0] - 25 or et_interval > ji_intervals[-1] + 25:
        raise IndexError("Increase number of octaves in ji_intervals!")

    index = find_nearest_index(ji_intervals, et_interval)
    return ji_intervals[index]

def peaks_by_slope(y, x, lookahead=20, delta=0.00003, 
                  average_hist=False, ref_peaks=None, ref_thresh=25):
    # In the average histogram, we get the ref_peaks and ref_valleys which are
    # then used to clean the peaks we get from a single performance. For
    # average histogram, lookahead should be ~ 20, delta = 0.000025 to pick up
    # only the prominent peaks. For histogram of a performance, lookahead
    # should be ~=15, delta = 0.00003 to pick up even the little peaks.
    _max, _min = peakdetect(y, x, lookahead, delta)
    x_peaks = [p[0] for p in _max]
    y_peaks = [p[1] for p in _max]
    x_valleys = [p[0] for p in _min]
    y_valleys = [p[1] for p in _min]
    if average_hist:
        ref_peaks = [x_peaks, y_peaks]
        ref_valleys = [x_valleys, y_valleys]
        return {"peaks": ref_peaks, "valleys": ref_valleys}
    else:
        if not ref_peaks:
            print "Reference peaks are not provided. Quitting."
            return
        x_clean_peaks = []
        y_clean_peaks = []
        # octave propagation of the reference peaks
        prop_thresh = 30  # NOTE: Hardcoded
        temp_peaks = [i + 1200 for i in ref_peaks[0]]
        temp_peaks.extend([i - 1200 for i in ref_peaks[0]])
        extended_peaks = []
        extended_peaks.extend(ref_peaks[0])
        for i in temp_peaks:
            # if a peak exists around, don't add this new one.
            nearest_ind = find_nearest_index(ref_peaks[0], i)
            diff = abs(ref_peaks[0][nearest_ind] - i)
            diff = np.mod(diff, 1200)
            if diff > prop_thresh:
                extended_peaks.append(i)
        # print extended_peaks
        for peak_location_index in xrange(len(x_peaks)):
            ext_peak_location_index = find_nearest_index(
                extended_peaks, x_peaks[peak_location_index])
            diff = abs(
                x_peaks[peak_location_index] - extended_peaks[ext_peak_location_index])
            diff = np.mod(diff, 1200)
            # print x_peaks[peak_location_index],
            # extended_peaks[ext_peak_location_index], diff
            if diff < ref_thresh:
                x_clean_peaks.append(x_peaks[peak_location_index])
                y_clean_peaks.append(y_peaks[peak_location_index])
        return {"peaks": [x_clean_peaks, y_clean_peaks], "valleys": [x_valleys, y_valleys]}


def peaks(y, x, method="JI", window=100, peak_amp_thresh=0.00005, valley_thresh=0.00003):
    """
    This function expects smoothed histogram (i.e., y).

    method can be JI/ET/slope/hybrid.
    JI and ET methods do not use generic peak detection algorithm. They use intervals and window
    to pick up the local maximum, and later filter out irrelevant peaks by using empirically found
    thresholds. Slope approach uses generic peak picking algorithm which first finds peaks by slope
    and then applies bounds. Hybrid approach first finds peaks using generic peak detection algo, 
    then filters the peaks heuristically as in JI/ET.
    
    window refers to the cent range used while picking up the maxima.

    The method returns:
    {"peaks":[[peak positions], [peak amplitudes]], "valleys": [[valley positions], [valley amplitudes]]}
    """
    data = zip(x, y)
    x = np.array(x)
    first_center = (min(x) + 1.5 * window) / window * window
    last_center = (max(x) - window) / window * window
    if first_center < -1200:
        first_center = -1200
    if last_center > 3600:
        last_center = 3600

    if method == "slope" or method == "hybrid":
        peaks = {}
        peak_info = peaks_by_slope(
            y, x, lookahead=20, delta=valley_thresh, average_hist=True)

        # find correspondences between peaks and valleys, and set valleys are left and right Indices
        # see the other method(s) for clarity!

        peak_data = peak_info["peaks"]
        valley_data = peak_info["valleys"]

        # print len(peak_data[0]), len(peak_data[1])
        for i in xrange(len(peak_data[0])):
            nearest_index = find_nearest_index(valley_data[0], peak_data[0][i])
            if valley_data[0][nearest_index] < peak_data[0][i]:
                left_index = find_nearest_index(
                    x, valley_data[0][nearest_index])
                if (len(valley_data[0][nearest_index + 1:]) == 0):
                    right_index = find_nearest_index(
                        x, peak_data[0][i] + window / 2.0)
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
                        x, peak_data[0][i] - window / 2.0)
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

    if method == "JI" or method == "ET" or method == "hybrid":
        peaks = {}
        # Obtain max value per interval
        if method == "JI" or method == "hybrid":
            first_center = nearest_ji(first_center)
            last_center = nearest_ji(last_center)

        interval = first_center
        prev_interval = first_center - window
        # NOTE: All *intervals are in cents. *indices are of x/y
        while interval < last_center:
            if method == "ET":
                left_index = find_nearest_index(
                    x, interval - window / 2)
                right_index = find_nearest_index(
                    x, interval + window / 2)
                interval += window
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
            if abs(p - p2[near_index]) < window / 2.0:
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
    """Each peak in src_peaks is checked for its presence in other octaves. If it does not exist, it is created. prop_thresh is the cent range within which the peak in the other octave is expected to be present, i.e., only if there is a peak within this cent range in other octaves, then the peak is considered to be present in that octave.
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
