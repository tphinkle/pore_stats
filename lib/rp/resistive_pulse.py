"""

RESISTIVE PULSE

* Contains tools for opening, analyzing resistive pulse data

* Sections:
    1. Imports
    2. Constants
    3. Classes
        - ResistivePulseEvent
    4. Functions
        - get_file_length()
        - get_data_atf()
        - get_data_raw()
        - get_baseline()
        - find_events_raw()
        - get_maxima_minima()

"""

# Imports

# Standard library
import sys
import os
PORE_STATS_BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__)).replace('/lib/rp', '')
print PORE_STATS_BASE_DIRECTORY
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib')
import copy

import json
import csv
from array import array
import struct
from itertools import islice


# Program specific
#import time_series as ts
import rp_file

# Scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters
import scipy.signal




class ResistivePulseEvent:
    """
    A resistive pulse event. Contains the time and current values for every data point
    in the event, as well as additional information about the event such as its
    minima and maxima locations, amplitude, duration, etc.
    """

    def __init__(self, data, baseline, id = None):
        # Declare member variables
        self._id = id

        self._data = None
        self._baseline = None

        self._extrema = None
        self._maxima = None
        self._minima = None

        # Initialize member variables
        self._data = data[:,:]
        self._duration = data[-1,0] - data[0,0]
        self._baseline = baseline

        # Needs to be fixed for negative events.
        self._amplitude = abs(self._baseline[1] - np.min(self._data[:,1]))

        return



    def set_extrema(self, maxima, minima):
        """
        * Description:
            - Sets the event's maxima, minima, and extrema (combined list of the two)
        * Return:
        * Arguments:
            - maxima: list [] of time indices of the positions of the maxima
            - minima: list [] of time indices of the positions of the maxima
        """

        self._extrema=maxima[:] + minima[:]
        self._extrema.sort()
        self._maxima=maxima[:]
        self._minima=minima[:]
        return










def get_baseline(data, start, baseline_avg_length, trigger_sigma_threshold):
    """
    * Description: Gets baseline array, an array containing time,
      average current value, positive threshold value, and negative
      threshold values. Called in 'find_events_raw' function.
    * Return: Returns baseline array (1-d numpy array: 'baseline')
    * Arguments:
        - data: numpy array containing time, current data
        - start: Data point at which to begin average
        - baseline_avg_length: Number of data points to average
          baseline over
        - trigger_sigma_threshold: Number of std dev above/below
          baseline average for an event to be detected
    """
    stop = start + baseline_avg_length
    if start < 0:
        start = 0
    baseline_avg=np.mean(data[start:stop+1,1])
    baseline_sigma=np.std(data[start:stop+1,1])
    trigger_threshold=trigger_sigma_threshold*baseline_sigma
    baseline=np.array([data[start,0], baseline_avg,
                        baseline_avg-trigger_threshold,
                        baseline_avg+trigger_threshold], dtype=float)

    return baseline







def get_sampling_frequency(data):
    """
    * Description:
        - Returns the sampling frequency from data.
    * Return:
        - Returns the sampling frequency.
    * Arguments:
        - data: numpy array of data, with first dimension (column) containing time
    """
    return int(1./(data[1,0]-data[0,0]))





def get_maxima_minima(data, sigma=0, refine_length=0, num_maxima = 0, num_minima = 0, return_by = 'accel'):
    """
    * Description: Finds the maxima, minima within 2-d, evenly sampled data.
        - First, applies a Gaussian filter (scipy) to the data to smooth it
        - Calculates first and second derivatives of data at each point
        - Extrema are detected via a change in sign of first derivatives
        - Maximum/minimum is determined by sign of second derivative
    * Return: Lists of max and mins indices (Two lists of floats: 'maxima',
      'minima')
    * Arguments:
        - data: 1-d array of data to be searched for extrema
        - sigma (optional): Standard deviation of Gaussian kernel. If
          unspecified, data will be unfiltered.
        - refine_length (optional): If a data point adjacent to a determined
          extremum within +/- refine_length has a larger value than the
          extremum, it will be replaced by that point.
    """

    # Smooth data
    if sigma != 0:
        smoothed_data = scipy.ndimage.filters.gaussian_filter(data, sigma=sigma)
    else:
        smoothed_data = np.array(data)

    # Find extrema
    maxima=[]
    d2_maxima = []
    minima=[]
    d2_minima = []

    # Calculate first derivatives everywhere--derivative defined as average of slopes of
    # point-in-question to left neightbor and right neighbor to left neighbor.
    d_data=[((smoothed_data[i]-smoothed_data[i-1])/2.+
            (smoothed_data[i]-smoothed_data[i-1]))/2.
            for i in xrange(0,len(smoothed_data)-1)]


    derivative_sign=(1*int(smoothed_data[1]>smoothed_data[0])-
                     1*int(smoothed_data[1]<=smoothed_data[0]))
    for i in xrange(0,smoothed_data.shape[0]-1):
        new_derivative_sign=(1*int(smoothed_data[i+1]>smoothed_data[i])-
                            1*int(smoothed_data[i+1]<=smoothed_data[i]))
        if new_derivative_sign != derivative_sign:
            # Found derivative = 0, calculate 2nd derivative to see if min/max
            d2=(((d_data[i]-d_data[i-1])+
               (d_data[(i+1)%len(d_data)]-d_data[i-1])/2.)/2.)

            if d2<=0:
                d2_maxima.append(abs(d2))
                maxima.append(i)
            else:
                d2_minima.append(abs(d2))
                minima.append(i)
        derivative_sign=new_derivative_sign

    # Refine maxima/minima


    for i, maximum in enumerate(maxima):
        temp_data = np.array(data[maximum-
                                  refine_length:maximum+refine_length+1])
        if temp_data.shape[0] > 0:
            maxima[i] = maximum + (np.argsort(temp_data)[-1]-refine_length)

    for i, minimum in enumerate(minima):
        temp_data = np.array(data[minimum-
                                  refine_length:minimum+refine_length+1])

        if temp_data.shape[0] > 0:
            minima[i] = minimum + (np.argsort(temp_data)[0]-refine_length)


    # Convert to np array for easier indexing.
    minima = np.array(minima, dtype = int)
    maxima = np.array(maxima, dtype = int)
    d2_minima = np.array(d2_minima)
    d2_maxima = np.array(d2_maxima)



    # Determine which maxima to return
    if num_maxima > 0:
        new_maxima = []
        if return_by == 'accel':
            maxima = maxima[np.argsort(d2_maxima)[-num_maxima:]]
        elif return_by == 'high':
            maxima = maxima[np.argsort(smoothed_data[maxima])[-num_maxima:]]
        elif return_by == 'low':
            maxima = maxima[np.argsort(smoothed_data[maxima])[:num_maxima]]
        else:
            raise TypeError('keyword ' + return_by + ' is not recognized; valid keywords are: ["accel", "high", "low"]')

    # Determine which minima to return
    if num_minima > 0 and minima.shape[0] > 1:
        new_minima = []
        if return_by == 'accel':
            minima = minima[np.argsort(d2_minima)[-num_minima:]]
        elif return_by == 'high':
            minima = minima[np.argsort(smoothed_data[minima])[-num_minima:]]
        elif return_by == 'low':
            minima = minima[np.argsort(smoothed_data[minima])][:num_minima]
        else:
            raise TypeError('keyword ' + return_by + ' is not recognized; valid keywords are: ["accel", "high", "low"]')

    return maxima, minima


def filter_events_length(events, length):
    """
    * Description:
        - Takes in a list of ResistivePulseEvent and returns a new filtered list
        - Filters events based on their length/duration
    * Return:
        - filtered_events: List [] of ResistivePulseEvent that has been filtered
    * Arguments:
        - events: List [] of ResistivePulseEvent to be filtered
        - length: Threshold value for the number of data points; events with < length
        data points are filtered out.
    """
    filtered_events = [event for event in events if event._data.shape[0] >= length]
    return filtered_events














"""""""""""""""""""""""""""""""""
Obsolete functions
"""""""""""""""""""""""""""""""""
"""
def get_baseline_1d(data, start, baseline_avg_length, trigger_sigma_threshold):
    stop = start + baseline_avg_length
    if start < 0:
        start = 0
    baseline_avg = 1.*sum(data[start:stop])/(stop-start)
    baseline_sigma = sum([(pt-baseline_avg)**2. for pt in data[start:stop]])**.5/(start-stop)
    trigger_threshold = trigger_sigma_threshold*baseline_sigma
    baseline = [start, baseline_avg, baseline_avg - trigger_threshold, baseline_avg + trigger_threshold]
    return baseline


def get_full_baseline(data, baseline_avg_length, trigger_sigma_threshold):

    * Description:
    * Return:
    * Arguments:
        -

    baseline = np.empty((0,4))
    start = 0
    stop = baseline_avg_length
    while stop < data.shape[0]:
        baseline = np.vstack((baseline, get_baseline(data, start,\
                              baseline_avg_length, trigger_sigma_threshold)))
        start+=baseline_avg_length
        stop+=baseline_avg_length

    return baseline
"""
"""
def open_event_file(file_path):
    events = []

    f = open(file_path, 'r')
    reader = csv.reader(f, delimiter = '\t')
    row = 5
    try:
        while row:
            row = reader.next()
            if row[0] == 'event#':
                baseline = []
                baseline.append(float(row[5]))
                baseline.append(float(row[6]))
                baseline.append(float(row[7]))
                baseline.append(float(row[8]))
                length = int(row[3])
                data = np.empty((length, 2), dtype = float)
                for i in xrange(length):
                    row = reader.next()
                    data[i,0] = float(row[0])
                    data[i,1] = float(row[1])
                events.append(ResistivePulseEvent(data, baseline))
    except:
        pass
    return events

    def find_events_data(search_data, raw_data = None, start = -1, stop = -1, baseline_avg_length = 500,
                        trigger_sigma_threshold = 6, max_search_length = 5000,
                        go_past_length = 0):

    * Description: Finds events within a full resistive pulse file
      (.raw file only)
    * Return: List of resistive pulse Events
      (list [] of Event: 'events')
    * Arguments:
        - file_path: Name of desired file to find events in
        - start: Data point to start search
        - stop: Data point to stop search
        - baseline_avg_length (optional): Number of data points to
          average baseline over
        - trigger_sigma_threshold (optional): Number of standard
          deviations above or below baseline for an event to be
          triggered
        - max_search_length (optional): Max number of data points
          searched before abandoning search for end of event

    # Search parameters
    segment_length = 100000000000

    # numpy array to hold baseline info:
    # [time, average current, positive threshold, negative threshold]
    baseline_history=np.empty((0,4), dtype=float)

    # numpy array to hold event start, stop indices
    event_indices=np.empty((0,2), dtype=int)

    # List that will store events and be returned
    events=[]

    # Get file length; define start, stop points; check points
    if start == -1:
        start = 0

    if stop == -1:
        stop = search_data.shape[-1]

    if stop > search_data.shape[-1] - 1:
        stop = search_data.shape[-1] - 1

    if start >= stop:
        print 'Check start, stop points!'
        return

    if raw_data == None:
        raw_data = search_data

    original_segments = []



    try:
        index = 0

        baseline=get_baseline(search_data, index, baseline_avg_length,
        trigger_sigma_threshold)
        baseline_history=np.vstack((baseline_history, baseline))

        while True:

            # Look for event start

            start_trigger_found = False
            while start_trigger_found == False:


                # Check if current exceeds threshold
                search_data[index]
                if ((search_data[index:index+4,1].mean() < baseline[2])
                    or (search_data[index:index+4,1].mean() > baseline[3])):

                    # Update baseline (i.e., in case of drift)
                    baseline=get_baseline(search_data, index-2*baseline_avg_length,
                                          baseline_avg_length,
                                          trigger_sigma_threshold)
                    baseline_history=np.vstack((baseline_history, baseline))

                    # Check if point still exceeds trigger threshold after
                    # updating baseline
                    if ((search_data[index:index+4,1].mean() < baseline[2])
                        or (search_data[index:index+4,1].mean() > baseline[3])):
                            # Trigger, get first point to exit baseline
                            start_index = index

                            # Requirement for reentry into baseline is that
                            # current returns to value half-way between
                            # the trigger value and the baseline average
                            # value
                            reentry_threshold = (baseline[1]+baseline[2])/2.
                            while start_trigger_found == False:

                                # Check if data point at start_index
                                # passes reentry_threshold
                                if abs(search_data[start_index,1]) >= abs(reentry_threshold):
                                    start_trigger_found = True
                                else:
                                    start_index-=1

                    else:
                        pass

                index += 1

            # Look for event stop
            stop_trigger_found = False
            while stop_trigger_found == False:


                in_baseline = False

                if ((search_data[index:index+4,1].mean() > baseline[2])
                    and (search_data[index:index+4,1].mean() < baseline[3])):
                        in_baseline = True

                # Check if return to baseline
                if in_baseline == True:


                    stop_index = index

                    # Requirement for reentry into baseline is that
                    # current returns to value half-way between
                    # the trigger value and the baseline average
                    # value
                    reentry_threshold=(baseline[1]+baseline[2])/2.
                    while stop_trigger_found == False:

                        if abs(search_data[stop_index,1])>=abs(reentry_threshold):

                            stop_trigger_found = True
                            stop_index = stop_index + go_past_length
                        else:
                            stop_index+=1

                    index=stop_index
                    event_indices=np.vstack((event_indices,
                                  np.array([start_index, stop_index])))

                    events.append(ResistivePulseEvent(raw_data[start_index:stop_index+1,:],
                                                      baseline))

                    print 'event # ', event_indices.shape[0], 't_i = ', search_data[start_index,0]
                    #print start_index, ',', stop_index, events[-1]._data[0,0], ',', events[-1]._data[-1,0]

                    # Replace event with baseline
                    original_segments.append(search_data[start_index:stop_index+1,:])
                    search_data[start_index:stop_index,1]=search_data[start_index-(stop_index-start_index)-baseline_avg_length:start_index-baseline_avg_length,1]



                # Check if exceeded max search length
                elif index-start_index >= max_search_length:
                    stop_trigger_found = True
                    index = start_index+max_search_length
                    baseline = get_baseline(search_data, index, baseline_avg_length,
                                          trigger_sigma_threshold)


                index += 1


    except:
        print 'error!'
        print index

    # Replace altered segments with original segments

    sampling_frequency = ts.get_sampling_frequency(search_data)
    for i, segment in enumerate(original_segments):
        i_i = ts.get_index_from_time(segment[0,0], sampling_frequency)
        i_f = ts.get_index_from_time(segment[-1,0], sampling_frequency)
        search_data[i_i:i_f+1,:] = segment[:,:]

    print 'found', event_indices.shape[0], 'events!'
    return events


    def find_events(file_path, start = -1, stop = -1, baseline_avg_length = 500,
                        trigger_sigma_threshold = 6, max_search_length = 5000,
                        go_past_length = 0, f_cutoff=-1):

    * Description: Finds events within a full resistive pulse file
      (.raw file only)
    * Return: List of resistive pulse Events
      (list [] of Event: 'events')
    * Arguments:
        - file_path: Name of desired file to find events in
        - start: Data point to start search
        - stop: Data point to stop search
        - baseline_avg_length (optional): Number of data points to
          average baseline over
        - trigger_sigma_threshold (optional): Number of standard
          deviations above or below baseline for an event to be
          triggered
        - max_search_length (optional): Max number of data points
          searched before abandoning search for end of event

        # Search parameters
        segment_length = 100000000000

        # numpy array to hold baseline info:
        # [time, average current, positive threshold, negative threshold]
        baseline_history=np.empty((0,4), dtype=float)

        # numpy array to hold event start, stop indices
        event_indices=np.empty((0,2), dtype=int)

        # List that will store events and be returned
        events=[]

        # Get file length; define start, stop points; check points

        file_length = rp_file.get_file_length(file_path)

        if start == -1:
            start = 0

        if stop == -1:
            stop = file_length - 1

        if stop > file_length - 1:
            stop = file_length - 1

        if start >= stop:
            print 'Check start, stop points!'
            return

        # Loop over all file segments
        num_segments=(stop-start)/segment_length


        for i in xrange(0, num_segments + 1):

            # Open file segment
            segment_start = start + i*segment_length
            segment_stop = segment_start + segment_length
            if segment_stop > stop:
                segment_stop = stop

            raw_data = rp_file.get_data(file_path, segment_start, segment_stop)

            if f_cutoff > 0:
                f_nyquist = get_sampling_frequency(raw_data)/2.
                filter_b, filter_a = butter(N=5, Wn=f_cutoff/f_nyquist, btype = 'low', analog = False)
                data = np.hstack((raw_data[:,0].reshape(-1,1), lfilter(filter_b, filter_a, raw_data[:,1]).reshape(-1,1)))
            else:
                data = raw_data[:,:]

            # Begin looking for events in segment
            try:
                if f_cutoff > 0:
                    index = 1000
                else:
                    index = 0

                baseline=get_baseline(data, index, baseline_avg_length,
                trigger_sigma_threshold)
                baseline_history=np.vstack((baseline_history, baseline))

                while True:

                    # Look for event start

                    start_trigger_found = False
                    while start_trigger_found == False:


                        # Check if current exceeds threshold
                        a=data[index]
                        if ((data[index:index+4,1].mean() < baseline[2])
                            or (data[index:index+4,1].mean() > baseline[3])):
                            # Update baseline (i.e., in case of drift)
                            baseline=get_baseline(data, index-2*baseline_avg_length,
                                                  baseline_avg_length,
                                                  trigger_sigma_threshold)
                            baseline_history=np.vstack((baseline_history, baseline))

                            # Check if point still exceeds trigger threshold after
                            # updating baseline
                            if ((data[index:index+4,1].mean() < baseline[2])
                                or (data[index:index+4,1].mean() > baseline[3])):
                                    # Trigger, get first point to exit baseline
                                    start_index = index

                                    # Requirement for reentry into baseline is that
                                    # current returns to value half-way between
                                    # the trigger value and the baseline average
                                    # value
                                    reentry_threshold = (baseline[1]+baseline[2])/2.
                                    while start_trigger_found == False:

                                        # Check if data point at start_index
                                        # passes reentry_threshold
                                        if abs(data[start_index,1]) >= abs(reentry_threshold):
                                            start_trigger_found = True
                                        else:
                                            start_index-=1

                            else:
                                pass

                        index += 1

                    # Look for event stop
                    stop_trigger_found = False
                    while stop_trigger_found == False:


                        in_baseline = False

                        if ((data[index:index+4,1].mean() > baseline[2])
                            and (data[index:index+4,1].mean() < baseline[3])):
                                in_baseline = True

                        # Check if return to baseline
                        if in_baseline == True:

                            stop_index = index

                            # Requirement for reentry into baseline is that
                            # current returns to value half-way between
                            # the trigger value and the baseline average
                            # value
                            reentry_threshold=(baseline[1]+baseline[2])/2.
                            while stop_trigger_found == False:

                                if abs(data[stop_index,1])>=abs(reentry_threshold):
                                    stop_trigger_found = True
                                    stop_index = stop_index + go_past_length
                                else:
                                    stop_index+=1
                            index=stop_index
                            event_indices=np.vstack((event_indices,
                                          np.array([start_index, stop_index])))

                            events.append(ResistivePulseEvent(raw_data[start_index:stop_index,:],
                                                              baseline))

                            print 'event # ', len(events)
                            print start_index, ',', stop_index, events[-1]._data[0,0], ',', events[-1]._data[-1,0]

                            # Replace event with baseline
                            data[start_index:stop_index,:]=data[start_index-(stop_index-start_index)-baseline_avg_length:start_index-baseline_avg_length,:]



                        # Check if exceeded max search length
                        elif index-start_index >= max_search_length:
                            stop_trigger_found = True
                            index=start_index+max_search_length
                            baseline=get_baseline(data, index, baseline_avg_length,
                                                  trigger_sigma_threshold)


                        index+=1


            except:
                print 'error!'
                print index
                continue

        print 'found', len(events), 'events!'
        return events
"""
