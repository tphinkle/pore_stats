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


import numpy as np
import csv
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter


# Constants
ATF_HEADER_LENGTH = 10          # Number of rows in axon text file (.atf) header

import numpy as np
from copy import copy

class ResistivePulseEvent:

    def __init__(self, data, start, stop, baseline_avg):
        # Declare member variables
        self._data=None
        self._start_index=None
        self._stop_index=None
        self._length=None
        self._duration=None
        self._amplitude=None
        self._baseline_avg=None

        self._extrema=None
        self._maxima=None
        self._minima=None

        # Initialize member variables
        self._data = data[start:stop,:]
        self._start_index = start
        self._stop_index = stop
        self._length = self._stop_index - self._start_index
        self._duration = self._data[-1,0]-self._data[0,0]
        self._amplitude = self._data[:, 1].max()-self._data[:, 1].min()
        self._baseline_avg = baseline_avg
        return



    def set_extrema(self, minima, maxima):
        self._extrema=minima+maxima
        self._extrema.sort()
        self._maxima=maxima
        self._minima=minima
        return


def get_file_length(file_name):
    """
    * Description: Get total number of rows in file
    * Return: # rows in file (Int)
    * Arguments:
        - file_name: Name of desired file
    """

    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
        f.close()
        return i+1

def atf_to_raw(file_name, current_column):
    output_file_name = file_name.split('.')[0]


    # Open file
    file_handle=open(file_name, 'r')
    output_file_handle=open(output_file_name, 'w')
    csv_reader=csv.reader(file_handle, delimiter='\t', quotechar='|')

    # Skip header
    for i in range(0, ATF_HEADER_LENGTH):
        row=csv_reader.next()

    # Read data
    while row:
        row=csv_reader.next()
        t=row[0]
        I=row[current_column]
        output_file_handle.write(str(t)+"\t"+str(I)+"\n")

    # Close file
    file_handle.close()
    output_file_handle.close()

    # Return
    return


def get_data_atf(file_name, current_column, start=-1, stop=-1):
    """
    * Description: Opens .atf file
    * Return: Time, Current data (numpy array: 'data')
    * Arguments:
        - file_name: Name of desired file to open
        - current_column: Column of interest (usually column containing current information; time column automatically loaded)
        - start (optional): Starting row to load
        - stop (optional): Last row to load
    """

    # Initialize numpy array that will be returned
    data = np.empty((stop-start,2))

    # Define start, stop points
    if start == -1:
        start = 0
    if stop == -1:
        stop = get_file_length(file_name)

    # Open file
    file_handle=open(file_name, 'r')
    csv_reader=csv.reader(file_handle, delimiter='\t', quotechar='|')

    # Skip header
    for i in range(0, ATF_HEADER_LENGTH):
        row=csv_reader.next()

    # Skip to 'start' row
    for i in range(0, start):
        row=csv_reader.next()

    # Read data
    for i in range(0, stop-start):
        row=csv_reader.next()
        data[i,0]=row[0]
        data[i,1]=row[current_column]

    # Close file
    file_handle.close()

    # Return
    return data


def get_data_raw(file_name, start=-1, stop=-1):
    """
    * Description: Opens raw data file, a data file consisting of only two columns: time and current
    * Return: Time, Current data (numpy array: 'data')
    * Arguments:
        - file_name: Name of desired file to open
        - start (optional): Starting row to load
        - stop (optional): Last row to load
    """

    # Initialize numpy array that will be returned


    # Define start, stop points
    if start == -1:
        start = 0
    if stop == -1:
        stop = get_file_length(file_name)

    data = np.empty((stop-start,2))

    # Open file
    file_handle=open(file_name, 'r')
    csv_reader=csv.reader(file_handle, delimiter='\t', quotechar='|')

    # Skip to 'start' row
    for i in range(0, start):
        row=csv_reader.next()

    # Read data
    for i in range(0, stop-start):
        row=csv_reader.next()
        data[i,0]=row[0]
        data[i,1]=row[1]

    # Close file
    file_handle.close()

    # Return
    return data


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

    if start < 0:
        start = 0
    baseline_avg=np.mean(data[start:start+baseline_avg_length,1])
    baseline_sigma=np.std(data[start:start+baseline_avg_length,1])
    trigger_threshold=trigger_sigma_threshold*baseline_sigma
    baseline=np.array([data[start,0], baseline_avg,
                        baseline_avg-trigger_threshold,
                        baseline_avg+trigger_threshold], dtype=float)

    return baseline

def plot_first_n_events(file_name, trigger_sigma_threshold, n = 5):
    events_found = 0
    events = []
    start = 0
    interval = 1000000
    stop = interval
    while events_found <= n:

        data = get_data_raw(file_name, start, stop)
        events = find_events_raw(file_name, start=start, stop=stop,
                                 trigger_sigma_threshold=trigger_sigma_threshold)
        print '# of events found!', len(events)
        for event in events:

            events_found +=1
            if events_found <= n:
                xi=event._start_index - 200
                if xi < 0:
                    xi = 0
                xf=event._stop_index + 200
                plt.plot(data[xi:xf,0], data[xi:xf,1])
                plt.plot(event._data[:,0], event._data[:,1], c = (1.,0,0))
                plt.xlim(data[xi,0], data[xf,0])
                plt.show()

        start += interval
        stop += interval

    return




def find_events_raw(file_name, start = -1, stop = -1, baseline_avg_length = 200,
                    trigger_sigma_threshold = 6, max_search_length = 1000):
    """
    * Description: Finds events within a full resistive pulse file
      (.raw file only)
    * Return: List of resistive pulse Events
      (list [] of Event: 'events')
    * Arguments:
        - file_name: Name of desired file to find events in
        - start: Data point to start search
        - stop: Data point to stop search
        - baseline_avg_length (optional): Number of data points to
          average baseline over
        - trigger_sigma_threshold (optional): Number of standard
          deviations above or below baseline for an event to be
          triggered
        - max_search_length (optional): Max number of data points
          searched before abandoning search for end of event
    """

    # Search parameters
    segment_length = 10000000

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
        file_length = get_file_length(file_name)
        stop = file_length - 1


    if start >= stop:
        print 'Check start, stop points!'
        return

    # Loop over all file segments
    segments=(stop-start)/segment_length
    for i in range(0, segments + 1):
        print 'starting segment:', i, '/', segments

        # Open file segment
        segment_start = i*segment_length
        segment_stop = segment_start + segment_length
        if segment_stop > stop:
            segment_stop = stop

        data = get_data_raw(file_name, segment_start, segment_stop)

        # Begin looking for events in segment
        try:
            index=0
            baseline=get_baseline(data, index, baseline_avg_length,
            trigger_sigma_threshold)
            baseline_history=np.vstack((baseline_history, baseline))

            while True:

                # Look for event start
                start_trigger_found = False
                while start_trigger_found == False:

                    # Check if current exceeds threshold
                    if ((data[index, 1] < baseline[2])
                            or (data[index, 1] > baseline[3])):
                        # Update baseline (i.e., in case of drift)
                        baseline=get_baseline(data, index-2*baseline_avg_length,
                                              baseline_avg_length,
                                              trigger_sigma_threshold)
                        baseline_history=np.vstack((baseline_history, baseline))

                        # Check if point still exceeds trigger threshold after
                        # updating baseline
                        if ((data[index, 1] < baseline[2])
                                or (data[index, 1] > baseline[3])):
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
                            continue

                    index += 1

                # Look for event stop
                stop_trigger_found = False
                while stop_trigger_found == False:
                    # Check if in baseline; to return to baseline, 10
                    # data points in a row must exceed the trigger threshold
                    in_baseline = True
                    for j in range(10):
                        if ((data[index, 1] < baseline[2])
                                or (data[index, 1] > baseline[3])):
                            in_baseline = False

                    # Check if return to baseline
                    if in_baseline == True:
                        print 'c', index
                        stop_index = index

                        # Requirement for reentry into baseline is that
                        # current returns to value half-way between
                        # the trigger value and the baseline average
                        # value
                        reentry_threshold=(baseline[1] + baseline[2])/2.
                        while stop_trigger_found == False:
                            if abs(data[stop_index,1])>=abs(reentry_threshold):
                                stop_trigger_found = True
                            else:
                                stop_index+=1
                        index=stop_index
                        event_indices=np.vstack((event_indices,
                                      np.array([start_index, stop_index])))
                        events.append(ResistivePulseEvent(data, start_index,
                                                          stop_index,
                                                          baseline[1]))

                        print 'event #', len(events)
                        print '\trange = ', start_index, stop_index

                    # Check if exceeded max search length
                    if index-start_index >= max_search_length:
                        stop_trigger_found = True
                        index=start_index+max_search_length
                        baseline=get_baseline(data, index, baseline_avg_length,
                                              trigger_sigma_threshold)

                    index+=1

        except:
            print 'error!'
            continue


    return events

def get_maxima_minima(data, sigma=0, refine_length=0):
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
        smoothed_data = gaussian_filter(data, sigma=sigma)

    # Find extrema
    maxima=[]
    minima=[]

    # Calculate first derivatives everywhere
    d_data=[((smoothed_data[(i+1)%len(smoothed_data)]-smoothed_data[i])+
            (smoothed_data[i]-smoothed_data[i-1]))/2.
            for i in range(len(smoothed_data))]


    derivative_sign=(1*int(smoothed_data[1]>smoothed_data[0])-
                     1*int(smoothed_data[1]<=smoothed_data[0]))
    for i in range(smoothed_data.shape[0]-1):
        new_derivative_sign=(1*int(smoothed_data[i+1]>smoothed_data[i])-
                            1*int(smoothed_data[i+1]<=smoothed_data[i]))
        if new_derivative_sign != derivative_sign:
            # Found derivative = 0, calculate 2nd derivative to see if min/max
            d2=(((d_data[(i+1)%len(d_data)]-d_data[i])+
               (d_data[i]-d_data[i-1]))/2.)
            if d2<=0:
                maxima.append(i)
            else:
                minima.append(i)
        derivative_sign=new_derivative_sign

    # Refine maxima/minima
    new_maxima=[]
    for i, maximum in enumerate(maxima):
        temp_data = np.array(data[maximum-
                                  refine_length:maximum+refine_length+1])
        print np.argsort(temp_data)
        new_maxima.append(maximum + (np.argsort(temp_data)[-1]-refine_length))
        maxima[i] = new_maxima
    maxima=new_maxima

    new_minima=[]
    for i, minimum in enumerate(minima):
        temp_data = np.array(data[minimum-
                                  refine_length:minimum+refine_length+1])
        new_minima = minimum + (np.argsort(temp_data)[0]-refine_length)
        minima[i] = new_minima


    return maxima, minima
