#######################################################################
## RESISTIVE PULSE
#######################################################################
# Contains tools for opening, analyzing resistive pulse data
# Sections:
# 1. Imports
# 2. Constants
# 3. Functions
    # get_file_length()
    # get_data_atf()
    # get_data_raw()

# Imports
import csv
import numpy as np
from event import Event

# Constants
ATF_HEADER_LENGTH = 10          # Number of rows in axon text file (.atf) header

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
        stop = file_len(file_name)

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
    data = np.empty((stop-start,2))

    # Define start, stop points
    if start == -1:
        start = 0
    if stop == -1:
        stop = file_len(file_name)

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


def find_events_raw(file_name, start = -1, stop = -1, baseline_avg_length = 200,
                    event_avg_length = 5, trigger_sigma_threshold = 6,
                    max_search_length = 1000):
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
        - event_avg_length (optional): Number of data points to
          average within an event
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
    file_length = get_file_length(file_name)

    if start == -1:
        start = 0

    if stop == -1:
        stop = file_length

    if stop > file_length:
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
                                        event_avg=np.mean(data[start_index-
                                            event_avg_length:start_index,1])
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

                        events.append(Event(data, start_index,
                                            stop_index, baseline[1]))

                    # Check if exceeded max search length
                    if index-start_index >= max_search_length:
                        print 'e'
                        stop_trigger_found = True
                        index=start_index+max_search_length
                        baseline=get_baseline(data, index, baseline_avg_length,
                                              trigger_sigma_threshold)

                    index+=1

        except:
            print 'error!'
            continue


    return events
