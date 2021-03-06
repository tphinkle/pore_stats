"""
- Contains all functions related to opening resistive pulse files
- Supported formats: .atf, .bts, .abf*
"""


# Imports

# Standard library
import csv
from array import array
import struct
from itertools import islice
import copy
import os.path
import json
import sys
import os


# Scipy
import numpy as np

# Program specific
PORE_STATS_BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/pore_stats')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/pore_stats/rp/python-neo-master/')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/pore_stats/rp/python-neo-master/neo')
import resistive_pulse
import rp_file
import neo.io


# Constants
ATF_HEADER_LENGTH_GAP_FREE = 10
ATF_HEADER_LENGTH_EPISODIC_STIMULATION = 11

BTS_HEADER_BYTES = 4
BTS_ELEMENT_BYTES = 8


class RPFile(object):

    """
    Class that describes an active resistive pulse data set.
    """
    def __init__(self, file_path = None):
        self._file_path = None

        self._directory = None

        self._file_type = None

        self._file_length = None

        if file_path:
            self.set_file(file_path)

    def set_file(self, file_path):
        """
        * Description: Initializes the class parameters based off the file at file_path.
        * Return:
        * Arguments:
            - file_path: Location and name of resistive pulse file to be opened.
        """
        self._file_path = file_path
        self._file_name = self._file_path.split('/')[-1]
        self._directory = self._file_path.split('/')[:-1]

        self._file_type = get_file_type(self._file_path)

        self._file_size = get_file_length(self._file_path)

        return




def open_event_file_json(file_path):
    """
    * Description: Loads resistive pulse events saved in a .json format.
    * Return:
        - events: list[] of ResistivePulseEvent
    * Arguments:
        - file_path: The location of the file to load.
    """
    events = []

    with open(file_path, 'r') as fh:
        json_reader = json.load(fh)
        for event in json_reader['events']:
            baseline = np.array(event['baseline'])
            data = np.array(event['data'])
            id = event['id']
            events.append(resistive_pulse.ResistivePulseEvent(data, baseline, id))

    return events

def save_rp_events_json(file_path, events):
    """
    * Description: Saves data in .json format
    * Return: None
    * Arguments:
        - file_path: Path to save file
    """



    with open(file_path, 'w') as fh:

        event_json_list = []

        for i, event in enumerate(events):
            event_json_list.append({'id': str(event._id),
                                    'baseline': event._baseline.tolist(),
                                    'data': event._data.tolist()})

        events = {'events': event_json_list}

        json.dump(events, fh)

    return

def get_file_type(file_path):
    """
    * Description: Get total number of rows in file
    * Return: # rows in file (Int)
    * Arguments:
        - file_path: Name of desired file
    """
    file_type = file_path.split('.')[-1]

    if file_type == 'bts':
        return 'bts'
    elif file_type == 'atf':
        return 'atf'
    elif file_type == 'abf':
        return 'abf'
    elif len(file_path.split('.')) == 1:
        return 'raw'


    else:
        return None


def get_file_length(file_path):
    """
    * Description: Get total number of rows in .bts, .atf, or raw file
    * Return: # rows in file (Int), or 0 if the file type is not understood
    * Arguments:
        - file_path: Name of desired file
    """

    file_type = get_file_type(file_path)

    if file_type == 'bts':
        with open(file_path) as f:
            file_length = int(os.path.getsize(file_path)/16)
            return file_length

    elif file_type == 'abf':
        r = neo.io.AxonIO(filename = file_path)

        block = r.read_block(lazy = False, cascade = True)
        duration = block.segments[0].analogsignals[0].duration.item()
        sampling_period = block.segments[0].analogsignals[0].sampling_period.item()
        data_points = int(duration/sampling_period)
        return data_points


    elif file_type == 'atf':
        acq_mode = get_atf_acquisition_mode(file_path)
        with open(file_path) as f:
            for i, l in enumerate(f):
                pass
            f.close()
            if acq_mode == 'Episodic Stimulation':
                return i+1-ATF_HEADER_LENGTH_EPISODIC_STIMULATION
            elif acq_mode == 'Gap Free':
                return i+1-ATF_HEADER_LENGTH_GAP_FREE

    elif file_type == 'raw':
        with open(file_path) as f:
            for i, l in enumerate(f):
                pass
            f.close()
            return i+1

    else:
        return 0

def get_file_sampling_frequency(file_path):
    """
    * Description:
    * Return:
    * Arguments:
        -
    """


    if get_file_type(file_path) == 'bts':
        with open(file_path, 'rb') as f:
            first_bytes = f.read(24)
            t0=struct.unpack('d', first_bytes[0:8])[0]
            t1=struct.unpack('d', first_bytes[16:24])[0]
            sampling_frequency = int(1./(t1-t0))
            return sampling_frequency
    elif get_file_type(file_path) == 'abf':
        r = neo.io.AxonIO(filename = file_path)

        block = r.read_block(lazy = False, cascade = True)
        sampling_period = block.segments[0].analogsignals[0].sampling_period.item()
        return int(1./sampling_period)




def np_to_bts(output_file_path, np_data, byte_type = 'd'):
    data_array = array(byte_type, np_data.reshape(-1,1))

    # Write to .bts file

    output_file_handle = open(output_file_path, 'wb')
    data_array.tofile(output_file_handle)

    return

def split_file(file_path, split_factor):
    file_type = (file_path)

    data = get_data(file_path)

    file_length = get_file_length(file_path)

    interval = int(1.*file_length/split_factor)


    if file_type == 'bts':
        for i in xrange(split_factor):
            start = i*interval
            stop = start + interval

            split_data = np.copy(data[start:stop, :])
            #split_data[:,0] = split_data[:,0] - split_data[0,0]



            output_file_path = file_path.split('.')[0]+'_split'+str(i)+'.bts'

            np_to_bts(output_file_path, split_data)

    return

def get_atf_acquisition_mode(file_path):
    """
    * Description: Reads the acquisition mode from the .atf file.
    * Return: acquisition mode (string)
    * Arguments:
        - file_path: Name of file
    """

    input_file_handle=open(file_path, 'r')
    input_reader=csv.reader(input_file_handle, delimiter='\t', quotechar='|')

    for i in xrange(3):
        row = input_reader.next()

    # Get acquisition mode
    acquisition_mode = row[0].split('=')[-1].split('\"')[0]

    input_file_handle.close()

    return acquisition_mode

def atf_to_bts(file_path, current_column = 1, byte_type = 'd'):
    """
    * Description: Converts an axon text file to a binary time-series file
    * Return: None
    * Arguments:
        - file_path: Name of file to load
        - current_column: Column where observable of interest (e.g. current) is located
    """

    # Length of file is first line of output
    file_length = get_file_length(file_path)





    acquisition_mode = get_atf_acquisition_mode(file_path)

    input_file_handle=open(file_path, 'r')
    input_reader=csv.reader(input_file_handle, delimiter='\t', quotechar='|')


    # Return to beginning of file
    input_file_handle.seek(0)

    if acquisition_mode == 'Gap Free':

        # Create empty list to hold all
        data=np.empty((file_length, 2))

        # Skip header
        for i in xrange(ATF_HEADER_LENGTH_GAP_FREE):
            row = input_reader.next()

        # Read contents of .atf into data array
        for i in xrange(file_length):
            row = input_reader.next()
            data[i,0] = float(row[0])
            data[i,1] = float(row[current_column])


        input_file_handle.close()
        # Write to .bts file
        output_file_path = file_path.split('.')[0]+'.bts'

        np_to_bts(output_file_path, data)

    elif acquisition_mode == 'Episodic Stimulation':
        # Skip header
        for i in xrange(ATF_HEADER_LENGTH_EPISODIC_STIMULATION):
            row = input_reader.next()

        # Get # of voltages taken
        row = input_reader.next()
        num_voltages = (len(row)-1)/2 # Subtract time, divide by two for current/voltage


        # Create the matrix
        # Create empty list to hold all
        data=np.empty((file_length, num_voltages+1)) # Add time data
        voltages = np.empty((file_length, num_voltages))

        for i in range(file_length - 1):
            voltages_same = False


            row = input_reader.next()
            data[i,0] = float(row[0])
            #voltages = [0 for i in range(num_voltages)]
            for j in range(num_voltages):
                #voltages[j] = round(float[row[2*(j+1)]])
                data[i,j+1] = float(row[2*j+1])
                voltages[i,j] = round(float(row[2*(j+1)]), 1)

        input_file_handle.close()

        voltage_list = [0 for i in range(num_voltages)]
        for i in range(num_voltages):
            voltage_list[i] = voltages[voltages.shape[0]/2,i]

        for i in range(num_voltages):
            output_file_path = file_path.split('.')[0]+'V_'+str(voltage_list[i]).replace('.', 'p')+'.bts'
            np_to_bts(output_file_path, data[:,[0,i+1]])



        # Read contents of .atf into data array
        #for i in xrange(file_length)

        return


    else:
        print 'Could not convert to .bts... Did not recognize file type!'

    return





def atf_to_raw(file_path, current_column):
    """
    * Description: Converts a .atf file to a raw data file. Saves the file with
      the same name, location as the input file but without the suffix.
    * Return: None
    * Arguments:
        - file_path: Name of desired file
        - current_column: Column that contains the desired data to analyze
          (usually the column containing the current data)
    """

    output_file_path = file_path.split('.')[0]

    # Open file
    file_handle=open(file_path, 'r')
    output_file_handle=open(output_file_path, 'w')
    csv_reader=csv.reader(file_handle, delimiter='\t', quotechar='|')

    # Read data
    for i, row in enumerate(csv_reader):
        if i < ATF_HEADER_LENGTH:
            pass
        else:
            t=row[0]
            I=row[current_column]
            output_file_handle.write(str(t)+"\t"+str(I)+"\n")

    # Close file
    file_handle.close()
    output_file_handle.close()

    return


def adjust_start_stop(file_path, start, stop, file_length = None):
    """
    * Description: Adjusts the start and stop arguments of a call to open data if they are
      illogical (e.g., negative, beyond file length)
    * Return: Returns the adjusted start and stop arguments
    * Arguments:
        - file_path: Name of file to be opened
        - start: Start index
        - stop: Stop index
    """

    if file_length == None:
        file_length = get_file_length(file_path)

    if start == -1:
        start = 0

    if stop == -1:
        stop = file_length

    elif stop > file_length:
        stop = file_length

    return start, stop




def get_data(file_path, start = -1, stop = -1, file_length = None):
    """
    * Description: Gets data from .bts, .atf, or raw data type.
    * Return: 2-D numpy array of data
    * Arguments:
        - file_path: Name of file to be opened
        - start: Starting data point
        - stop: Ending data point
        - file_length (optional): Makes read slightly faster if file is .atf
    """

    file_type = get_file_type(file_path)

    #start, stop = adjust_start_stop(file_path, start, stop, file_length)

    if file_type == 'atf':
        return get_data_atf(file_path, start, stop)

    elif file_type == 'abf':
        return get_data_abf(file_path, start, stop)

    elif file_type == 'bts':
        return get_data_bts(file_path, start, stop)

    # Raw
    elif file_type == 'raw':
        return get_data_raw(file_path, start, stop, file_length)

    else:
        print 'Did not recognize file type.'
        return None


def get_data_abf(file_path, start = -1, stop = -1):
    """
    """

    r = neo.io.AxonIO(filename = file_path)

    block = r.read_block(lazy = False, cascade = True)
    duration = block.segments[0].analogsignals[0].duration.item()
    sampling_period = block.segments[0].analogsignals[0].sampling_period.item()



    # +1 added because there was a mismatch between the signal.times object below and the data array size
    data_points = int(1.*duration/sampling_period)




    data = np.empty((data_points,2))

    # Dont' know why I had this before
    #data = block.segments[0].analogsignals[0].duration*block.segments[0].analogsignals[0].sampling_period

    for i, signal in enumerate(block.segments[0].analogsignals):
        if i == 0:
            data = np.empty((len(signal.times), 2))
            data[:,0] = signal.times
            data[:,1] = signal[:,0].flatten()

    return data


def get_data_bts(file_path, start = -1, stop = -1):
    """
    * Description: Loads data from a binary time series file
    * Return: 2-D numpy array of [time, current] values
    * Arguments:
        - file_path: Name of file to be opened
        - start: Start index
        - stop: Stop index
    """
    f = open(file_path, 'rb')
    data = np.fromfile(f, dtype = np.float64).reshape(-1,2)

    if start == -1:
        start = 0
    if stop == -1:
        stop = data.shape[0]-1



    return data[start:stop,:]



def get_data_raw(file_path, start=-1, stop=-1, file_length = None):
    """
    * Description: Opens raw data file, a data file consisting of only two columns: time and current
    * Return: Time, Current data (numpy array: 'data')
    * Arguments:
        - file_path: Name of desired file to open
        - start (optional): Starting row to load
        - stop (optional): Last row to load
    """
    data = np.empty((stop-start,2))

    # Open file
    file_handle=open(file_path, 'r')
    csv_reader=csv.reader(file_handle, delimiter='\t', quotechar='|')

    # Skip to 'start' row
    for i in xrange(0, start):
        row=csv_reader.next()

    # Read data
    for i in xrange(0, stop-start):
        row=csv_reader.next()
        data[i,0]=row[0]
        data[i,1]=row[1]

    # Close file
    file_handle.close()

    # Return
    return data
