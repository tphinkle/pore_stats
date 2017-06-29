import PyQt4.QtCore as QtCore

import numpy as np

import sys
import os
PORE_STATS_BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__)).replace('/lib', '')
sys.path.append(PORE_STATS_BASE_DIRECTORY)
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/threads')


import math
import time

import rp_file

class TimeSeries(QtCore.QObject):
    """
    * Description:
        - Creates a time series class that is capable of presenting a full time
        series or a decimated time series within a given time range
        - Data is contained in numpy structures
        - Data can be decimated so that when data is requested from timeseries, the
        appropriate decimation is chosen; for instance, the timeseries will never return
        data that has more than max_pts_returned points; it will choose the data set
        with the least decimation with less than max_pts_returned data points.
        - This is done for visualization purposes; since a time series might consist of
        1E9 data points, displaying all of that data is unneccessary and very expensive.
        Thus the time-series will instead supply a decimated data array instead, with the
        degree of decimation calculated automatically.
        - The decimated data lists are static, i.e. not calculated for every request. This
        incurs an overhead memory cost but makes the program run much smoother.
        - The degree of decimation of each decimation level is exponential, for instance
        degree 0 decimation has a factor of 1X, degree 1 decimation has a factor of
        (1/decimation_factor)X, degree 2 decimation has a factor of
        (1/decimation_factor)(1/decimation_factor)X, etc. This means the overhead memory
        cost is a geometric series which is guaranteed to always be less than or equal
        to the size of the memory itself. The case with greatest overhead corresponds to
        2X decimation at each tier.
    """

    analyze_ready = QtCore.pyqtSignal(bool)
    display_ready = QtCore.pyqtSignal(bool)
    all_ready = QtCore.pyqtSignal(bool)


    def __init__(self, key_parameters):
        super(TimeSeries,self).__init__(None)


        self._analyze_ready = False

        self._display_ready = False

        self._all_ready = False

        self._visible = False

        self._key_parameters = {key_parameter: None for key_parameter in key_parameters}

    def initialize(self, file_path, full_data, \
                   max_pts_returned, decimation_factor, key_parameters = {}):
        """
        * Description:
            - Initializes the time_series with its parameter information, such as file
            location, the complete data, etc.
            - Initializes either from a full, undecimated data array -- or -- a file_path,
            in which case the file is opened.
        * Return:
        * Arguments:
            - file_path: File location and name
            - full_data: Numpy array containing the full (non-decimated) data set
            - max_pts_returned: Maximum points that will be returned when a request for
            data is made. This along with teh decimation factor determines the decimation
            tier that will be returned.
            - decimation_factor: The factor by which data is decimated at every tier.
            E.g., this could be 10x, which means the decimated data arrays stored would be
            decimated by 1/10X, 1/100X, 1/1000X...
            - key_parameters: dict {} of the parameters that were used in the search to
            generate the time_series; this is important for creating filtered time-series.
        """

        self._file_path = file_path
        self._full_data = full_data


        self._max_pts_returned = max_pts_returned
        self._decimation_factor = decimation_factor

        if file_path != None:
            self._data_points = rp_file.get_file_length(str(file_path))
            self._sampling_frequency = rp_file.get_file_sampling_frequency(str(file_path))



        if full_data != None:
            self._t0 = full_data[0,0]
            self._full_data = full_data
            self._data_points = self._full_data.shape[0]
            self._sampling_frequency = int(1./(full_data[1,0] - full_data[0,0]))


        self._decimation_tiers = int(math.ceil(math.log(1.*self._data_points/\
                                     self._max_pts_returned,\
                                     self._decimation_factor)))+1

        # Data does not need to be decimated.
        if self._decimation_tiers <= 1:
            self._decimation_tiers = 1
            self._display_ready = True

        # Initialize the list of decimated data
        self._decimated_data_list = [None for i in xrange(self._decimation_tiers)]
        self._decimated_data_list[0] = self._full_data

        self._key_parameters = key_parameters

        return



    def add_decimated_data_tier(self, data, tier):
        """
        * Description: Adds the data array to the correct position in
        decimated_data_list; also updates the readiness of the time_series, e.g., if it is
        ready to display or to be calculated on.
        * Return:
        * Arguments:
            - data: The decimated data array
            - tier: The decimated data tier, e.g., 0, 1, ...
        """
        self._decimated_data_list[tier]=data

        # Data is ready to analyze if the lowest decimation tier has been loaded.
        if tier == 0:
            self._t0 = data[0,0]
            self._analyze_ready = True

        # Data is ready to display if the highest decimation tier has been loaded.
        if tier == self._decimation_tiers - 1:
            self._display_ready = True


        # Check if all the tiers have been loaded
        all_tiers_loaded = True
        for tier in xrange(self._decimation_tiers):
            if self._decimated_data_list[tier] == None:
                all_tiers_loaded = False

        if all_tiers_loaded == True:
            self._all_ready = True

        return



    def return_data(self, t_i, t_f):
        """
        * Description:
            - Returns the data from t_i to t_f from the correct decimation tier.
            - First calculates the correct decimation tier, then determines the indices
            to return from within that tier.
        * Return:
            - data: The decimated data array.
        * Arguments:
            - t_i: Starting index.
            - t_f: Ending index.
        """




        # Get the number of data points in the undecimated data that is requested.
        if t_i < self._decimated_data_list[0][0,0]:
            t_i = self._t0
        if t_f > self._decimated_data_list[0][-1,0]:
            t_f = self._decimated_data_list[0][-1,0]
        num_points = int((t_f-t_i)*self._sampling_frequency)





        # Calculate the correct decimation tier based off of the range requested
        decimation_tier = 0
        while num_points > self._max_pts_returned:
            num_points=num_points/self._decimation_factor
            decimation_tier += 1



        # Check to make sure the requested decimation tier actually exists; if it does
        # not, move up to the next decimation tier.
        tier_found = False
        while tier_found == False and decimation_tier < self._decimation_tiers:
            if self._decimated_data_list[decimation_tier] == None:
                decimation_tier += 1
            else:
                tier_found = True

        if tier_found == False:
            return None

        # Get the indices for the correct decimated data array
        i_i = self.get_index_from_time_decimated(self._decimated_data_list[decimation_tier],\
         t_i, self._sampling_frequency, self._decimation_factor**decimation_tier)

        i_f = self.get_index_from_time_decimated(self._decimated_data_list[decimation_tier],\
         t_f, self._sampling_frequency, self._decimation_factor**decimation_tier)+1


        data = self._decimated_data_list[decimation_tier][i_i:i_f,:]

        self._current_decimation_factor = self._decimation_factor**decimation_tier

        return data




    def get_index_from_time_decimated(self, decimated_data, time, sampling_frequency, decimation):
        """
        * Description: Given a time-value, decimation level, and sampling_frequency of the
        signal, returns the correct index for that time in the decimated array.
        * Return:
            - index: The closest index corresponding to time in the decimated array.
        * Arguments:
            - decimated_data: The data array
            - time: Time we are looking for index for
            - sampling_frequency: Frequency of the __non-decimated__ data
            - decimation: The degree of decimation of the data in which we are looking
            for the index.
        """
        if decimation == 0:
            index = int((time-self._t0)*sampling_frequency)
            #index = int(time*sampling_frequency)
        else:
            index = int((time-self._t0)*sampling_frequency/decimation)
            #index = int(time*sampling_frequency/decimation)

        if index < 0:
            index = 0

        if index > decimated_data.shape[0]:
            index = decimated_data.shape[0]-1

        return index


def decimate_data(data, decimation_factor):
    """
    * Description: Function that actually performs the decimation on a data set.
    * Return:
        - Decimated data: The data to be decimated
    * Arguments:
        - decimation_factor: Factor that data is reduced by, e.g. 10X gives data with
        1/10th the number of data points.
    """
    if decimation_factor > 1:
        decimated_data = np.empty((int(data.shape[0]/decimation_factor), data.shape[1]))
        for i in xrange(decimated_data.shape[0]):
            decimated_data[i,:] = data[decimation_factor*i,:]

        return decimated_data

    else:
        return data
