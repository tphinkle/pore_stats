import PyQt4.QtCore as QtCore

import numpy as np

import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/')
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/qt_app/threads/')


import math
import time

import rp_file

class TimeSeries(QtCore.QObject):
    """
    * Description: Creates a time series class that is capable of presenting a full time
      series or a decimated time series within a given time range
    * Class variables:
    * Member variables:
    * Class methods:
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

        self._file_path = file_path
        self._full_data = full_data

        self._max_pts_returned = max_pts_returned
        self._decimation_factor = decimation_factor

        if file_path != None:
            self._data_points = rp_file.get_file_length(file_path)
            self._sampling_frequency = rp_file.get_file_sampling_frequency(file_path)

        if full_data != None:
            self._t0 = full_data[0,0]
            self._full_data = full_data
            self._data_points = self._full_data.shape[0]
            self._sampling_frequency = int(1./(full_data[1,0] - full_data[0,0]))


        self._decimation_tiers = int(math.ceil(math.log(1.*self._data_points/\
                                     self._max_pts_returned,\
                                     self._decimation_factor)))+1

        if self._decimation_tiers <= 1:
            self._decimation_tiers = 1
            self._display_ready = True

        self._decimated_data_list = [None for i in xrange(self._decimation_tiers)]

        self._decimated_data_list[0] = self._full_data

        self._key_parameters = key_parameters

        return



    def add_decimated_data_tier(self, data, tier):
        self._decimated_data_list[tier]=data

        if tier == 0:
            self._t0 = data[0,0]
            self._analyze_ready = True

        if tier == self._decimation_tiers - 1:
            self._display_ready = True

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
        * Return:
        * Arguments:
            -
        """




        if t_i < self._decimated_data_list[0][0,0]:
            t_i = self._t0
            #t_i = 0
        if t_f > self._decimated_data_list[0][-1,0]:
            t_f = self._decimated_data_list[0][-1,0]
        num_points = int((t_f-t_i)*self._sampling_frequency)


        decimation_tier = 0
        while num_points > self._max_pts_returned:
            num_points=num_points/self._decimation_factor
            decimation_tier += 1

        tier_found = False
        while tier_found == False and decimation_tier < self._decimation_tiers:
            if self._decimated_data_list[decimation_tier] == None:
                decimation_tier += 1
            else:
                tier_found = True

        if tier_found == False:
            return None

        i_i = self.get_index_from_time_decimated(self._decimated_data_list[decimation_tier],\
         t_i, self._sampling_frequency, self._decimation_factor**decimation_tier)

        i_f = self.get_index_from_time_decimated(self._decimated_data_list[decimation_tier],\
         t_f, self._sampling_frequency, self._decimation_factor**decimation_tier)+1


        data = self._decimated_data_list[decimation_tier][i_i:i_f,:]

        self._current_decimation_factor = self._decimation_factor**decimation_tier

        return data




    def get_index_from_time_decimated(self, decimated_data, time, sampling_frequency, decimation):
        """
        * Description:
        * Return:
        * Arguments:
            -
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

def get_sampling_frequency(data):
    """
    * Description:
    * Return:
    * Arguments:
        -
    """
    sampling_frequency = int(1./(data[1,0] - data[0,0]))

    return sampling_frequency


def decimate_data(data, decimation_factor):
    """
    * Description:
    * Return:
    * Arguments:
        -
    """
    if decimation_factor > 1:
        decimated_data = np.empty((int(data.shape[0]/decimation_factor), data.shape[1]))
        for i in xrange(decimated_data.shape[0]):
            decimated_data[i,:] = data[decimation_factor*i,:]

        return decimated_data

    else:
        return data
