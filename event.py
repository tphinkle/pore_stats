import numpy as np
from copy import copy

class Event:

    def __init__(self):
        self._start_index=None
        self._stop_index=None
        self._length=None
        self._data=None

        self._smoothed_data=None
        self._maxima=None
        self._minima=None
        self._extrema=None

        self._amplitude=None
        self._duration=None
        return

    def set_start_stop_index(self, start_index, stop_index):
        self._start_index=copy(int(1.*start_index))
        self._stop_index=copy(int(1.*stop_index))
        self._length=copy(int(stop_index-start_index))
        return

    def set_data(self, data):
        self._data=copy(data)
        return

    def set_smoothed_data(self, smoothed_data):
        self._smoothed_data=copy(smoothed_data)
        return

    def set_extrema(self, minima, maxima):
        self._extrema=minima+maxima
        return

    def set_maxima(self, maxima):
        self._maxima=copy(maxima)
        return

    def set_minima(self, minima):
        self._minima=copy(minima)
        return

    def calculate_amplitude(self):
        self._amplitude=self._data[:,1].max()-self._data[:,1].min()
        return

    def calculate_duration(self):
        self._duration=self._data[-1,0]-self._data[0,0]
        return
