import numpy as np

class Event:

    def __init__(self):
        self._start_index=None
        self._stop_index=None
        self._length=None
        self._data=None
        self._predictions=None
        self._levels=None
        self._split_list=[]
        self._peaks=None
        self._antipeaks=None
        return

    def set_start_stop_index(self, start_index, stop_index):
        self._start_index=start_index
        self._stop_index=stop_index
        self._length=stop_index-start_index
        return

    def set_data(self, data):
        self._data=data
        return

    def set_predictions(self, predictions):
        self._predictions=predictions
        return

    def set_levels(self, levels):
        self._levels=levels
        return

    def set_splits(self, splits):
        self._splits=splits
        return

    def set_peaks(self, peaks):
        self._peaks=peaks
        return

    def set_antipeaks(self, antipeaks):
        self._antipeaks=antipeaks
        return
