import PyQt4.QtCore as QtCore
import sys
import traceback
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/qt_app/')
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/')
import time_series as ts
import resistive_pulse as rp
import rp_file
import math
from scipy.signal import butter, lfilter

import numpy as np
import copy
import time

class EventFinder(QtCore.QThread):

    event_found = QtCore.pyqtSignal('PyQt_PyObject')
    finished = QtCore.pyqtSignal()

    def __init__(self, raw_data,
                ti,\
                tf,\
                baseline_avg_length,\
                trigger_sigma_threshold,\
                max_search_length,\
                filtered_data = None,\
                go_past_length = 0):

        super(EventFinder, self).__init__(parent = None)



        self._sampling_frequency = int(1./(raw_data[1,0] - raw_data[0,0]))

        self._ti = ti
        self._tf = tf

        if self._ti > self._tf:
            temp = self._ti
            self._ti = self._tf
            self._tf = temp

        if self._ti < 0:
            self._ti = 0

        if self._tf > raw_data[-1,0]:
            self._tf = raw_data[-1,0]



        self._ii = int((self._ti-raw_data[0,0])*self._sampling_frequency)
        self._if = int((self._tf-raw_data[0,0])*self._sampling_frequency)
        print 'times', self._ti, self._tf
        print 'indices', self._ii, self._if

        self._raw_data = copy.copy(raw_data[self._ii:self._if,:])
        if filtered_data != None:
            self._search_data = copy.copy(filtered_data[self._ii:self._if,:])
        else:
            self._search_data = copy.copy(raw_data[self._ii:self._if,:])


        self._baseline_avg_length = baseline_avg_length
        self._trigger_sigma_threshold = trigger_sigma_threshold
        self._max_search_length = max_search_length
        self._go_past_length = go_past_length


    @QtCore.pyqtSlot()
    def run(self):

        index = 0
        start_index = 0
        stop_index = 0



        baseline = rp.get_baseline(self._search_data, index, self._baseline_avg_length,
            self._trigger_sigma_threshold)


        raw_offset = int((self._search_data[0,0] - self._raw_data[0,0])*
                          self._sampling_frequency)

        events_found = 0
        keep_going = True

        events = []

        try:
            while keep_going == True:
                # Look for event start
                start_trigger_found = False
                while start_trigger_found == False:

                    # Check if current exceeds threshold
                    if ((self._search_data[index,1] < baseline[2])
                        or (self._search_data[index,1] > baseline[3])):

                        # Update baseline (i.e., in case of drift)

                        baseline=rp.get_baseline(self._search_data, index-2*self._baseline_avg_length,
                                              self._baseline_avg_length,
                                              self._trigger_sigma_threshold)


                        # Check if point still exceeds trigger threshold after
                        # updating baseline
                        if ((self._search_data[index:index+10,1].mean() < baseline[2])
                            or (self._search_data[index:index+10,1].mean() > baseline[3])):

                            print 'b'
                            # Trigger, get first point to exit baseline
                            start_index = index
                            print 'time = ', self._search_data[index,0]

                            # Requirement for reentry into baseline is that
                            # current returns to value half-way between
                            # the trigger value and the baseline average
                            # value
                            reentry_threshold = (baseline[1]+baseline[2])/2. # Was just baseline[1]
                            while start_trigger_found == False:

                                # Check if data point at start_index
                                # passes reentry_threshold
                                if abs(self._search_data[start_index,1]) >= abs(reentry_threshold):
                                    start_trigger_found = True
                                else:
                                    start_index = (start_index - 1)
                                    if start_index < stop_index:
                                        start_index = stop_index
                                        start_trigger_found = True
                                        print 'invalid start index'
                                        #print 'start index < stop index!\n'
                                        #start_index = stop_index
                                        #start_trigger_found = True

                        else:
                            pass

                    index += 1

                # Look for event stop
                stop_trigger_found = False
                while stop_trigger_found == False:
                    in_baseline = False

                    if ((self._search_data[index:index+10,1].mean() > baseline[2])
                        and (self._search_data[index:index+10,1].mean() < baseline[3])):
                            in_baseline = True

                    # Check if return to baseline
                    if in_baseline == True:

                        stop_index = index

                        # Requirement for reentry into baseline is that
                        # current returns to value half-way between
                        # the trigger value and the baseline average
                        # value
                        reentry_threshold=(baseline[1]+baseline[2])/2. # Was just baseline[1]
                        while stop_trigger_found == False:
                            if abs(self._search_data[stop_index,1])>=abs(reentry_threshold):

                                stop_trigger_found = True
                                stop_index = stop_index + self._go_past_length
                            else:
                                stop_index+=1

                            if stop_index-start_index >= self._max_search_length:
                                stop_trigger_found = True
                                index = start_index+self._max_search_length
                                baseline = rp.get_baseline(self._search_data, index, self._baseline_avg_length,
                                                      self._trigger_sigma_threshold)

                        index=stop_index
                        event = rp.ResistivePulseEvent(copy.deepcopy(self._raw_data[start_index+raw_offset:stop_index+1+raw_offset,:]),
                                                          baseline)
                        events.append(event)
                        events_found += 1
                        print 'event #', events_found, 'index = ', start_index, 'time = ', \
                              self._search_data[start_index,0]

                        # Replace event with baseline

                        replace_start_index = start_index - (stop_index - start_index) - self._baseline_avg_length
                        replace_stop_index = start_index - self._baseline_avg_length

                        if replace_start_index >= 0:
                            # Good
                            self._search_data[start_index:stop_index,1] =\
                                self._search_data[replace_start_index:replace_stop_index,1]
                        else:
                            # Replacement interval starts at negative index. Replace with
                            # as much baseline as possible.
                            interval_length = start_index
                            intervals = (stop_index - start_index)/(start_index+1)+1
                            for i in xrange(intervals):
                                self._search_data[i*start_index:(i+1)*start_index,1] = \
                                    self._search_data[:start_index,1]



                    # Check if exceeded max search length
                    elif index-start_index >= self._max_search_length:
                        stop_trigger_found = True
                        index = start_index+self._max_search_length
                        baseline = rp.get_baseline(self._search_data, index, self._baseline_avg_length,
                                              self._trigger_sigma_threshold)


                    index += 1

        except Exception as inst:
            print 'error! line num = ', sys.exc_info()[2].tb_lineno
            print(type(inst))
            print(inst.args)
            print(inst)
            print 'error! index = ', index, 'start_index = ', start_index, 'stop_index = ',\
                stop_index
            self.emit(QtCore.SIGNAL('events_found(PyQt_PyObject)'), events)
            self.emit(QtCore.SIGNAL('finished()'))

            return
