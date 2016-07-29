import PyQt4.QtCore as QtCore
import sys
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


    def __init__(self, raw_data,\
                baseline_avg_length,\
                trigger_sigma_threshold,\
                max_search_length,\
                f_cutoff = 0,\
                go_past_length = 0):
        super(EventFinder, self).__init__()

        self._raw_data = raw_data
        self._baseline_avg_length = baseline_avg_length
        self._trigger_sigma_threshold = trigger_sigma_threshold
        self._max_search_length = max_search_length
        self._f_cutoff = 0
        self._go_past_length = go_past_length

    def __del__(self):
        pass

    def run(self):

        # numpy array to hold baseline info:
        # [time, average current, positive threshold, negative threshold]
        baseline_history=np.empty((0,4), dtype=float)

        # numpy array to hold event start, stop indices
        event_indices=np.empty((0,2), dtype=int)

        # List that will store events and be returned
        events=[]

        # Get file length; define start, stop points; check points
        if self._f_cutoff > 0:
            nyquist = .5/(self._raw_data[1,0] - self._raw_data[0,0])
            b, a = butter(N=5, Wn=self._f_cutoff/nyquist, btype = 'low', analog = False)
            search_data = np.hstack((self._raw_data[:,0], lfilter(b, a, self._raw_data[:,1])))
        else:
            search_data = self._raw_data[:,:]

        original_segments = []




        index = 0
        baseline=rp.get_baseline(search_data, index, self._baseline_avg_length,
        self._trigger_sigma_threshold)
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
                    baseline=rp.get_baseline(search_data, index-2*self._baseline_avg_length,
                                          self._baseline_avg_length,
                                          self._trigger_sigma_threshold)
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
                            stop_index = stop_index + self._go_past_length
                        else:
                            stop_index+=1

                    index=stop_index
                    event_indices=np.vstack((event_indices,
                                  np.array([start_index, stop_index])))


                    events.append(rp.ResistivePulseEvent(copy.deepcopy(search_data[start_index:stop_index+1,:]),
                                                      baseline))

                    self.emit(QtCore.SIGNAL('event_found(PyQt_PyObject)'), events[-1])
                    print 'event # ', event_indices.shape[0], 't_i = ', search_data[start_index,0]
                    #print start_index, ',', stop_index, events[-1]._data[0,0], ',', events[-1]._data[-1,0]

                    # Replace event with baseline
                    search_data[start_index:stop_index,1]=search_data[start_index-\
                        (stop_index-start_index)-self._baseline_avg_length:start_index-self._baseline_avg_length,1]



                # Check if exceeded max search length
                elif index-start_index >= self._max_search_length:
                    stop_trigger_found = True
                    index = start_index+self._max_search_length
                    baseline = get_baseline(search_data, index, self._baseline_avg_length,
                                          self._trigger_sigma_threshold)


                index += 1




        # Replace altered segments with original segments



        return
