import sys

PORE_STATS_DIR = '/home/preston/Desktop/Science/Research/pore_stats/'
sys.path.append(PORE_STATS_DIR)
import resistive_pulse as rp
import rp_file
import time_series as ts
import time

class RPModel(object):
    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        self._path_name = file_name
        self.announce_update_file_path()

    @property
    def display_data(self):
        return self._display_data

    @display_data.setter
    def display_data(self, display_data):
        self._display_data = display_data
        self.announce_update_display_data()

    @property
    def display_baseline_data(self):
        return self._display_baseline_data

    @display_baseline_data.setter
    def display_baseline_data(self, display_baseline_data):
        self._display_baseline_data = display_baseline_data
        self.announce_update_display_baseline_data()

    @property
    def t_range(self):
        return self._t_range

    @t_range.setter
    def t_range(self, t_range):
        self._t_range = t_range

    @property
    def raw_events(self):
        return self._raw_events

    @raw_events.setter
    def raw_events(self, raw_events):
        self._raw_events = raw_events
        self.announce_update_raw_events()

    @property
    def targeted_event(self):
        return self._targeted_event

    @targeted_event.setter
    def targeted_event(self, targeted_event):
        self._targeted_event = targeted_event
        self.announce_update_targeted_event()

    display_decimation_threshold = 100000
    decimation_factor = 2




    def __init__(self):
        self._active_file = None

        self._data = None
        self._baseline_data = None

        self._t_range=(0,0)

        self._raw_events = []
        self._targeted_event = None

        self._subscribers = []


        return


    def set_baseline_avg_length(self, baseline_avg_length):
        self._baseline_avg_length = baseline_avg_length

        return

    def set_trigger_sigma_threshold(self, trigger_sigma_threshold):
        self._trigger_sigma_threshold = trigger_sigma_threshold

        return

    def set_max_search_length(self, max_search_length):
        self._max_search_length = max_search_length

        return

    def set_active_file(self, file_path):
        self._active_file = rp_file.RPFile(file_path)

        return

    def get_data_from_file(self):
        print 't1!:', time.time()
        data = rp_file.get_data(self._active_file._file_path)
        print 't2!:', time.time()
        self._data = ts.TimeSeries(data, self.display_decimation_threshold, self.decimation_factor)
        print 't3!:', time.time()

        return

    def set_t_range(self, t_range):

        # Change t_range
        self._t_range = t_range

        if self._data != None:
            self.display_data = self._data.return_data(self._t_range[0], self._t_range[1])





        #if self._baseline_data != None:
            #self.display_baseline_data = self._baseline_data.get_decimated_data(self._t_range[0], self._t_range[1])

        return

    def calculate_baseline(self):
        self._baseline_data = ts.TimeSeries(rp.get_full_baseline(\
                                                                 self._data._data, \
                                                                 self._baseline_avg_length,\
                                                                 self._trigger_sigma_threshold),\
                                            self.display_decimation_threshold,
                                            self.decimation_factor)

        return




    def find_events(self):
        self._events =\
          rp.find_events_data(self._data._data, raw_data = None,
                                baseline_avg_length = self._baseline_avg_length,
                                trigger_sigma_threshold = self._trigger_sigma_threshold,
                                max_search_length = self._max_search_length)


        if self._targeted_event == None and len(self._events) > 0:
            self.targeted_event = self._events[0]

    def increment_targeted_event(self):
        new_index = (self._targeted_event._index + 1)%len(self._events)




    def add_subscriber(self, subscriber):
        self._subscribers.append(subscriber)
        return


    def announce_updates(self):
        self.announce_update_file_path()
        self.announce_update_display_data()
        self.announce_update_baseline()
        return

    def announce_update_file_path(self):
        for subscriber in self._subscribers:
            subscriber.receive_update_file_path(self._active_file._file_path)
        return

    def announce_update_display_data(self):
        for subscriber in self._subscribers:
            subscriber.receive_update_data(self._display_data)
        return

    def announce_update_display_baseline_data(self):
        for subscriber in self._subscribers:
            subscriber.receive_update_baseline_data(self._display_baseline_data)
        return

    def announce_update_raw_events(self):
        for subscriber in self._subscribers:
            for event in self._raw_events:
                pass

    def announce_update_targeted_event(self):
        for subscriber in self._subscribers:
            subscriber.receive_update_targeted_event(self._targeted_event._data[:,:])
