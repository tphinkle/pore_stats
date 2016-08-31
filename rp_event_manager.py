import csv
import copy
import numpy as np
import json

class RPEventManager(object):
    def __init__(self):
        self._parameters = []

        self._events = []
        self._selected_events = []
        self._targeted_event = None


    def add_events(self, events):
        for event in events:
            event._id = len(self._events)
            self._events.append(event)

        self.select_event(events[0])

        if self._targeted_event == None:
            self._targeted_event = events[0]

        self.sort_events(self._events)
        self.sort_events(self._selected_events)

        return

    def sort_events(self, events):
        moves = -1
        while moves != 0:
            moves = 0
            for i, event in enumerate(events[:-2]):
                if events[i]._data[0,0] > events[i+1]._data[0,0]:
                    moves+=1
                    temp_event = copy.copy(events[i])
                    events[i] = events[i+1]
                    events[i+1] = temp_event

        return



    def clear_events(self):
        self._events = []
        self._selected_events = []
        self._targeted_event = None

        self._parameters = []

        return

    def set_targeted_event(self, event):
        self._targeted_event = event
        return

    def increment_targeted_event(self):
        try:
            targeted_id = self._targeted_event._id

            for i, event in enumerate(self._events):
                if event._id == targeted_id:
                    targeted_index = (i + 1)%len(self._events)
                    break

            self._targeted_event = self._events[targeted_index]

        except:
            pass

        return

    def decrement_targeted_event(self):
        try:
            targeted_id = self._targeted_event._id
            targeted_index = 0
            for i, event in enumerate(self._events):
                if event._id == targeted_id:
                    targeted_index = (i - 1)%len(self._events)
                    break

            self._targeted_event = self._events[targeted_index]

        except:
            pass

        return

    def get_next_targeted_event(self):
        try:
            targeted_id = self._targeted_event._id
            targeted_index = 0
            for i, event in enumerate(self._events):
                if event._id == targeted_id:
                    targeted_index = (i + 1)%len(self._events)
                    break



        except:
            pass

        return self._events[targeted_index]

    def get_previous_targeted_event(self):
        try:
            targeted_id = self._targeted_event._id
            targeted_index = 0
            for i, event in enumerate(self._events):
                if event._id == targeted_id:
                    targeted_index = (i - 1)%len(self._events)
                    break



        except:
            pass

        return self._events[targeted_index]

    def get_previous_targeted_selected_event(self):
        targeted_event_id = self._targeted_event._id
        new_targeted_event_id = None
        for i, event in enumerate(self._selected_events):
            if event._id >= targeted_event_id:
                new_targeted_event_id = self._selected_events[i-1]._id
                print new_targeted_event_id
                break

        if new_targeted_event_id == None:
            if len(self._selected_events) != 0:
                new_targeted_event_id = self._selected_events[-1]._id
            else:
                new_targeted_event_id = self._targeted_event._id

        return self._events[new_targeted_event_id]

    def get_next_targeted_selected_event(self):
        targeted_event_id = self._targeted_event._id
        new_targeted_event_id = None
        for i, event in enumerate(reversed(self._selected_events)):

            if event._id <= targeted_event_id:
                new_targeted_event_id = \
                    self._selected_events[(len(self._selected_events)-i-1+1)%(len(self._selected_events))]._id
                break

        if new_targeted_event_id == None:
            if len(self._selected_events) != 0:
                new_targeted_event_id = self._selected_events[0]._id
            else:
                new_targeted_event_id = self._targeted_event._id

        return self._events[new_targeted_event_id]

    def toggle_selected(self, event):
        event_id = event._id
        if self.is_selected(event) == True:
            self._selected_events = [event for event in self._selected_events if event._id != event_id]
        else:
            self._selected_events.append(event)

        self.sort_events(self._selected_events)

        return

    def select_event(self, event):
        event_id = event._id
        if self.is_selected(event) == False:
            self._selected_events.append(event)
            self.sort_events(self._selected_events)
        return

    def unselect_event(self, event):
        event_id = event._id
        if self.is_selected(event) == True:
            self._selected_events = [event for event in self._selected_events if event._id != event_id]
            self.sort_events(self._selected_events)
        return

    def is_selected(self, event):
        event_id = event._id
        is_selected = False
        if event._id in [event._id for event in self._selected_events]:
            is_selected = True
        else:
            is_selected = False

        return is_selected

    def get_targeted_index(self):
        for i, event in enumerate(self._events):
            if self.is_targeted(event):
                return i

    def is_targeted(self, event):
        is_targeted = False
        if self._targeted_event._id == event._id:
            is_targeted = True

        return is_targeted

    def save_events(self, file_path):

        print 'saving events!'

        f = open(file_path, 'w')
        writer = csv.writer(f, delimiter = '\t')

        for i, event in enumerate(self._selected_events):
            header_row = ['event#', i, 'length', event._data.shape[0],\
                'baseline', event._baseline[0], event._baseline[1], event._baseline[2], event._baseline[3]]

            writer.writerow(header_row)

            for row in event._data:
                writer.writerow([row[0], row[1]])

        f.close()

        return

    def save_events_json(self, file_path):
        """
        * Description: Saves data in .json format
        * Return: None
        * Arguments:
            - file_path: Path to save file
        """

        print 'saving events!'

        with open(file_path, 'w') as fh:

            event_json_list = []

            for i, event in enumerate(self._selected_events):
                event_json_list.append({'id': str(event._id),
                                        'baseline': event._baseline.tolist(),
                                        'data': event._data.tolist()})

            events = {'events': event_json_list}

            json.dump(events, fh)

        return

    def get_selected_dur_amp(self):
        """
        * Description: Returns a 2-D numpy array of duration and amplitude for all
          selected events (e.g. to make a scatter plot).
        * Return: 2-D numpy array of duration and amplitude (np.array)
        * Arguments:
        """

        dur_amp = np.empty((len(self._selected_events), 2), dtype = np.float32)

        for i, event in enumerate(self._selected_events):
            dur_amp[i,0] = event._duration
            dur_amp[i,1] = event._amplitude

        return dur_amp

    def get_unselected_dur_amp(self):
        """
        * Description: Returns a 2-D numpy array of duration and amplitude for all
          unselected events (e.g. to make a scatter plot).
        * Return: 2-D numpy array of duration and amplitude (np.array)
        * Arguments:
        """

        dur_amp = np.empty((len(self._events) - len(self._selected_events), 2), dtype = np.float32)

        for i, event in enumerate([e for e in self._events if (self.is_selected(e) == False)]):
            dur_amp[i,0] = event._duration
            dur_amp[i,1] = event._amplitude

        return dur_amp
