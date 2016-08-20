"""

IMAGING

* Contains tools for opening, analyzing imaging data from experiments

* Sections:
    1. Imports
    2. Constants
    3. Classes
        - OpticalEvent: All data associated with passage of particle through
          the stage
        - Stage: A struct containing data about the 'stage', including pixel
          data of empty stage to be used as a template, channel coordinates,
          etc.
    4. Functions
"""

"""
Imports
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt
from array import array
import oi_file
import os
import copy
import time

"""
Constants
"""
FFMPEG_BIN_FILENAME = '/home/preston/ffmpeg-3.0.2-64bit-static/ffmpeg'


"""
Classes
"""
class OpticalDetection:

    def __init__(self, tf = None, pixels = None):
        self._tf = None
        self._pixels = None
        self._px = None
        self._py = None
        self._pvx = None
        self._pvy = None
        self._parea = None
        self._pwidth = None
        self._pheight = None

        if pixels != None:

            self._tf = int(tf)
            self._pixels = pixels
            self._px = int((pixels[:,1].max()+pixels[:,1].min())/2)
            self._py = int((pixels[:,0].max()+pixels[:,0].min())/2)
            self._pvx = 0
            self._pvy = 0

            self._parea = len(self._pixels)
            self._pwidth = pixels[:,1].max() - pixels[:,1].min()
            self._pheight = pixels[:,0].max() - pixels[:,0].min()

        return

class OpticalEvent:

    def __init__(self, detections = []):
        self._detections = detections
        self._channel_enter_tf = None
        self._channel_exit_tf = None

        self._channel_enter_index = None
        self._channel_exit_index = None



        return

    def sort_detections(self):

        return

    def add_detection(self, detection):
        self._detections.append(detection)
        self.sort_detections()

        return



    def connect_event(self, optical_event):
        if self._detections[-1]._tf < optical_event._detections[0]._tf:
            self._detections = self._detections + optical_event._detections[:]

        else:
            self._detections =  optical_event._detections[:] + self._detections


        return

    def get_px(self):
        px = []
        for detection in self._detections:
            px.append(detection._px)
        return px

    def get_py(self):
        py = []
        for detection in self._detections:
            py.append(detection._py)
        return py

    def get_tf(self):
        tf = []
        for detection in self._detections:
            tf.append(detection._tf)
        return tf

    def get_channel_enter_exit_tf(self, stage):
        pxs = self.get_px()
        pys = self.get_py()
        tfs = self.get_tf()

        channel_enter_tf = None

        if stage.get_channel_coordinates(pxs[0], pys[0])[0] < 0:
            for i in xrange(len(pxs)):
                if stage.get_channel_coordinates(pxs[i], pys[i])[0] > 0:
                    self._channel_enter_tf = tfs[i]
                    self._channel_enter_index = i
                    break

        if self._channel_enter_tf != None:
            for i in xrange(len(pxs)):
                if stage.get_channel_coordinates(pxs[i], pys[i])[0] > stage._channel_plength:
                    self._channel_exit_tf = tfs[i]
                    self._channel_exit_index = i
                    break

        return










class Stage:
    """
    * Description: A struct containing data about the 'stage', the areas
      including and surrounding the microfluidic channel, including pixel
      and real-space coordinates of channel entrances and exits, etc.
    * Class variables:
        - channel_length: Length of channel in microns (Int)
        - channel_width_large: Largest width of channel in microns (Int)
        - channel_width_small: Smallest width of channel in microns (Int)
        - channel_depth: Depth of channel in microns (Int)
    * Member variables:
        - _channel_plength: Length of channel in pixels (Int)
        - _channel_pwidth_large: Largest width of channel in pixels (Int)
        - _channel_pwidth_small: Smallest width of channel in pixels (Int)
        - _channel_pdepth: Depth of channel in pixels (Int)
        - _channel_pcorner_0: Pixel coordinates of top left corner of
          channel_pdepth (2 element numpy array of Int)
        - _channel_pcorner_1: Pixel coordinates of bottom left corner of
          channel_pdepth (2 element numpy array of Int)
        - _channel_pcorner_2: Pixel coordinates of bottom right corner of
          channel_pdepth (2 element numpy array of Int)
        - _channel_pcorner_3: Pixel coordinates of top right corner of
          channel_pdepth (2 element numpy array of Int)
        - _template_image: Pixel data of the stage without any OpticalEvents
          occurring (2-D numpy array of RGB pixel data (3 element numpy array))
    * Class functions:
        - get_channel_coordinates()
    """

    channel_length = 125
    channel_width_large = 35
    channel_width_small = 25
    channel_depth = 45

    def __init__(self, template_frame, channel_pcorner_0, channel_pcorner_1,
                 channel_pcorner_2, channel_pcorner_3):

        # 0       3
        # 1       2
        self._template_frame = None
        self._channel_porigin = None
        self._channel_pcorner_0 = None
        self._channel_pcorner_1 = None
        self._channel_pcorner_2 = None
        self._channel_pcorner_3 = None

        self._channel_plength = None
        self._channel_pwidth = None

        self._template_frame = template_frame
        self._channel_porigin = (channel_pcorner_0 + channel_pcorner_1)/2.
        self._channel_pcorner_0 = channel_pcorner_0
        self._channel_pcorner_1 = channel_pcorner_1
        self._channel_pcorner_2 = channel_pcorner_2
        self._channel_pcorner_3 = channel_pcorner_3


        axis_0 = self._channel_pcorner_1 - self._channel_pcorner_0
        axis_1 = self._channel_pcorner_2 - self._channel_pcorner_1
        axis_2 = self._channel_pcorner_3 - self._channel_pcorner_2
        axis_3 = self._channel_pcorner_0 - self._channel_pcorner_3

        self._channel_plength = ((axis_1[0]**2.+axis_1[1]**2.)**.5+
                                 (axis_3[0]**2.+axis_3[1]**2.)**.5)/2.

        self._channel_pwidth = ((axis_0[0]**2.+axis_0[1]**2.)**.5+
                                 (axis_2[0]**2.+axis_2[1]**2.)**.5)/2.

        self._channel_xaxis = ((self._channel_pcorner_3 +
                             self._channel_pcorner_2)/2. -
                            (self._channel_pcorner_0 +
                             self._channel_pcorner_1)/2.)

        self._channel_xaxis = (self._channel_xaxis/(self._channel_xaxis[0]**2. +
                                              self._channel_xaxis[1]**2.)**.5)

        self._channel_yaxis = np.array((-self._channel_xaxis[1],
                                        self._channel_xaxis[0]))

    def get_channel_coordinates(self, x, y):
        px = ((x - self._channel_porigin[0])*self._channel_xaxis[0] +
              (y - self._channel_porigin[1])*self._channel_xaxis[1])
        py = ((x - self._channel_porigin[0])*self._channel_yaxis[0] +
              (y - self._channel_porigin[1])*self._channel_yaxis[1])

        return (px, py)

    def plot_stage(self):
        fig = plt.figure(figsize = (20, 12))

        plt.scatter(self._channel_porigin[0], self._channel_porigin[1], marker = 'x')

        plt.plot([self._channel_porigin[0], self._channel_porigin[0]+1000*self._channel_xaxis[0]],
         [self._channel_porigin[1], self._channel_porigin[1]+1000*self._channel_xaxis[1]], lw = 2, ls = '--', c = (1.,0,0))

        plt.plot([self._channel_porigin[0], self._channel_porigin[0]+1000*self._channel_yaxis[0]],
         [self._channel_porigin[1], self._channel_porigin[1]+1000*self._channel_yaxis[1]], lw = 2, ls = '--', c = (1.,0,0))

        plt.plot([self._channel_porigin[0], self._channel_porigin[0]-1000*self._channel_xaxis[0]],
         [self._channel_porigin[1], self._channel_porigin[1]-1000*self._channel_xaxis[1]], lw = 2, ls = '--', c = (1.,0,0))

        plt.plot([self._channel_porigin[0], self._channel_porigin[0]-1000*self._channel_yaxis[0]],
         [self._channel_porigin[1], self._channel_porigin[1]-1000*self._channel_yaxis[1]], lw = 2, ls = '--', c = (1.,0,0))

        plt.imshow(self._template_frame, cmap = 'gray', vmin = 0, vmax = 1, origin = 'lower')
        for c in [self._channel_pcorner_0, self._channel_pcorner_1,
                   self._channel_pcorner_2, self._channel_pcorner_3]:
             plt.scatter(c[0], c[1], marker = 'x')

        plt.text(self._channel_pcorner_0[0], self._channel_pcorner_0[1], str(self._channel_pcorner_0))

        plt.xlim(0, self._template_frame.shape[1])
        plt.ylim(0, self._template_frame.shape[0])
        plt.show()

        return



"""
Functions
"""


def get_frame_vid(vid, tf):
    """
    * Description: Gets the nth frame of data from video stream. Converts pixel
      RGB values to type Int.
    * Return: Frame data (2-D numpy array of RGB pixel data scaled to range [0,1)
      (3 element numpy array))
    * Arguments
        - vid: Video connection established by imageio class (use
          'open_video_connection') to get
        - tf: Number of desired frame to retrieved
    """
    frame=np.array(vid.get_data(tf)[:,:,0])/255.

    return frame

def preprocess_video(vid, output_file_name, sigma, alpha, beta = 'avg'):
    writer = imageio.get_writer(output_file_name, fps = 30)

    tf_start = 0
    tf_stop = vid._meta['nframes']

    if beta == 'avg':
        frame = get_frame(vid,0)
        beta = frame.sum()/(frame.shape[0]*frame.shape[1])

    for tf in xrange(0, tf_stop):
        frame = get_frame(vid,tf)
        frame = frame*alpha + (.5-beta)
        writer.append_data(frame)

    writer.close()

    return




def change_frame_contrast(frame, alpha, beta):
    frame = frame*alpha + beta
    return frame

def preprocess_frame(frame, sigma, alpha, beta):
    # Gaussian filter
    if sigma != None and sigma != 0:
        frame = gaussian_filter(frame, sigma = sigma)

    frame = frame*alpha + beta

    return frame

def plot_frame_number(vid, tf):
    """
    * Description: Plots the tf'th frame from video source vid
    * Return: None
    * Arguments:
        - vid: imageio video source
        - tf: Integer value of frame to plot (Int)
    """
    fig=plt.figure(figsize=(10,8))
    frame = get_frame_vid(vid, tf)
    plt.imshow(frame, cmap='gray')
    plt.xlim(0, frame.shape[1])
    plt.ylim(0, frame.shape[0])
    plt.show()
    return

def plot_highlighted_frame(frame, cluster_list):
    """
    * Description: Alters frame so pixels within cluster_list clusters are
      highlighted green
    * Return: None
    * Arguments:
        - frame: frame to highlight clusters in
        - cluster_list: List of pixel clusters (e.g. obtained from
          'find_clusters')
    """

    new_frame = np.empty((frame.shape[0], frame.shape[1], 3), dtype=float)
    for i in range(new_frame.shape[0]):
        for j in range(new_frame.shape[1]):
            for k in range(new_frame.shape[2]):
                new_frame[i,j,k]=frame[i,j]


    for cluster in cluster_list:
        for k in xrange(len(cluster)):
            i = cluster[k,0]
            j = cluster[k,1]
            #new_frame[i,j,0] =  0#frame[i,j]/1.  # R
            #new_frame[i,j,1] = 1.#frame[i,j]*5.
            #new_frame[i,j,2] =  0#frame[i,j]/1.                 # B

    fig = plt.figure(figsize = (10, 8))
    plt.imshow(new_frame, vmin = 0, vmax = 1)

    plt.xlim(0, frame.shape[1])
    plt.ylim(0, frame.shape[0])

    plt.show()

    return


def find_clusters(frame, template_frame, threshold_difference = 15,
                  cluster_threshold = 20):
    """
    * Description: Calling function to start a recursive search for
      clusters of differing pixels between a frame and template frame.
    * Return: List of pixel clusters ('cluster_list', List [] of 2-D numpy
      arrays)
    * Arguments:
        - frame: The frame to find clusters in
        - template_frame: The frame to compare to
        - threshold_difference (optional): Minimum difference in pixel
          brightness for pixel to be flagged
        - cluster_threshold (optional): Minimum number of pixels in a cluster
          for cluster to be considered
    """
    negative_frame=abs(frame-template_frame)
    cluster_list = []
    pixel_check_array=np.ones((negative_frame.shape[0], negative_frame.shape[1]))

    for i in xrange(pixel_check_array.shape[0]):
        for j in xrange(pixel_check_array.shape[1]):
            if pixel_check_array[i,j]==1:
                if negative_frame[i,j] >= threshold_difference:
                    cluster_pixels, pixel_check_array=add_pixel_to_cluster\
                    (negative_frame, pixel_check_array, i, j, threshold_difference)

                    if cluster_pixels.shape[0]>cluster_threshold:
                        cluster_list.append(cluster_pixels)

    return cluster_list

def find_clusters_percentage_based(frame, template_frame, threshold_difference = .01,
                  cluster_threshold = 20):
    """
    * Description: Calling function to start a recursive search for
      clusters of differing pixels between a frame and template frame.
    * Return: List of pixel clusters ('cluster_list', List [] of 2-D numpy
      arrays)
    * Arguments:
        - frame: The frame to find clusters in
        - template_frame: The frame to compare to
        - threshold_difference (optional): Minimum difference in pixel
          brightness for pixel to be flagged
        - cluster_threshold (optional): Minimum number of pixels in a cluster
          for cluster to be considered
    """
    negative_frame=abs(frame-template_frame)
    cluster_list = []
    pixel_check_array=np.ones((negative_frame.shape[0], negative_frame.shape[1]))

    for i in xrange(pixel_check_array.shape[0]):
        for j in xrange(pixel_check_array.shape[1]):
            if pixel_check_array[i,j]==1:
                #if negative_frame[i,j]/(template_frame[i,j]) >= threshold_difference:
                if negative_frame[i,j] >= threshold_difference:
                    cluster_pixels, pixel_check_array=add_pixel_to_cluster\
                    (negative_frame, pixel_check_array, i, j, threshold_difference)

                    if cluster_pixels.shape[0]>cluster_threshold:
                        cluster_list.append(cluster_pixels)

    return cluster_list



def add_pixel_to_cluster(negative_frame, pixel_check_array, i, j,
                         threshold_difference,
                         cluster_pixels=np.empty((0,2), dtype=int),
                         direction='center'):
    """
    * Description: Recursive function that will continuously absorb adjacent
      pixels into a cluster if they exceed the threshold_difference. Called
      by 'find_clusters'.
    * Return: List of pixels in cluster, array of pixels that have already
      between checked (('cluster_pixels', 'pixel_check_array'), List [] of
      pixel coordinates, array [[], []] of binary values (1 or 0))
    * Arguments:
        - negative_frame: The frame to look in ('negative' means the template
          frame has been subtracted from the frame)
        - pixel_check_array: Array of binary values that describe whether the
          pixel element has been checked for clustering
        - i: row of pixel that is to be added to cluster
        - j: column of pixel that is to be added to cluster
        - threshold_difference: Minimum difference in pixel brightness for pixel
          to be flagged
        - cluster_pixels (optional): List containing coordinates of pixels
          that have already been added to cluster
        - direction (optional): Direction of next pixel to check
    """

    cluster_pixels=np.vstack((cluster_pixels,[i,j]))
    pixel_check_array[i,j]=0

    # Center (first point)
    if direction == 'center':
        pixel_check_array[i,j]=0

    # Right
    if (direction != 'right' and j != negative_frame.shape[1] -1
        and pixel_check_array[i,j+1] == 1):
        if negative_frame[i,j+1]>=threshold_difference:
            cluster_pixels, pixel_check_array=add_pixel_to_cluster(
                negative_frame, pixel_check_array, i, j+1, threshold_difference,
                cluster_pixels, 'left')

    # Below
    if (direction != 'below' and i != negative_frame.shape[0] - 1
        and pixel_check_array[i+1,j] == 1):
        if negative_frame[i+1,j]>=threshold_difference:
            cluster_pixels, pixel_check_array=add_pixel_to_cluster(
                negative_frame, pixel_check_array, i+1, j, threshold_difference,
                cluster_pixels, 'above')

    # Left
    if direction != 'left' and j != 0 and pixel_check_array[i,j-1] == 1:
        if negative_frame[i,j-1]>=threshold_difference:
            cluster_pixels, pixel_check_array=add_pixel_to_cluster(
                negative_frame, pixel_check_array, i, j-1, threshold_difference,
                cluster_pixels, 'right')

    # Above
    if direction != 'above' and i != 0 and pixel_check_array[i-1,j] == 1:
        if negative_frame[i-1,j]>=threshold_difference:
            cluster_pixels, pixel_check_array=add_pixel_to_cluster(
                negative_frame, pixel_check_array, i-1, j, threshold_difference,
                cluster_pixels, 'below')

    return cluster_pixels, pixel_check_array

def match_events_to_detections(active_events, inactive_events, detections, tf, tracking_threshold = 5):
    distance_threshold = 20

    new_active_events = []

    # Create, fill distance_matrix
    distance_matrix = np.zeros((len(detections)+1, len(active_events)+1),
                               dtype = float)

    for i in xrange(distance_matrix.shape[0]):
        distance_matrix[i,0] = i - 1
    for j in xrange(distance_matrix.shape[1]):
        distance_matrix[0,j] = j - 1

    for i in xrange(1, len(detections) + 1):
        for j in xrange(1, len(active_events) + 1):
            xi = detections[i-1]._px

            yi = detections[i-1]._py

            xj = (active_events[j-1]._detections[-1]._px )
            #+
            #      active_events[j-1]._detections[-1]._pvx *
            #      (tf - active_events[j-1]._detections[-1]._tf))



            yj = (active_events[j-1]._detections[-1]._py)
            # +
            #      active_events[j-1]._detections[-1]._pvy *
            #                  (tf - active_events[j-1]._detections[-1]._tf))

            distance_matrix[i,j] = ((xi - xj)**2. + (yi - yj)**2.)**.5

    # Match columns to row
    check_rows = [i for i in xrange(0, len(detections))]

    for j in xrange(0, len(active_events)):

        # Get index of row with minimum distance
        if check_rows != []:
            temp_min_row = np.argmin(distance_matrix
                                     [[row+1 for row in check_rows], j+1], 0)

            min_row = int(distance_matrix[[row+1 for row in check_rows],[0]]
                                          [temp_min_row])

            min_col = np.argmin(distance_matrix[min_row+1, 1:])


            # Check to see if point still good
            # (i.e., if the minimum makes sense)

            if ((distance_matrix[min_row+1, j+1] >= distance_threshold) or
                (min_col != j)):
                min_row = None

        else:
            min_row = None

        # Check if match is found
        if min_row != None:
            # Remove row from check list
            check_rows = [row for row in check_rows if row != min_row]

            # Update the particle that is being tracked with the
            # matched particle
            active_events[j].add_detection(detections[min_row])

            # Add the particle to new_active_events list
            new_active_events.append(active_events[j])

        else:
            # Added to help particles that are temporarily 'lost' to be found again
            if tf - active_events[j]._detections[-1]._tf > tracking_threshold:
                inactive_events.append(active_events[j])
            else:
                new_active_events.append(active_events[j])

    # Add rows that did not match to new_active_events
    for row in check_rows:
        new_active_events.append(OpticalEvent([detections[row]]))

    return new_active_events, inactive_events



def connect_loose_events(events_, tf_sep_threshold = 5, dist_threshold = 20):
    moves = -1
    events = [event for event in events_]
    while moves != 0:

        moves = 0
        i_match = None

        for i, event1 in enumerate(events):
            j_match = None

            # First loop
            distance = np.array([float("inf") for m in xrange(len(events))], dtype=float)
            for j, event2 in enumerate(events):

                if (event1._detections[-1]._tf < event2._detections[0]._tf or
                    event1._detections[0]._tf > event2._detections[-1]._tf):
                    overlap = False
                else:
                    overlap = True

                if overlap == False:
                    if event1._detections[-1]._tf < event2._detections[0]._tf:
                        tf_sep = event2._detections[0]._tf - event1._detections[-1]._tf
                        if tf_sep <= tf_sep_threshold:
                            distance[j] = ((event1._detections[-1]._px - event2._detections[0]._px)**2.+
                                           (event1._detections[-1]._py - event2._detections[0]._py)**2.+
                                           (event1._detections[-1]._tf - event2._detections[0]._tf)**2.)**.5


                    else:
                        tf_sep = event1._detections[0]._tf - event2._detections[-1]._tf
                        if tf_sep <= tf_sep_threshold:
                            distance[j] = ((event2._detections[-1]._px - event1._detections[0]._px)**2.+
                                           (event2._detections[-1]._py - event1._detections[0]._py)**2.+
                                           (event2._detections[-1]._tf - event1._detections[0]._tf)**2.)**.5



            j_match = np.argsort(distance)[0]
            eventj = events[j_match]

            # Second loop
            distance = np.array([float("inf") for m in xrange(len(events))], dtype=float)
            for k, event3 in enumerate(events):

                if (eventj._detections[-1]._tf < event3._detections[0]._tf or
                    eventj._detections[0]._tf > event3._detections[-1]._tf):
                    overlap = False
                else:
                    overlap = True

                if overlap == False:
                    if eventj._detections[-1]._tf < event3._detections[0]._tf:
                        tf_sep = event3._detections[0]._tf - eventj._detections[-1]._tf
                        if tf_sep <= tf_sep_threshold:
                            distance[k] = ((eventj._detections[-1]._px - event3._detections[0]._px)**2.+
                                           (eventj._detections[-1]._py - event3._detections[0]._py)**2.)**.5

                    else:
                        tf_sep = eventj._detections[0]._tf - event3._detections[-1]._tf
                        if tf_sep <= tf_sep_threshold:
                            distance[k] = ((event3._detections[-1]._px - eventj._detections[0]._px)**2.+
                                           (event3._detections[-1]._py - eventj._detections[0]._py)**2.)**.5


            i_match = np.argsort(distance)[0]
            if ((i_match == i) and (distance[i_match] <= dist_threshold)):
                event1.connect_event(eventj)
                events = [events[k] for k in xrange(0, len(events)) if k != j_match]
                moves = moves + 1

            else:
                pass

            if moves != 0:
                break

    return events




def find_events_vid(file_name, threshold_difference = 30, cluster_threshold = 20,
                tf_start = -1, tf_stop = -1, template_frame = None,
                sigma = None, alpha = None, beta = None):
    """
    * Description: Finds all events within optical imaging data. An event
      is defined as the entrance and exit of a particle (represented as a
      cluster of pixels) from the 'stage'.
    * Return: A list of OpticalEvent ('optical_events': List [] of OpticalEvent)
    * Arguments:
        - file_name: The file_name of .mp4 file
        - threshold_difference (optional): The difference in pixel brightness
          for pixel to register as belonging to a particle
        - cluster_threshold (optional): Minimum pixel area for cluster to be
          considered
        - tf_start (optional): Frame index at which to begin search (will start
          at first frame if left default)
        - tf_stop (optional): Frame index at which to end search (will stop at
          last frame if left default)
    """
    # Open video connection
    vid = open_video_connection(file_name)

    # Get start and stop frame numbers
    if tf_start == -1:
        tf_start = 0
    if tf_stop == -1:
        tf_stop = vid._meta['nframes']

    # Define template frame
    if template_frame == None:
        template_frame = get_frame(vid, tf_start)
    template_frame = change_frame_contrast(template_frame, alpha, beta)

    active_events=[]
    inactive_events=[]

    # Search frames for clusters
    for tf in xrange(tf_start+1, tf_stop):

        frame = get_frame(vid, tf)
        frame = change_frame_contrast(frame, alpha, beta)

        clusters = find_clusters_percentage_based(
                                   frame, template_frame,
                                   threshold_difference = threshold_difference,
                                   cluster_threshold = cluster_threshold)



        detections = [OpticalDetection(tf, cluster) for cluster in
                              clusters]

        if tf % 100 == 0:
            print 'tf: ', tf, '/', tf_stop, '\tnum detections:', len(detections), '\tnum clusters:', len(clusters)

        active_events, inactive_events = match_events_to_detections(
                                             active_events, inactive_events,
                                             detections, tf)

    # Append all events that are still active by last frame to inactive_events
    # list
    for active_event in active_events:
        inactive_events.append(active_event)


    return inactive_events



def find_events_bvi(filepath, threshold_difference = .0375, cluster_threshold = 20,
                tf_start = -1, tf_stop = -1, template_frame = None):
    """
    * Description: Finds all events within optical imaging data. An event
      is defined as the entrance and exit of a particle (represented as a
      cluster of pixels) from the 'stage'.
    * Return: A list of OpticalEvent ('optical_events': List [] of OpticalEvent)
    * Arguments:
        - file_name: The file_name of .mp4 file
        - threshold_difference (optional): The difference in pixel brightness
          for pixel to register as belonging to a particle
        - cluster_threshold (optional): Minimum pixel area for cluster to be
          considered
        - tf_start (optional): Frame index at which to begin search (will start
          at first frame if left default)
        - tf_stop (optional): Frame index at which to end search (will stop at
          last frame if left default)
    """


    # Open video connection
    filehandle = open(filepath, 'rb')



    dim0 = 256#128
    dim1 = 512#256

    bytes_per_frame = 4*dim0*dim1

    # Get start and stop frame numbers
    if tf_start == -1:
        tf_start = 0
    if tf_stop == -1:
        size = os.path.getsize(filepath)
        tf_stop = int(size/(dim0*dim1*4))
        #tf_stop = oi_file.get_filelength_bvi(filepath)

    # Define template frame

    data = np.memmap(filepath, mode = 'r', dtype = 'float32', offset = tf_start*bytes_per_frame,\
     shape = (tf_stop-tf_start, dim0, dim1))

    if template_frame == None:
        template_frame = copy.copy(data[0,:,:])


    active_events=[]
    inactive_events=[]



    # Search frames for clusters
    for tf in xrange(tf_start+1, tf_stop):

        frame = data[tf-tf_start,:,:]




        clusters = find_clusters_percentage_based(
                                   frame, template_frame,
                                   threshold_difference = threshold_difference,
                                   cluster_threshold = cluster_threshold)



        detections = [OpticalDetection(tf, cluster) for cluster in
                              clusters]

        if tf % 100 == 0:
            print 'tf: ', tf, '/', tf_stop, '\tactive:', len(active_events), '\tinactive:', len(inactive_events)

        active_events, inactive_events = match_events_to_detections(
                                             active_events, inactive_events,
                                             detections, tf)



    # Append all events that are still active by last frame to inactive_events
    # list
    for active_event in active_events:
        inactive_events.append(active_event)


    return inactive_events



def plot_trajectory(template_frame, event_oi):
    plt.imshow(template_frame, alpha = 0.5, cmap = 'gray')
    plt.plot(event_oi.get_px(), event_oi.get_py())
    plt.show()

    return


















"""
OLD/OBSOLETE CODE
"""


"""
def match_events_to_detections(active_events, inactive_events,
                               detections, tf):

    stage_boundary = 350
    distance_threshold = 10

    new_active_events = []

    # Create, fill distance_matrix
    distance_matrix = np.zeros((len(detections)+1, len(active_events)+1),
                               dtype = float)

    for i in range(distance_matrix.shape[0]):
        distance_matrix[i,0] = i - 1
    for j in range(distance_matrix.shape[1]):
        distance_matrix[0,j] = j - 1

    for i in range(1, len(detections) + 1):
        for j in range(1, len(active_events) + 1):
            xi = detections[i-1]._px

            yi = detections[i-1]._py

            xj = (active_events[j-1]._detections[-1]._px +
                  active_events[j-1]._detections[-1]._pvx *
                  (tf - active_events[j-1]._detections[-1]._tf))



            yj = (active_events[j-1]._detections[-1]._py +
                  active_events[j-1]._detections[-1]._pvy *
                  (tf - active_events[j-1]._detections[-1]._tf))

            distance_matrix[i,j] = ((xi - xj)**2. + (yi - yj)**2.)**.5

    # Match columns to row
    check_rows = [i for i in range(0, len(detections))]

    for j in range(0, len(active_events)):

        # Get index of row with minimum distance
        if check_rows != []:
            temp_min_row = np.argmin(distance_matrix
                                     [[row+1 for row in check_rows], j+1], 0)

            min_row = int(distance_matrix[[row+1 for row in check_rows],[0]]
                                          [temp_min_row])


            # Check to see if point still good
            # (i.e., if the minimum makes sense)
            #if (active_events[j]._detections[-1]._px >= stage_boundary and
                #distance_matrix[min_row+1, j+1] >= distance_threshold):
                #min_row = None

            if distance_matrix[min_row+1, j+1] >= distance_threshold:
                min_row = None

        else:
            min_row = None

        # Check if match is found
        if min_row != None:
            # Remove row from check list
            check_rows = [row for row in check_rows if row != min_row]

            # Update the particle that is being tracked with the
            # matched particle
            active_events[j].add_detection(detections[min_row])

            # Add the particle to new_active_events list
            new_active_events.append(active_events[j])

        else:
            inactive_events.append(active_events[j])

    # Add rows that did not match to new_active_events
    for row in check_rows:
        new_active_events.append(OpticalEvent(detections[row]))

    return new_active_events, inactive_events

"""




#class OpticalEvent:
"""
    * Description: Contains all data associated with passage of particle
      through the optical stage.
    * Member variables:
        - _tf: List of frame indices that event is active (list [] of Int)
        - _pixels: Time-series data of pixel coordinates, the pixel coordinates
          being the location of the particle in (row, column) format at a
          particular time (list [] of 2-d numpy arrays)
        - _px: Time-series of event's x (column) position in pixels (list [] of
          Int)
        - _py: Time-series of event's y (row) position in pixels (list [] of
          Int)
        - _parea: Time-series of pixel areas of particle (list [] of Int)
        - _pwidth: Time-series of pixel widths of particle (list [] of Int)
        - _pheight: Time-series of pixel heights of particle (list [] of Int)
    * Member functions:
        - merge_events()

"""
"""
    def __init__(self, tf, pixels):
        self._tf = []
        self._pixels = []
        self._px = []
        self._py = []
        self._pvx = []
        self._pvy = []
        self._parea = []
        self._pwidth = []
        self._pheight = []

        self.add_frame_data(tf, pixels)

        return

    def add_frame_data(self, tf, pixels):
        self._tf.append(tf)
        self._pixels.append(pixels)

        px = (pixels[:,1].max()+pixels[:,1].min())/2
        py = (pixels[:,0].max()+pixels[:,0].min())/2
        self._px.append(px)
        self._py.append(py)

        if len(self._tf) >= 2:
            pvx = (1.*(self._px[-1]-self._px[-2])/
                      (self._tf[-1]-self._tf[-2]))
            pvy = (1.*(self._py[-1]-self._py[-2])/
                      (self._tf[-1]-self._tf[-2]))
        else:
            pvx=0
            pvy=0
        self._pvx.append(pvx)
        self._pvy.append(pvy)


        parea = len(pixels)
        pwidth = pixels[:,1].max() - pixels[:,1].min()
        pheight = pixels[:,0].max() - pixels[:,0].min()
        self._parea.append(parea)
        self._pwidth.append(pwidth)
        self._pheight.append(pheight)

        return

    def connect_event(self, optical_event):
        if self._tf[-1] < optical_event._tf[0]:
            first_event = self
            second_event = optical_event
        else:
            first_event = optical_event
            second_event = self


        self._tf = first_event._tf + second_event._tf
        self._pixels = first_event._pixels + second_event._pixels
        self._px = first_event._px + second_event._px
        self._py = first_event._py + second_event._py
        self._pvx = first_event._pvx + second_event._pvx
        self._pvy = first_event._pvy + second_event._pvy
        self._parea = first_event._parea + second_event._parea
        self._pwidth = first_event._pwidth + second_event._pwidth
        self._pheight = first_event._pheight + second_event._pheight

        return
"""
"""
def find_clusters_percentage_based(frame, template_frame,
                                   threshold_percentage_difference = 0.1,
                                   cluster_threshold = 20):

    * Description: Calling function to start a recursive search for
      clusters of differing pixels between a frame and template frame.
    * Return: List of pixel clusters ('cluster_list', List [] of 2-D numpy
      arrays)
    * Arguments:
        - frame: The frame to find clusters in
        - template_frame: The frame to compare to
        - threshold_difference (optional): Minimum difference in pixel
          brightness for pixel to be flagged
        - cluster_threshold (optional): Minimum number of pixels in a cluster
          for cluster to be considered

    negative_frame=np.empty((frame.shape), dtype=int)
    negative_frame=abs(frame-template_frame)
    cluster_list = []
    pixel_check_array=np.ones((negative_frame.shape[0], negative_frame.shape[1]))

    for i in range(pixel_check_array.shape[0]):
        for j in range(pixel_check_array.shape[1]):
            if pixel_check_array[i,j]==1:
                percentage_difference = (1.*negative_frame[i,j]/
                                            (1+template_frame[i,j]))
                if percentage_difference >= threshold_percentage_difference:
                    cluster_pixels, pixel_check_array=(
                    add_pixel_to_cluster_percentage_based
                     (negative_frame, template_frame, pixel_check_array, i, j,
                      threshold_percentage_difference))

                    if cluster_pixels.shape[0]>cluster_threshold:
                        cluster_list.append(cluster_pixels)

    return cluster_list
"""
"""
def add_pixel_to_cluster_percentage_based(negative_frame, template_frame, pixel_check_array, i, j,
                         threshold_percentage_difference,
                         cluster_pixels=np.empty((0,2), dtype=int),
                         direction='center'):

    * Description: Recursive function that will continuously absorb adjacent
      pixels into a cluster if they exceed the threshold_difference. Called
      by 'find_clusters'.
    * Return: List of pixels in cluster, array of pixels that have already
      between checked (('cluster_pixels', 'pixel_check_array'), List [] of
      pixel coordinates, array [[], []] of binary values (1 or 0))
    * Arguments:
        - negative_frame: The frame to look in ('negative' means the template
          frame has been subtracted from the frame)
        - pixel_check_array: Array of binary values that describe whether the
          pixel element has been checked for clustering
        - i: row of pixel that is to be added to cluster
        - j: column of pixel that is to be added to cluster
        - threshold_difference: Minimum difference in pixel brightness for pixel
          to be flagged
        - cluster_pixels (optional): List containing coordinates of pixels
          that have already been added to cluster
        - direction (optional): Direction of next pixel to check

    cluster_pixels=np.vstack((cluster_pixels,[i,j]))
    pixel_check_array[i,j]=0

    # Center (first point)
    if direction == 'center':
        pixel_check_array[i,j]=0

    # Right
    if (direction != 'right' and j != negative_frame.shape[1] -1
        and pixel_check_array[i,j+1] == 1):
        if 1.*negative_frame[i,j+1]/(1.+template_frame[i,j+1])>=threshold_percentage_difference:
            cluster_pixels, pixel_check_array=add_pixel_to_cluster_percentage_based(
                negative_frame, template_frame, pixel_check_array, i, j+1, threshold_percentage_difference,
                cluster_pixels, 'left')

    # Below
    if (direction != 'below' and i != negative_frame.shape[0] - 1
        and pixel_check_array[i+1,j] == 1):
        if 1.*negative_frame[i+1,j]/(1.+template_frame[i+1,j])>=threshold_percentage_difference:
            cluster_pixels, pixel_check_array=add_pixel_to_cluster_percentage_based(
                negative_frame, template_frame, pixel_check_array, i+1, j, threshold_percentage_difference,
                cluster_pixels, 'above')

    # Left
    if direction != 'left' and j != 0 and pixel_check_array[i,j-1] == 1:
        if 1.*negative_frame[i,j-1]/(1.+template_frame[i,j-1])>=threshold_percentage_difference:
            cluster_pixels, pixel_check_array=add_pixel_to_cluster_percentage_based(
                negative_frame, template_frame, pixel_check_array, i, j-1, threshold_percentage_difference,
                cluster_pixels, 'right')

    # Above
    if direction != 'above' and i != 0 and pixel_check_array[i-1,j] == 1:
        if 1.*negative_frame[i-1,j]/(1.+template_frame[i-1,j])>=threshold_percentage_difference:
            cluster_pixels, pixel_check_array=add_pixel_to_cluster_percentage_based(
                negative_frame, template_frame, pixel_check_array, i-1, j, threshold_percentage_difference,
                cluster_pixels, 'below')

    return cluster_pixels, pixel_check_array
"""
"""

def match_events(active_events, new_cluster_list, inactive_events, tf):

    * Description: Matches a list of cluster pixels to events from previous
      frames. This is necessary to track individual particles across frames.
    * Return: Returns two lists: list of active events and list of inactive
      events ('active_events', 'inactive_events': Lists [] of OpticalEvent)
    * Arguments:
        - active_events: The list of events that are ongoing that we will
          update
        - new_cluster_list: The list of clusters of pixels that we just found
          to be particles.
        - inactive_events: The list of events that have previously occurred that
          we will append to if a particle exits the scene
        - tf: The integer value of the current frame to be analyzed

    stage_boundary = 350
    distance_threshold = 10

    new_active_events = []

    # Create, fill distance_matrix
    distance_matrix = np.zeros((len(new_cluster_list)+1, len(active_events)+1),
                               dtype = float)

    for i in range(distance_matrix.shape[0]):
        distance_matrix[i,0] = i - 1
    for j in range(distance_matrix.shape[1]):
        distance_matrix[0,j] = j - 1

    for i in range(1, len(new_cluster_list) + 1):
        for j in range(1, len(active_events) + 1):
            xi = int((new_cluster_list[i-1][:,1].max()+
                  new_cluster_list[i-1][:,1].min())/2)

            yi = int((new_cluster_list[i-1][:,0].max()+
                  new_cluster_list[i-1][:,0].min())/2)

            xj = int(active_events[j-1]._px[-1]+
                  active_events[j-1]._pvx[-1]*
                  (tf - active_events[j-1]._tf[-1]))

            yj = int(active_events[j-1]._py[-1]+
                  active_events[j-1]._pvy[-1]*
                  (tf - active_events[j-1]._tf[-1]))

            distance_matrix[i,j] = ((xi - xj)**2. + (yi - yj)**2.)**.5

    # Match columns to row
    check_rows = [i for i in range(0, len(new_cluster_list))]

    for j in range(0,len(active_events)):

        # Get index of row with minimum distance
        if check_rows != []:
            temp_min_row = np.argmin(distance_matrix
                                     [[row+1 for row in check_rows], j+1], 0)

            min_row = int(distance_matrix[[row+1 for row in check_rows],[0]]
                                          [temp_min_row])


            # Check to see if point still good
            # (i.e., if the minimum makes sense)
            #if (active_events[j]._px[-1] >= stage_boundary and
        #        distance_matrix[min_row+1, j+1] >= distance_threshold):
        #        min_row = None

            if distance_matrix[min_row+1, j+1] >= distance_threshold:
                min_row = None

        else:
            min_row = None

        # Check if match is found
        if min_row != None:
            # Remove row from check list
            check_rows = [row for row in check_rows if row != min_row]

            # Update the particle that is being tracked with the
            # matched particle
            active_events[j].add_frame_data(tf,
                                            new_cluster_list[min_row])

            # Add the particle to new_active_events list
            new_active_events.append(active_events[j])

        else:
            inactive_events.append(active_events[j])

    # Add rows that did not match to new_active_events
    for row in check_rows:
        new_active_events.append(OpticalEvent(tf, new_cluster_list[row]))

    return new_active_events, inactive_events







def find_events_windows(file_name, windows, threshold_difference = 30,
                        cluster_threshold = 20):
    """
"""

    vid = open_video_connection(file_name)

    template_frame = get_frame(vid, 0)
    events = []
    for i, window in enumerate(windows):
        print 'window:', i, window[0], window[1]

        tf_start = window[0]
        tf_stop = window[1]

        events = (events +
                  find_events(
                              file_name,
                              threshold_difference = threshold_difference,
                              cluster_threshold = cluster_threshold,
                              template_frame = template_frame,
                              tf_start = window[0],
                              tf_stop = window[1]))

    return events

"""
