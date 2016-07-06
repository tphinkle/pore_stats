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

"""
Constants
"""
FFMPEG_BIN_FILENAME = '/home/preston/ffmpeg-3.0.2-64bit-static/ffmpeg'


"""
Classes
"""

class OpticalEvent:
    """
    * Description: Contains all data associated with passage of particle
      through the optical stage.
    * Member variables:
        - _data: Time-series data of pixel coordinates, the pixel coordinates
          being the location of the particle in (row, column) format at a
          particular time (list [] of 2-d numpy arrays)
        - _area: Time-series of pixel areas of particle (list [] of Int)
        - _width: Time-series of pixel widths of particle (list [] of Int)
        - _height: Time-series of pixel heights of particle (list [] of Int)
    * Member functions:

    """

    def __init__(self):
        self._data = None
        self._area = None
        self._width = None
        self._height = None
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
        - channel_plength: Length of channel in pixels (Int)
        - channel_pwidth_large: Largest width of channel in pixels (Int)
        - channel_pwidth_small: Smallest width of channel in pixels (Int)
        - channel_pdepth: Depth of channel in pixels (Int)
        - channel_pcorner_0: Pixel coordinates of top left corner of channel_pdepth
          (2 element numpy array of Int)
        - channel_pcorner_1: Pixel coordinates of bottom left corner of channel_pdepth
          (2 element numpy array of Int)
        - channel_pcorner_2: Pixel coordinates of bottom right corner of channel_pdepth
          (2 element numpy array of Int)
        - channel_pcorner_3: Pixel coordinates of top right corner of channel_pdepth
          (2 element numpy array of Int)
        - _template_image: Pixel data of the stage without any OpticalEvents
          occurring (2-D numpy array of RGB pixel data (3 element numpy array))
    """

    channel_length = 125
    channel_width_large = 35
    channel_width_small = 25
    channel_depth = 45

    def __init__(self):
        self._channel_plength=None
        self._channel_pwidth_large=None
        self._channel_pwidth_small=None
        self._channel_pdepth=None
        self._channel_pcorner_0=None
        self._channel_pcorner_1=None
        self._channel_pcorner_2=None
        self._channel_pcorner_3=None
        self._channel_template_image=None



"""
Functions
"""

def open_video_connection(file_name):
    """
    * Description: Opens video connection so frame data can be retrieved from
      .MP4 file with H.264 encoding.
    * Return: Video connection ('vid': imageio.reader)
    * Arguments:
        - file_name: Name of .MP4 file
    """

    return imageio.get_reader(file_name, 'ffmpeg')

def get_frame(vid, n):
    """
    * Description: Gets the nth frame of data from video stream. Converts pixel
      RGB values to type Int.
    * Return: Frame data (2-D numpy array of RGB pixel data (3 element
      numpy array))
    * Arguments:
        - vid: Video connection established by imageio class (use
          'open_video_connection') to get
        - n: Number of desired frame to retrieved
    """
    frame=np.array(vid.get_data(n)[:,:,0], dtype=int)

    return frame

def plot_frame(frame):
    """
    * Description: A simple wrapper for matplotlib.pyplot.imshow()
    * Return: None
    * Arguments:
        - frame: Frame data (2-D numpy array of RGB pixel data (3 element
          numpy array))
    """

    plt.imshow(frame, vmin=0, vmax=255, cmap='gray')
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
                new_frame[i,j,k]=frame[i,j]/255.

    for cluster in cluster_list:
        for k in range(len(cluster)):
            i = cluster[k,0]
            j = cluster[k,1]
            new_frame[i,j,0] =  0   # R
            new_frame[i,j,1] =  1.  # G
            new_frame[i,j,2] =  0   # B

    plt.imshow(new_frame)
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
    negative_frame=np.empty((frame.shape), dtype=int)
    negative_frame=abs(frame-template_frame)
    cluster_list = []
    pixel_check_array=np.ones((negative_frame.shape[0], negative_frame.shape[1]))

    for i in range(pixel_check_array.shape[0]):
        for j in range(pixel_check_array.shape[1]):
            if pixel_check_array[i,j]==1:
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
