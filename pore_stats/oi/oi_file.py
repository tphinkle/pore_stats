"""
Optical imaging

* Contains tools for opening files related to optical imaging, including video files
and event files.
* Currently only uses the .cine format for video files; but will move to a generalized
binary format in the future.
* OIEvents are contained in formatted .json files; older plain text files are obsolete.
"""


"""
Imports
"""

# Standard library
import sys
import csv
import struct
import copy
import json
from array import array
import os

# pore_stats specific
import optical_imaging as oi

# Scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

# CV
import cv2







"""
Classes
"""

class Video():
    """
    - Contains member variables and functions used to work with raw binary files.
    - For now, user has to supply the resolution, sampling frequency, and
    exposure time.
    """

    def __init__(self, file_path, image_width, image_height,
                 fps = 0, exp_time = 0):
        self._file_path = file_path
        self._file_handle = open(file_path, 'rb')
        self._image_width = image_width
        self._image_height = image_height
        self._bytes_per_frame = self._image_width * self._image_height
        self._fps = fps
        self._exp_time = exp_time

        self._total_frames = os.path.getsize(self._file_path)/\
        (self._image_width*self._image_height)


    def get_frame(self, frame, average = False, blur = False):
        self._file_handle.seek(int(frame*self._bytes_per_frame))
        frame = np.fromfile(self._file_handle, dtype = np.uint8, count = self._bytes_per_frame)

        # Gaussian blur the frame
        if blur:
            frame = cv2.GaussianBlur(frame,(5,5),0)


        frame = frame.reshape(self._image_height, self._image_width)
        frame = frame/255.

        # Convenience function for shifting pixel values to mean
        # Mean is determined on the boolean state of norm, either .5 or 127.5

        if average == True:
            mean = int(norm)*.5 + int(not norm)*255./2
            frame = frame + (mean-np.mean(frame))
            frame[frame > 1] = 1
            frame[frame < 0] = 0





        return frame



class Cine():
    """
    - Contains member variables and functions used to work with .cine files.
    - .cine files are a Vision Research proprietary file system for storing grayscale images
    and meta data in a binary format.
    - Some info about the file structure can be found at
      https://wiki.multimedia.cx/index.php/Phantom_Cine
    """

    def __init__(self, file_path):

        self._file_path = file_path

        self._file_handle = open(file_path, 'rb')

        # Read header
        self._file_handle.seek(0)
        header = self._file_handle.read(44)

        # This byte order is specific to the header
        header = struct.unpack('< 2c 2c 2c 2c I I I I I I I 2I', header[:44])

        self._total_frames = header[9]
        off_image_offset = header[14]

        # Read bitmap info
        bitmapinfo = self._file_handle.read(40)
        bitmapinfo = struct.unpack('< I 2l 2H 2I 2l 2I', bitmapinfo)

        self._image_width = bitmapinfo[1]
        self._image_height = bitmapinfo[2]
        self._bytes_per_frame = self._image_width * self._image_height

        # Get first image pointer
        self._file_handle.seek(off_image_offset)
        self._first_image_byte = self._file_handle.read(4)
        self._first_image_byte = struct.unpack('< I', self._first_image_byte)[0]



    def get_frame(self, t, norm = True, average = False):

        """
        * Description:
            - Returns a frame from a loaded .cine file.
            - Contains a convenience function to automatically rescale the intensity
            values.

        * Return:
            - Frame
        * Arguments:
            - t: The frame index
            - norm: Normalize the grayscale values (0,255)->(0,1)
            - average: Set the frame intensity after to 0.5
        """


        # plus 8 bytes
        # of meta data per frame
        frame_byte = self._first_image_byte + t*(self._bytes_per_frame + 8)
        self._file_handle.seek(frame_byte)

        frame = np.fromfile(self._file_handle, dtype = np.dtype('u1'), count = self._bytes_per_frame)

        frame = frame.reshape(self._image_height, self._image_width)
        frame = frame/255.

        # Convenience function for shifting pixel values to mean
        # Mean is determined on the boolean state of norm, either .5 or 127.5
        mean = int(norm)*.5 + int(not norm)*255./2
        if average == True:
            frame = frame + (mean-np.mean(frame))

        frame[frame > 1] = 1
        frame[frame < 0] = 0

        return frame



"""
Functions
"""

def make_animation(vid, t0, t1):

    template_frame = vid.get_frame(0)
    dim = template_frame.shape

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()


    plot = ax.imshow(vid.get_frame(t0), cmap = 'gray', origin = 'lower')

    def init():
        plot.set_data(vid.get_frame(t0))
        plt.xticks([])
        plt.yticks([])
        #plt.xlim(100, 412)
        #plt.ylim(200, 288)
        return (plot,)


    # animation function. This is called sequentially
    texts = []
    def animate(i):
        plot.set_data(vid.get_frame(i))
        new_text = plt.text(0.0, 0.0, 'frame='+str(i-t0)+'/'+str(t1-t0)+'\nt='+str(1000.*(i-t0)/50000.)+'ms',\
         transform = ax.transAxes, color = 'red', size = 20, ha = 'left', va = 'bottom')

        # Hack to replace text... this just sets the old text to invisible
        for text in texts:
            text.set_visible(False)
        texts.append(new_text)
        plt.xticks([])
        plt.yticks([])
        #plt.xlim(50, 490)
        #plt.ylim(95, 288)
        return (plot,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    #anim = matplotlib.animation.FuncAnimation(fig, animate, np.arange(t0, t1), init_func=init, interval=200, blit=True)
    anim = matplotlib.animation.FuncAnimation(fig, animate, np.arange(t0, t1, dtype = np.uint64), interval=200, blit=False)

    return anim

def make_animation_frames(vid, frames):

    template_frame = vid.get_frame(0)
    dim = template_frame.shape

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()


    plot = ax.imshow(vid.get_frame(frames[0]), cmap = 'gray', origin = 'lower')

    def init():
        plot.set_data(vid.get_frame(t0))
        plt.xticks([])
        plt.yticks([])
        plt.xlim(100, 412)
        plt.ylim(200, 288)
        return (plot,)


    # animation function. This is called sequentially
    texts = []
    def animate(i):
        plot.set_data(vid.get_frame(i))
        new_text = plt.text(0, 200, 'frame='+str(i-frames[0]+1)+'/'+str(len(set(frames)))+'\nt='+str(1000.*(i-frames[0])/50000.)+'ms',\
         transform = ax.transAxes, color = 'red', size = 20, ha = 'left', va = 'bottom')
        for text in texts:
            text.set_visible(False)
        texts.append(new_text)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(50, 490)
        plt.ylim(95, 288)
        return (plot,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    #anim = matplotlib.animation.FuncAnimation(fig, animate, np.arange(t0, t1), init_func=init, interval=200, blit=True)
    anim = matplotlib.animation.FuncAnimation(fig, animate, frames, interval=200, blit=False)

    return anim


def save_oi_events_json(file_path, oi_events):
    '''
    * Description:
        - Saves OI events to a .json formatted file.
    * Arguments:
        - output_filepath: The filepath to save to.
        - oi_events: A list of OIEvent (optical_imaging.py) objects
    '''

    with open(file_path, 'w') as fh:

        event_json_list = []

        for i, event in enumerate(oi_events):
            det_list = []
            for j, det in enumerate(event._detections):
                tf = det._tf
                pixels = det._pixels.tolist()
                det_list.append({'tf': tf, 'pixels': pixels})

            event_json_list.append({'id': str(i),
                                    'detections': det_list})

        events = {'events': event_json_list}

        json.dump(events, fh)


def open_event_file_json(file_path):
    """
    * Description:
        - Loads an events .json file.
    * Return:
        - List [] of OIEvent
    * Arguments:
        - file_path: File location.
    """

    events = []

    with open(file_path, 'r') as fh:
        json_reader = json.load(fh)
        for event in json_reader['events']:
            dets = []
            for det in event['detections']:
                tf = det['tf']
                pixels = np.array(det['pixels'])
                dets.append(oi.OpticalDetection(tf, pixels))

            events.append(oi.OpticalEvent(dets))

        return events




"""""""""""""""""""""""""""""""""""""""
Deprecated
"""""""""""""""""""""""""""""""""""""""

"""
Constants
"""
FFMPEG_BIN_FILENAME = '/home/preston/ffmpeg-3.0.2-64bit-static/ffmpeg'

BVI_HEADER_BYTES = 12

"""
Classes
"""

class OIFile(object):
    header_bytes = 2*3
    def __init__(self, file_path = None):
        self._directory = None
        self._file_path = None
        self._file_length = None



"""
Functions
"""

def save_oi_events(output_filepath, oi_events):
    '''
    * Description:
        - Saves OI events to file.
    * Arguments:
        - output_filepath: The filepath to save to.
        - oi_events: A list of OIEvent (optical_imaging.py) objects
    '''

    f = open(output_filepath, 'w')

    f.write(output_filepath+'\n')
    for i, event in enumerate(oi_events):
        f.write('event#\t'+str(i)+'\n')
        for j, detection in enumerate(event._detections):
            f.write(str(int(detection._tf)) + '\t' + str(int(detection._px)) + '\t' + str(int(detection._py)) + '\n')

    f.close()

    return



def load_oi_events(input_filepath):
    """
    * Description:
        - Loads a regular text file containing OIEvents
    * Return:
        - List [] of OIEvent
    * Arguments:
        - input_filepath: Filepath to be opened.
    """


    f = open(input_filepath, 'r')

    reader = csv.reader(f, delimiter = '\t')

    row = reader.next()

    try:
        events = []
        while row:
            row = reader.next()
            if row[0] == 'event#':
                oe = copy.deepcopy(oi.OpticalEvent())
                events.append(oe)


            else:
                od = oi.OpticalDetection()
                od._tf = int(row[0])
                od._px = int(row[1])
                od._py = int(row[2])
                oe.add_detection(od)

    except Exception as inst:
        print 'error!'
        print 'line num = ', sys.exc_info()[2].tb_lineno
        print(type(inst))
        print(inst.args)
        print(inst)
        pass

    return events



def get_filelength_bvi(filepath):
    """
    * Description:
        - Returns the total number of frames in a .bvi file.
    * Return: file_length; the number of frames
    * Arguments:
        - filepath: File location.
    """
    with open(filepath, 'rb') as f:
        header_bytes = f.read(12)


    file_length = struct.unpack('I', header_bytes[8:12])[0]
    print 'file_length = ', file_length

    return file_length

def get_dimensions_bvi(filepath):
    """
    * Description:
        - Gets the image resolution of a .bvi file.
    * Return:
        - dim0: # rows
        - dim1: # cols
    * Arguments:
        - filepath: File location.
    """
    with open(filepath, 'rb') as f:
        header_bytes = f.read(8)

    dim0 = struct.unpack('I', header_bytes[0:4])[0]
    dim1 = struct.unpack('I', header_bytes[4:8])[0]

    print 'dim0 = ', dim0, 'dim1 = ', dim1


    return dim0, dim1

def mp4_to_oi(input_filepath, output_filepath, alpha = 1, beta = 0):
    """
    * Description:
    * Return:
    * Arguments:
        -
    """
    vid = open_video_connection(input_filepath)

    tf_stop = vid._meta['nframes']

    res = get_frame_vid(vid, 0).shape

    output_filehandle = open(output_filepath, 'wb')

    #Header line = Rows    Columns    Frames
    header_array = array('I', [res[0], res[1], tf_stop])
    header_array.tofile(output_filehandle)


    for i in range(tf_stop):
        if i % 100 == 0:
            print 'frame', i, 'out of', tf_stop
        frame = get_frame_vid(vid, i).reshape(-1,1)[:,0]

        frame = change_frame_contrast(frame, alpha, beta)
        data_array = array('f', frame.reshape(-1,1))
        data_array.tofile(output_filehandle)

    return




def get_frame_bvi(filepath, frame_num, dim0, dim1):
    """
    * Description:
    * Return:
    * Arguments:
        -
    """
    filehandle = open(filepath, 'rb')
    filehandle.seek(BVI_HEADER_BYTES+4*dim0*dim1*frame_num)
    #print dim0*dim1
    data = filehandle.read(4*dim0*dim1)

    data = np.fromstring(data, dtype = 'float32').reshape((dim0, dim1))
    #print data.shape



    return data



"""
MP4 FUNCTIONS
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

def cine_to_mp4(cine_file_path, mp4_file_path):
    cin = cine.Cine(cine_file_path)
    writer = imageio.get_writer(mp4_file_path, fps = 30)
    print 'total_frames:', cin._total_frames
    for frame in xrange(cin._total_frames-1):
        try:
            print frame
            writer.append_data(cin.get_frame(frame, average = True))
        except:
            break

    writer.close()

    return
