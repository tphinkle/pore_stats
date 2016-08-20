"""

IMAGING

* Contains tools for opening, analyzing imaging data from experiments

* Sections:
    1. Imports
    2. Constants
    3. Classes
    4. Functions
"""


"""
Imports
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt
from array import array
import optical_imaging as oi
import struct
import csv
import sys
import copy

"""
Constants
"""
FFMPEG_BIN_FILENAME = '/home/preston/ffmpeg-3.0.2-64bit-static/ffmpeg'

BVI_HEADER_BYTES = 6

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
    f = open(output_filepath, 'w')

    f.write(output_filepath+'\n')
    for i, event in enumerate(oi_events):
        f.write('event#\t'+str(i)+'\n')
        for j, detection in enumerate(event._detections):
            f.write(str(int(detection._tf)) + '\t' + str(int(detection._px)) + '\t' + str(int(detection._py)) + '\n')

    f.close()

    return

def load_oi_events(input_filepath):
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
    with open(filepath, 'rb') as f:
        header_bytes = f.read(12)


    file_length = struct.unpack('I', header_bytes[8:12])[0]
    print 'file_length = ', file_length

    return file_length

def get_dimensions_bvi(filepath):
    with open(filepath, 'rb') as f:
        header_bytes = f.read(8)

    dim0 = struct.unpack('I', header_bytes[0:4])[0]
    dim1 = struct.unpack('I', header_bytes[4:8])[0]

    print 'dim0 = ', dim0, 'dim1 = ', dim1


    return dim0, dim1

def save_oi_file(input_filepath, output_filepath, alpha = 1, beta = 0):
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




def get_frame_bvi(filehandle, frame_num, dim0, dim1):
    """
    * Description:
    * Return:
    * Arguments:
        -
    """
    filehandle.seek(BVI_HEADER_BYTES+2*dim0*dim1*frame_num)
    data = filehandle.read(2*dim0*dim1)
    data = np.fromstring(data).reshape()
    print data.shape


    return data

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




def change_frame_contrast(frame, alpha, beta):
    frame = frame*alpha + beta
    return frame
