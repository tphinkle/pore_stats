"""
pore_stats/cine.py

* Contains the Cine class
"""

import struct
import numpy as np


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
        mean = int(norm)*.5 + int(!norm)*255./2
        if average == True:
            frame = frame + (mean-np.mean(frame))

        frame[frame > 1] = 1
        frame[frame < 0] = 0

        return frame
