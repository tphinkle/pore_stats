def PreprocessFrame(frame, template_frame, preprocessing_steps = {}):
    processed_frame = CopyFrame(frame)
    processed_template_frame = CopyFrame(template_frame)


    for preprocessing_step in preprocessing_steps:
        if preprocessing_step[0] == 'crop':
            parameters = preprocessing_step[1]
            x = parameters
            processed_frame = CropFrame(frame, )



def CopyFrame(frame):
    return np.copy(frame)


def CropFrame(frame, x, y, crop_distance):
    x0 = int(x - crop_distance)
    x1 = int(x + crop_distance)
    y0 = int(y - crop_distance)
    y1 = int(y + crop_distance)

    return frame[y0:y1, x0:x1]
