import cv2
import numpy as np
import sklearn.mixture
import optical_imaging as oi
import scipy.ndimage


def copy_frame(frame):
    return np.copy(frame)

def crop_frame(frame, x, y, crop_distance):
    '''
    Crop frame around points x,y so that only pixels local to the
    particle's position are considered.
    '''
    x0 = int(x - crop_distance)
    x1 = int(x + crop_distance)
    y0 = int(y - crop_distance)
    y1 = int(y + crop_distance)


    cropped_frame = np.copy(frame)[y0:y1, x0:x1]
    return cropped_frame

def normalize(frame):
    '''
    '''
    normalized_frame = (frame - np.min(frame))/(np.max(frame)-np.min(frame))

    return normalized_frame


def gaussian_blur(frame, blur_kernel):
    '''
    Blur frame by centering a 2D gaussian w/ std. dev. given by the blur kernel
    Sample values:
    (3,3); (15,15); etc.
    blur_kernel numbers must be odd integers
    '''

    blurred_frame = cv2.GaussianBlur(frame, blur_kernel, 0)

    return blurred_frame



def negative(frame, template_frame, direction):
    '''
    Subtract the frame off of the template.
    The direction of the subtraction is determined by direction, and can be
    either 'neg', 'pos', or 'abs'
    '''

    if direction == 'neg':
        negative_frame = template_frame - frame
    elif direction == 'pos':
        negative_frame = frame - template_frame
    elif direction == 'abs':
        negative_frame = np.abs(frame - template_frame)


    return negative_frame



def gradient(frame):
    '''
    Compute the Laplacian gradient of the image
    Useful for finding edges in imagse
    '''

    gradient_frame = cv2.Laplacian(frame, cv2.CV_64F)

    return gradient_frame


def invert(frame):
    '''
    Turn white pixels black and vice-versa
    '''

    inverted_frame = 1 - frame

    return inverted_frame



def twogaussian_threshold(frame, sigma_multiplier):
    '''
    Thresholds the frame so that all pixels above a cutoff are set to 1;
    all frames below that threshold are set to 0

    The cutoff level is determined by the following protocol:

    Two gaussians are fit to the histogram of the pixel intensities

    The idea is to separate a population of mostly black pixels and a separate
    population of mostly white pixels; the left Gaussian represents the dark pixels,
    the right Gaussian represents the light pixels

    The cutoff is set in between the two gaussians, at a distance of
    sigma_multiplier * std. dev. to the left of the bright gaussian
    '''

    model = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
    model.fit(frame.reshape(-1,1))




    weights = model.weights_.flatten()
    means = model.means_.flatten()
    covariances = model.covariances_.flatten()



    index = 1
    if means[0] > means[1]:
        index = 0

    threshold = means[index] - sigma_multiplier*covariances[index]**.5

    thresholded_frame[frame < threshold] = 0
    thresholded_frame[frame > threshold] = 1

    return thresholded_frame

def gaussian_threshold(frame, sigma_multiplier):

    '''
    Threshold the pixels so that all pixels above a cutoff are set to 1, all pixels below
    that threshold are set to zero.

    Fits a single Gaussian to the data, and sets the cutoff to be sigma_multiplier Standard
    deviations of hte Gaussian to the right of the Gaussian's center
    '''



    model = sklearn.mixture.GaussianMixture(n_components=1, covariance_type='full')
    model.fit(frame.reshape(-1,1))



    weight = model.weights_.flatten()[0]
    mean = model.means_.flatten()[0]
    covariance = model.covariances_.flatten()[0]



    threshold = mean + sigma_multiplier*(covariance**.5)


    thresholded_frame = np.copy(frame)
    thresholded_frame[frame < threshold] = 0
    thresholded_frame[frame >= threshold] = 1


    return thresholded_frame

def largest_cluster(frame):
    '''
    Form clusters of white pixels that are touching, and keep only the pixels that belong
    to the largest cluster
    '''



    clusters = oi.find_clusters_iterative_percentage_based(frame, np.zeros((frame.shape[0], frame.shape[1])), diag = True)
    largest_cluster = sorted(clusters, key = lambda x: len(x))[-1]
    cluster_frame = np.zeros(frame.shape, dtype = np.uint8)
    for pixel in largest_cluster:
        cluster_frame[pixel[0], pixel[1]] = 1

    return cluster_frame





def threshold_clusters(frame, cluster_threshold):
    '''
    Form clusters of white pixels that are touching, and only retain the clusters
    that exceed cluster_threshold number of pixels in size
    '''




    clusters = oi.find_clusters_iterative_percentage_based(frame, np.zeros((frame.shape[0],\
                                                                            frame.shape[1])),\
                                                 diag = True, cluster_threshold = cluster_threshold)

    clusters = [cluster for cluster in clusters if len(cluster) > cluster_threshold]

    cluster_frame = np.zeros(frame.shape, dtype = np.uint8)
    for cluster in clusters:
        for pixel in cluster:
            cluster_frame[pixel[0], pixel[1]] = 1


    return cluster_frame



def morphological_closing(frame, morph_kernel):
    '''
    Close holes in the image; for instance, if there is a cavity in the detected
    ellipse it will fill the cavity
    '''


    # Pad
    pad_width = 20
    morphed_frame = np.lib.pad(frame, pad_width, 'constant')

    # Close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    morphed_frame = cv2.morphologyEx(np.array(morphed_frame, dtype = np.uint8), cv2.MORPH_CLOSE, kernel)

    # Unpad
    morphed_frame = np.copy(morphed_frame[pad_width:-pad_width, pad_width:-pad_width])


    return morphed_frame




def erodesubtract(frame, iterations):
    '''
    Erode the image, and subtract the eroded image from the original image

    This step transforms a solid cluster of pixels into a shell of pixels around the solid

    Increasing hte number of itereations is useful if the highlighted pixel area
    exceeds the bounds of the actual cell itself

    '''


    if iterations > 1:
        erodesubtracted_frame = frame - scipy.ndimage.morphology.binary_erosion(frame, iterations = iterations - 1)

    erodesubtracted_frame = erodesubtracted_frame - scipy.ndimage.morphology.binary_erosion(erodesubtracted_frame, iterations = 1)

    return erodesubtracted_frame

def dilatesubtract(frame, iterations):
    '''
    Similar to erode subtract, but tends to make the shell slightly larger
    '''

    if iterations > 1:
        dilatesubtracted_frame = frame - scipy.ndimage.morphology.binary_erosion(frame, iterations = iterations - 1)

    dilatesubtracted_frame = dilatesubtracted_frame - scipy.ndimage.morphology.binary_dilation(erodesubtracted_frame, iterations = 1)

    return dilatesubtracted_frame




'''






###################################
# Fit ellipse
###################################


cell_pixels = np.where(processed_frame == 1)




ellipse = oi.fit_ellipse_image_aligned(cell_pixels[1], cell_pixels[0])


# Center
ellipse_center = oi.get_ellipse_center(ellipse)
ellipse_center_adjusted = [ellipse_center[0] + detection._px - crop_distance, ellipse_center[1] + detection._py - crop_distance]

# Axes
ellipse_axes_lengths = oi.get_ellipse_axes_lengths(ellipse)

# Angle
ellipse_angle = oi.get_ellipse_angle(ellipse)



if debug != 'none':
    # Create perimeter line
    ellipse_points = np.empty((100,2))
    for i in range(100):
        angle = i*2*np.pi/99.
        x = ellipse_axes_lengths[0]*np.cos(angle)
        y = ellipse_axes_lengths[1]*np.sin(angle)
        ellipse_points[i,0] = ellipse_center[0] + np.cos(ellipse_angle)*x + np.sin(ellipse_angle)*y
        ellipse_points[i,1] = ellipse_center[1] + np.sin(ellipse_angle)*x - np.cos(ellipse_angle)*y

    # Turn pixels green
    green_processed_frame = np.zeros((processed_frame.shape[0], processed_frame.shape[1], 3))
    green_processed_frame[:,:,1] = processed_frame

    # Begin plot
    fig, axes = plt.subplots(1,3,figsize = (9,3))


    # Axes 0
    plt.sca(axes[0])

    plt.imshow(frame, cmap = 'gray', origin = 'lower', interpolation = 'none')
    plt.imshow(np.zeros(frame.shape), alpha = 0.5, cmap = 'gray')
    plt.xlim(0, processed_frame.shape[1])
    plt.ylim(0, processed_frame.shape[0])

    plt.xticks([])
    plt.yticks([])


    # Axes 1
    plt.sca(axes[1])

    plt.imshow(frame, cmap = 'gray', origin = 'lower', interpolation = 'none')
    plt.imshow(green_processed_frame, alpha = .5, origin = 'lower', interpolation = 'none')
    #plt.plot(ellipse_points[:,0], ellipse_points[:,1], lw = 3, c = 'red')

    #plt.scatter(ellipse_center[0], ellipse_center[1], marker = 'x', lw = 5, color = 'red', s = 50)

    plt.xlim(0, processed_frame.shape[1])
    plt.ylim(0, processed_frame.shape[0])

    plt.xticks([])
    plt.yticks([])

    # Axes 2
    plt.sca(axes[2])

    plt.imshow(green_processed_frame, alpha = .5, origin = 'lower', interpolation = 'none')
    #plt.imshow(processed_frame, cmap = 'gray', origin = 'lower', interpolation = 'none')
    #plt.imshow(green_processed_frame, alpha = .35, origin = 'lower', interpolation = 'none', zorder = 10)
    plt.plot(ellipse_points[:,0], ellipse_points[:,1], lw = 1, ls = '--', c = 'white')

    #plt.scatter(ellipse_center[0], ellipse_center[1], marker = 'x', c = 'white', lw = 3, s = 20)


    ellipse_axis_a = [ellipse_axes_lengths[0]*np.cos(ellipse_angle), ellipse_axes_lengths[0]*np.sin(ellipse_angle)]
    ellipse_axis_b = [ellipse_axes_lengths[1]*np.sin(ellipse_angle), -ellipse_axes_lengths[1]*np.cos(ellipse_angle)]


    ax0 = ellipse_center[0]
    ax1 = ax0 + ellipse_axis_a[0]
    ay0 = ellipse_center[1]
    ay1 = ay0 + ellipse_axis_a[1]

    bx0 = ellipse_center[0]
    bx1 = bx0 + ellipse_axis_b[0]
    by0 = ellipse_center[1]
    by1 = by0 + ellipse_axis_b[1]

    plt.plot([ax0, ax1], [ay0, ay1], lw = 1, ls = '--', c = 'white')
    plt.plot([bx0, bx1], [by0, by1], lw = 1, ls = '--', c = 'white')

    #plt.text((ax0+ax1)/2., (ay0+ay1)/2., 'a', color = 'white', size = 20, ha = 'left', va = 'bottom', fontweight = 'bold')


    #plt.text((bx0+bx1)/2., (by0+by1)/2., 'b', color = 'white', size = 20, ha = 'left', va = 'bottom', fontweight = 'bold')
    plt.text(ax1 + 4, ay1, 'a', color = 'white', size = 32, ha = 'left', va = 'center', fontweight = 'bold')
    plt.text(bx1, by1 - 4, 'b', color = 'white', size = 32, ha = 'center', va = 'top', fontweight = 'bold')



    a_um = oi_stage.pixels_to_meters(ellipse_axes_lengths[0])
    b_um = oi_stage.pixels_to_meters(ellipse_axes_lengths[1])

    #plt.text(0, 0.9, r'a='+str(round(a_um,2)) + r'$\mu$m', transform=plt.gca().transAxes, ha = 'left', va = 'bottom', size = 16, color = 'white', fontweight = 'bold')
    #plt.text(0, 0.8, r'b='+str(round(b_um,2)) + r'$\mu$m', transform=plt.gca().transAxes, ha = 'left', va = 'bottom', size = 16, color = 'white', fontweight = 'bold')
    plt.text(0, 1.0, r'a/b=' + str(round(a_um/b_um, 2)), transform = plt.gca().transAxes, ha = 'left', va = 'top', size = 18, color = 'white', fontweight = 'bold')
    plt.text(0, 0.9, r'$\theta=$'+str(round(ellipse_angle*180./np.pi,3)), transform=plt.gca().transAxes, ha = 'left', va = 'top', size = 18, color = 'white', fontweight = 'bold')



    plt.xlim(0, processed_frame.shape[1])
    plt.ylim(0, processed_frame.shape[0])

    plt.xticks([])
    plt.yticks([])


    fig.tight_layout()

    plt.show()
'''
