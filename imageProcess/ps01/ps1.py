import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)

    return temp_image[:, :, 2]


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """

    temp_image = np.copy(image)

    return temp_image[:, :, 1]


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """

    temp_image = np.copy(image)

    return temp_image[:, :, 0]


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image = np.copy(image)

    blue = extract_blue(temp_image)
    green = extract_green(temp_image)

    temp_image[:, :, 0] = green
    temp_image[:, :, 1] = blue

    return temp_image



def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """

    tempSrcImage = np.copy(src)
    tempDstImage = np.copy(dst)

    srcHeight, srcWidth = src.shape
    srcMidWidth = int(srcWidth / 2)
    srcMidHeight = int(srcHeight / 2)

    dstHeight, dstWidth = dst.shape
    dstMidWidth = int(dstWidth / 2)
    dstMidHeight = int(dstHeight / 2)

    cutHeight, cutWidth = shape
    cutHalfWidth = int(cutWidth/2)
    cutHalfHeight = int(cutHeight / 2)

    tempDstImage[dstMidHeight-cutHalfHeight: dstMidHeight+cutHalfHeight, dstMidWidth-cutHalfWidth: dstMidWidth+cutHalfWidth] = \
        tempSrcImage[srcMidHeight-cutHalfHeight: srcMidHeight+cutHalfHeight, srcMidWidth-cutHalfWidth: srcMidWidth+cutHalfWidth]

    return tempDstImage


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(image).astype(np.float)

    imageMin = np.min(temp_image)
    imageMax = np.max(temp_image)
    imageMean = np.mean(temp_image)
    imageStdDev = np.std(temp_image)

    return (imageMin, imageMax, imageMean, imageStdDev)


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    temp_image = np.copy(image)


    (imageMin, imageMax, imageMean, imageStdDev) = image_stats(temp_image)

    normalizeImage = scale * (temp_image - imageMean) / imageStdDev + imageMean

    return normalizeImage



def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image).astype(np.float)

    shiftImage = temp_image[:, shift:]

    return cv2.copyMakeBorder(shiftImage, 0, 0, 0, shift, cv2.BORDER_REPLICATE)


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    tempImg1 = np.copy(img1).astype(float)
    tempImg2 = np.copy(img2).astype(float)

    newImage = tempImg1 - tempImg2
    normalizeNewImage = cv2.normalize(newImage, newImage, 0, 255, cv2.NORM_MINMAX)

    return normalizeNewImage


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    temp_image = np.copy(image).astype(float)
    extractChannel = temp_image[:, :, channel]

    noise = np.random.normal(0, sigma, extractChannel.shape)

    extractChannel += noise
    temp_image[:, :, channel] = extractChannel

    return temp_image

