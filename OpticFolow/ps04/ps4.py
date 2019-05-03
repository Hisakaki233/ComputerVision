"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    return cv2.Sobel(image, -1, 1, 0, ksize=3, scale=1.0/8.0)


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    return cv2.Sobel(image, -1, 0, 1, ksize=3, scale=1.0/8.0)


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    tempImageA = np.copy(img_a)
    tempImageB = np.copy(img_b)

    # Calculate Ix, Iy, It for Lucas - Kanade method
    Ix = gradient_x(tempImageA)
    Iy = gradient_y(tempImageA)
    It = tempImageB - tempImageA

    threeDimensionMatrix = np.zeros(tempImageA.shape + (6,))

    # Calculate all necessary elements
    threeDimensionMatrix[..., 0] = Ix * Ix
    threeDimensionMatrix[..., 1] = Ix * Iy
    threeDimensionMatrix[..., 2] = Iy * Ix
    threeDimensionMatrix[..., 3] = Iy * Iy
    threeDimensionMatrix[..., 4] = Ix * It
    threeDimensionMatrix[..., 5] = Iy * It

    # Implement Uniform kernel
    if k_type == "uniform":

        # Define Kernel, matrix with all 1's
        # Calulate Window SUM for threeDimensionMatrix
        # Use Uniform convolution?

        uniformKernel = np.ones((k_size, k_size), np.float32)
        Ixx = cv2.filter2D(threeDimensionMatrix[..., 0], -1, uniformKernel)
        Ixy = cv2.filter2D(threeDimensionMatrix[..., 1], -1, uniformKernel)
        Iyx = cv2.filter2D(threeDimensionMatrix[..., 2], -1, uniformKernel)
        Iyy = cv2.filter2D(threeDimensionMatrix[..., 3], -1, uniformKernel)
        Ixt = cv2.filter2D(threeDimensionMatrix[..., 4], -1, uniformKernel)
        Iyt = cv2.filter2D(threeDimensionMatrix[..., 5], -1, uniformKernel)

        u = np.zeros(tempImageA.shape)
        v = np.zeros(tempImageA.shape)

        denominatorM = Ixx * Iyy - Ixy * Iyx
        numeratorU = Ixy * Iyt - Iyy * Ixt
        numeratorV = Iyx * Ixt - Ixx * Iyt

        with np.errstate(divide='ignore', invalid='ignore'):

            op_flow_x = np.true_divide(numeratorU, denominatorM)
            op_flow_x[~ np.isfinite(op_flow_x)] = 0

            op_flow_y = np.true_divide(numeratorV, denominatorM)
            op_flow_y[~ np.isfinite(op_flow_y)] = 0

        u[:, :] = op_flow_x[:, :]
        v[:, :] = op_flow_y[:, :]

        u[:, :] = op_flow_x[:, :]
        v[:, :] = op_flow_y[:, :]

    if k_type == "gaussian":

        # Define Kernel, matrix with all 1's
        # Calulate Window SUM for threeDimensionMatrix
        # Use Uniform convolution?

        Ixx = cv2.GaussianBlur(threeDimensionMatrix[..., 0], (k_size, k_size), sigma)
        Ixy = cv2.GaussianBlur(threeDimensionMatrix[..., 1], (k_size, k_size), sigma)
        Iyx = cv2.GaussianBlur(threeDimensionMatrix[..., 2], (k_size, k_size), sigma)
        Iyy = cv2.GaussianBlur(threeDimensionMatrix[..., 3], (k_size, k_size), sigma)
        Ixt = cv2.GaussianBlur(threeDimensionMatrix[..., 4], (k_size, k_size), sigma)
        Iyt = cv2.GaussianBlur(threeDimensionMatrix[..., 5], (k_size, k_size), sigma)

        u = np.zeros(tempImageA.shape)
        v = np.zeros(tempImageA.shape)

        denominatorM = Ixx * Iyy - Ixy * Iyx
        numeratorU = Ixy * Iyt - Iyy * Ixt
        numeratorV = Iyx * Ixt - Ixx * Iyt

        with np.errstate(divide='ignore', invalid='ignore'):

            op_flow_x = np.true_divide(numeratorU, denominatorM)
            op_flow_x[~ np.isfinite(op_flow_x)] = 0

            op_flow_y = np.true_divide(numeratorV, denominatorM)
            op_flow_y[~ np.isfinite(op_flow_y)] = 0

        u[:, :] = op_flow_x[:, :]
        v[:, :] = op_flow_y[:, :]

    return u, v


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    # 5-tap separable filter (1 4 6 4 1) / 16, with a = 3/8
    filter = np.array([1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16])
    kernel = np.outer(filter, filter)

    # Filtering using Kernel derived
    reducedImage = cv2.filter2D(image, -1, kernel)[::2, ::2]

    return reducedImage


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    tempImage = np.copy(image)
    returnImageList = list()

    for i in range(levels):

        if i == 0:
            returnImageList.append(tempImage)
            continue

        tempImage = reduce_image(tempImage)
        returnImageList.append(tempImage)

    return returnImageList


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    # Define New image Dimension
    width = list()
    height = list()

    for i in range(len(img_list)):
        elements = np.array(img_list[i])

        height.append(elements.shape[0])
        width.append(elements.shape[1])

    width.insert(0, 0)
    cumWidth = np.cumsum(np.asarray(width))

    # Concatenation
    returnImage = np.zeros((max(height), max(cumWidth)), np.float32)

    for i in range(len(img_list)):

        normalizedImage = normalize_and_scale(img_list[i], scale_range=(0, 255))
        returnImage[0: height[i], cumWidth[i]: cumWidth[i+1]] = normalizedImage

    return returnImage


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    combineKernal = np.array([1.0 / 8, 4.0 / 8, 6.0 / 8, 4.0 / 8, 1.0 / 8])
    combineKernal = np.outer(combineKernal, combineKernal)

    expandImage = np.zeros((2 * image.shape[0], 2 * image.shape[1]))
    expandImage[::2, ::2] = image

    return cv2.filter2D(expandImage, -1, combineKernal)


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    returnImageList = [0] * len(g_pyr)

    for i in range(len(g_pyr)):

        if i == len(g_pyr) - 1:
            returnImageList[i] = g_pyr[i]
        else:
            img_o = g_pyr[i]
            img_e = expand_image(g_pyr[i+1])[:img_o.shape[0], :img_o.shape[1]]

            returnImageList[i] = img_o - img_e

    return returnImageList


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    tempImage = np.copy(image)

    M, N = tempImage.shape
    X, Y = np.meshgrid(range(N), range(M))

    newX = X.astype('float32') + U.astype('float32')
    newY = Y.astype('float32') + V.astype('float32')

    return cv2.remap(tempImage, newX, newY, interpolation=interpolation, borderMode=border_mode)


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    # Create Gaussian pyramid list
    tempImageA = gaussian_pyramid(img_a, levels)
    tempImageB = gaussian_pyramid(img_b, levels)

    aggU = np.zeros(tempImageA[levels - 1].shape)
    aggV = np.zeros(tempImageA[levels - 1].shape)

    # Iterative Lucas Kanade Method
    for i in range(len(tempImageA)):

        # calculate optic flow
        inputImageA = tempImageA[levels - i - 1]
        inputImageB = tempImageB[levels - i - 1]

        if i == 0:
            u, v = optic_flow_lk(inputImageA, inputImageB, int(k_size / (i+1)), k_type, int(sigma / (i+1)))
            aggU = u
            aggV = v

        else:
            # warp image
            aggU = 2 * expand_image(aggU)[:inputImageA.shape[0], :inputImageA.shape[1]]
            aggV = 2 * expand_image(aggV)[:inputImageA.shape[0], :inputImageA.shape[1]]

            warpImageA = warp(inputImageA, -aggU, -aggV, interpolation, border_mode)

            u, v = optic_flow_lk(warpImageA, inputImageB, k_size, k_type, sigma)

            aggU = u + aggU
            aggV = v + aggV

    return aggU, aggV


def frame_interpolation(img_a, img_b, img_a_blur, img_b_blur, levels, k_size, k_type, sigma,
                        interpolation, border_mode):

    u, v = hierarchical_lk(img_a_blur, img_b_blur, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    img_t02 = warp(img_a, -0.2 * u, -0.2 * v, interpolation, border_mode)
    img_t04 = warp(img_a, -0.4 * u, -0.4 * v, interpolation, border_mode)
    img_t06 = warp(img_a, -0.6 * u, -0.6 * v, interpolation, border_mode)
    img_t08 = warp(img_a, -0.8 * u, -0.8 * v, interpolation, border_mode)
    img_t10 = warp(img_a, -1.0 * u, -1.0 * v, interpolation, border_mode)

    row1 = np.hstack([img_a, img_t02, img_t04])
    row2 = np.hstack([img_t06, img_t08, img_b])

    returnImage = np.vstack([row1, row2])

    return returnImage


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None