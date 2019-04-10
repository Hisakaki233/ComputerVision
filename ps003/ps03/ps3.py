"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    arrayP0 = np.asarray(p0)
    arrayP1 = np.asarray(p1)

    return np.linalg.norm(arrayP0 - arrayP1, axis=1)


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    tempImage = np.copy(image)

    # Image Shape
    rows = tempImage.shape[0]
    columns = tempImage.shape[1]

    # Four Marker Corresponding Location
    # (X,Y)
    topLeft = (0, 0)
    bottomLeft = (0, rows - 1)

    topRight = (columns - 1, 0)
    bottomRight = (columns - 1, rows - 1)

    return [topLeft, bottomLeft, topRight, bottomRight]


def rotate_image(image, theta=-90):

    tempImage = np.copy(image)
    rows, cols, _ = tempImage.shape

    # rotation Matrix
    M = cv2.getRotationMatrix2D((int(cols/2), int(rows/2)), theta, 1)
    rotateImage = np.copy(tempImage)

    for i in list(range(rows)):
        for j in list(range(cols)):
            u = int(M[0][0] * i + M[0][1] * j + M[0][2])
            v = int(M[1][0] * i + M[1][1] * j + M[1][2])

            if 0 < u < cols and 0 < v < rows:
                rotateImage[i, j] = tempImage[u, v]

    return rotateImage


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    # Initialize Lists
    tempImage = np.copy(image)
    rows, columns, _ = tempImage.shape

    # Implement HarrisCorner Detection
    # tempImage = cv2.GaussianBlur(tempImage, (5, 5), 0)
    # blurred = cv2.medianBlur(tempImage, 5)
    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)

    # res = cv2.cornerHarris(gray, 9, 9, 0.001)
    res = cv2.cornerHarris(gray, 11, 11, 0.0005)
    res = cv2.normalize(res,                # src
                        None,               # dst
                        0,                  # alpha
                        255,                # beta
                        cv2.NORM_MINMAX,    # norm type
                        cv2.CV_32F,         # dtype
                        None                # mask
                        )

    pts = np.where(res >= 0.15 * res.max())
    pts = list(zip(pts[1], pts[0]))
    pts = np.asarray(pts)
    pts = np.float32(pts)

    # Points Grouping and filtering - Using K-Mean Cluster with 4 groups
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    ret, label, center = cv2.kmeans(pts, 4, None, criteria, 1000, cv2.KMEANS_RANDOM_CENTERS)

    # Remove outlier
    thres = 15;
    pts2 = list()

    for i in pts:
        if (abs(i[0] - center[0][0]) + abs(i[1] - center[0][1]) < thres) or \
           (abs(i[0] - center[1][0]) + abs(i[1] - center[1][1]) < thres) or \
           (abs(i[0] - center[2][0]) + abs(i[1] - center[2][1]) < thres) or \
           (abs(i[0] - center[3][0]) + abs(i[1] - center[3][1]) < thres):

            pts2.append(i)

    pts2 = np.asarray(pts2)

    # Points Grouping and filtering - Using K-Mean Cluster with 4 groups
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1.0)
    ret, label, center = cv2.kmeans(pts2, 4, None, criteria, 500, cv2.KMEANS_RANDOM_CENTERS)

    distance = [(x[0], x[1], x[0] ** 2 + x[1] ** 2, x[0] ** 2 + (x[1] - rows) ** 2,
                (x[0] - columns) ** 2 + x[1] ** 2, (x[0] - columns) ** 2 + (x[1] - rows) ** 2) for x in center]

    d1 = distance[:]
    d1.sort(key=lambda var: var[2])
    d2 = [pts for pts in d1 if pts == d1[1] or pts == d1[2] or pts == d1[3]]
    d2.sort(key=lambda var: var[3])
    d3 = [pts for pts in d2 if pts == d2[1] or pts == d2[2]]
    d3.sort(key=lambda var: var[4])

    returnList = [d1[0], d2[0], d3[0], d3[1]]
    returnList = [(int(pts[0]), int(pts[1])) for pts in returnList]

    return returnList


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """

    cv2.line(image, markers[0], markers[1], 255, thickness=thickness)
    cv2.line(image, markers[0], markers[2], 255, thickness=thickness)
    cv2.line(image, markers[1], markers[3], 255, thickness=thickness)
    cv2.line(image, markers[2], markers[3], 255, thickness=thickness)

    return image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    inverseH = np.linalg.inv(homography)
    tempImageA = imageA.copy()
    tempImageB = imageB.copy()

    # Image Dimension
    imgArows, imgAcolumns, _ = tempImageA.shape
    imgBrows, imgBcolumns, _ = tempImageB.shape

    # Looping all elements seems to be too slow...
    '''
        # Doing reversed Way, Searching all points on imageB in imageA
        for i in range(imgBcolumns):
            for j in range(imgBrows):
    
                # Homogeneous Coordinates
                p_prime = np.matrix((i, j, 1)).transpose()
                # Find p in image A based on p prime in Image B
                p = np.dot(inverseH, p_prime)
                p = p / p[-1]
                # Convert Back
                imageArows = int(p[1])
                imageAcolumns = int(p[0])
    
                if imageArows >= 0 and imageArows < imgArows and imageAcolumns >= 0 and imageAcolumns < imgAcolumns:
                    tempImageB[j, i] = tempImageA[imageArows, imageAcolumns]
                else:
                    continue
    
        return tempImageB
    '''

    # Buildup an overall matrix containing all elements instead of doing iteration using Loop
    idy, idx = np.indices((imgBrows, imgBcolumns), dtype=np.float32)

    # Homogeneous Coordinates
    MatrixPPrime = np.array([idx.ravel(), idy.ravel(), np.ones_like(idx).ravel()])

    # Find p in image A based on p prime in Image B
    MatrixP = np.dot(inverseH, MatrixPPrime)

    # Convert Back to coordinates
    MatrixP = MatrixP[:] / MatrixP[-1]
    MatrixP = MatrixP[0:2]

    # Reconstruct the matrix to pull x-y information
    U = MatrixP[0].reshape(imgBrows, imgBcolumns)
    V = MatrixP[1].reshape(imgBrows, imgBcolumns)

    # Map (X, Y) in imageB to (U, V) in imageA
    for i in range(imgBcolumns):
        for j in range(imgBrows):

            imageArows = int(V[j, i])
            imageAcolumns = int(U[j, i])

            if imageArows >= 0 and imageArows < imgArows and imageAcolumns >= 0 and imageAcolumns < imgAcolumns:
                tempImageB[j, i] = tempImageA[imageArows, imageAcolumns]
            else:
                continue
    
    return tempImageB


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    # each points generate two equation
    # solve PH = 0

    P = list()
    for pts in list(range(len(src_points))):
        srcX, srcY = src_points[pts]
        dstX, dstY = dst_points[pts]

        P.append((-srcX, -srcY, -1, 0, 0, 0, dstX*srcX, dstX*srcY, dstX))
        P.append((0, 0, 0, -srcX, -srcY, -1, dstY*srcX, dstY*srcY, dstY))

    P = np.asmatrix(P)

    U, D, V = np.linalg.svd(P)

    # homography matrix
    L = V[-1, :] / V[-1, -1]
    H = L.reshape(3, 3)
    H = np.asarray(H)

    return H


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
