"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def canny_edge(img, minVal, maxVal):

    edges = cv2.Canny(img, minVal, maxVal)

    return edges


def detect_circles(img, threshold, radiusList, param1=15, param2=10):

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, threshold, param1=param1, param2=param2, minRadius=min(radiusList), maxRadius=max(radiusList))

    # return 3-D tuples for Circles found
    circlesList = []
    if circles is not None:
        circles = circles.astype(int)
        for item in circles[0]:
            x = item[0]
            y = item[1]
            r = item[2]

            circlesList.append((x, y, r))

    # sort output by Y axis
    circlesList.sort(key=lambda var: var[0])

    return circlesList


def detect_line(img, threshold, degree=180, minLineLength=20, maxLineGap=10):

    lines = cv2.HoughLinesP(img, 1, np.pi / degree, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # image shape
    height, width = img.shape

    lineList = []
    if lines is not None:
        for [[x1, y1, x2, y2]]in lines:
            # Drop parallel line ~ Pic Length
            if (x2 - x1) > width * 0.35:
                continue

            dist = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            lineList.append((x1, y1, x2, y2, dist))

    lineList.sort(key=lambda var: var[0])
    return lineList


def detect_yield_sign(coordinates, tolerance=5):
    # A yield sign basically need a set of 3 lines with similar length and theta approximately pi/3
    if len(coordinates) >= 3:

        for line1 in coordinates:
            for line2 in coordinates:
                if line1 != line2:
                    # Find two line nearby and with theta around pi/3
                    if np.sqrt((line1[0] - line2[0]) ** 2 + (line1[1] - line2[1]) ** 2) <= tolerance:
                        if line1[2] != line2[2]:
                            slope = (line1[3] - line2[3]) / (line1[2] - line2[2])
                            for line3 in coordinates:
                                if line3 != line1 and line3 != line2 and line3[0] != line3[2]:
                                    if abs((line3[3] - line3[1])/(line3[2] - line3[0]) - slope) <= 0.1:
                                        if line1[2] > line2[2]:
                                            x = line1[0] + 2 * line1[2] + line2[0] + 2 * line2[2]
                                            y = line1[1] + 2 * line1[3] + line2[1] + 2 * line2[3]
                                            centroid = (int(x/6), int(y/6))
                                            return centroid
                                        else:
                                            continue

                    elif np.sqrt((line1[2] - line2[2]) ** 2 + (line1[3] - line2[3]) ** 2) <= tolerance:
                        if line1[0] != line2[0]:
                            slope = (line1[1] - line2[1]) / (line1[0] - line2[0])
                            for line3 in coordinates:
                                if line3 != line1 and line3 != line2 and line3[0] != line3[2]:
                                    if abs((line3[3] - line3[1])/(line3[2] - line3[0]) - slope) <= 0.1:
                                        if line1[0] < line2[0]:
                                            x = 2 * line1[0] + line1[2] + 2 * line2[0] + line2[2]
                                            y = 2 * line1[1] + line1[3] + 2 * line2[1] + line2[3]
                                            centroid = (int(x/6), int(y/6))
                                            return centroid
                                        else:
                                            continue


def detect_stop_sign(coordinates):

    # A strop sign basically need a set of 8 lines with similar length
    # Screen out Lines parallel to x-axis or y-axis
    # detect four different direction lines
    # slope with 1/-1 & vertical & horizontal line

    edges = []
    for lines in coordinates:
        if lines[1] == lines[3]:
            edges.append((lines[0], lines[1], lines[2], lines[3], lines[4], "H"))
        elif lines[0] == lines[2]:
            edges.append((lines[0], lines[1], lines[2], lines[3], lines[4], "V"))
        elif abs((lines[1] - lines[3]) / (lines[0] - lines[2]) - 1) <= 0.15:
            edges.append((lines[0], lines[1], lines[2], lines[3], lines[4], "+1"))
        elif abs((lines[1] - lines[3]) / (lines[0] - lines[2]) + 1) <= 0.15:
            edges.append((lines[0], lines[1], lines[2], lines[3], lines[4], "-1"))

    edges.sort(key=lambda var: (var[0], var[5]))
    # Choose Four edges
    for i in range(len(edges) - 3):
        selectedEdges = edges[i:i+6]
        edgeSets = set()
        for j in selectedEdges:
            edgeSets.add(j[5])

        if edgeSets == {'-1', 'V', '+1', 'H'}:
            centroidX = 0
            centroidY = 0
            for lines in selectedEdges:

                if lines[5] == "H":
                    centroidX = int((lines[0] + lines[2]) / 2)
                if lines[5] == "V":
                    centroidY = int((lines[1] + lines[3]) / 2)

            for lines in selectedEdges:

                if np.sqrt((lines[0] - centroidX) ** 2 + (lines[1] - centroidY) ** 2) >= lines[4] * 1.5:
                    centroidX = -1
                    centroidY = -1
                    break

            if centroidX == -1 and centroidY == -1:
                continue
            else:
                return centroidX, centroidY

        else:
            continue


def detect_warning_construction_sign(img_in, coordinates, signToDetect = "warning"):

    # Screen out Lines parallel to x-axis or y-axis
    coordinates = [lines for lines in coordinates if (lines[0] != lines[2]) and (lines[1] != lines[3])]

    # The signs are basically the same, need another function to detect the color to tell the difference

    for line1 in coordinates:
        linesList = [line1]
        for line2 in coordinates:
            if abs(line2[4]-line1[4]) <= 4:
                linesList.append(line2)

        edges = []
        for lines in linesList:
            if (lines[0] < lines[2]) and (lines[1] > lines[3]):
                edges.append((lines[0], lines[1], lines[2], lines[3], lines[4], "U"))
            elif (lines[0] < lines[2]) and (lines[1] < lines[3]):
                edges.append((lines[0], lines[1], lines[2], lines[3], lines[4], "D"))

        edges.sort(key=lambda var: (var[0], var[5]))

        # drop similar line
        edges = list(set(edges))
        for lineA in edges:
            for lineB in edges:
                if edges.index(lineA) == edges.index(lineB):
                    continue
                elif abs(lineB[0]-lineA[0]) <= 8 and abs(lineB[1]-lineA[1]) <= 8 and (lineB[5]==lineA[5]):
                    edges.pop(edges.index(lineB))

        # Detect colors - based on HSV conversion
        if len(edges) >= 4:

            edges.sort(key=lambda var: (var[0]))

            for i in range(len(edges)-3):

                linesPosition = [(lines[0], lines[1], lines[2], lines[3]) for lines in edges[i:i+4]]

                sumList = np.sum(linesPosition, axis=0)
                centroid = (int((sumList[0] + sumList[2]) / 8),
                            int((sumList[1] + sumList[3]) / 8))

                tempImage = np.copy(img_in)
                cropImage = tempImage[centroid[1] - 5:centroid[1] + 5, centroid[0] - 5: centroid[0] + 5]

                hsv = cv2.cvtColor(cropImage, cv2.COLOR_BGR2HSV)
                avgHSV = np.mean(hsv, axis=1)

                hsvValue = np.array([i[0] for i in avgHSV]).astype(int)

                if signToDetect == "construction" and np.all(hsvValue == 15):
                    return centroid
                elif signToDetect == "warning" and np.all(hsvValue == 30):
                    return centroid
                else:
                    continue
        else:
            continue


def detect_do_not_enter_sign(roundCoordinates, lineCoordinates):

    # Do not enter sign is a combination of Round detection and line detection.
    # Only pulling in relative long lines, should be parallel and contained in the circle

    lineScreen = []
    for rows in lineCoordinates:
        if abs(rows[1] - rows[3]) <= 5:
            lineScreen.append(rows)

    if len(lineScreen) < 2:
        centroid = (0, 0)

    else:
        for rounds in roundCoordinates:
            roundProperty = [rounds]
            linesProperty = []
            for lines in lineCoordinates:
                if np.sqrt((lines[0] - rounds[0]) ** 2 + (lines[1] - rounds[1]) ** 2) <= rounds[2] + 5:
                    linesProperty.append(lines)

            if len(roundProperty) == 1 and len(linesProperty) >= 2:
                centroid = (roundProperty[0][0], roundProperty[0][1])
                return centroid
            else:
                continue


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """

    # Convert Input image to Edge image using Canny Edge detection
    # tempImage = np.copy(img_in)
    tempImage = np.copy(img_in)

    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
    edgeImage = canny_edge(gray, 25, 75)

    # Adjust Input Radius List by convert [a,b) to [a,b]
    radiusList = list(radii_range)
    radiusList.append(max(radiusList) + 1)

    # Find Circles using Hough Transform
    coordinates = detect_circles(edgeImage, 15, radiusList)

    # Cleanup the coordinates - remove outlier x
    while(len(coordinates) > 3):
        if coordinates[0][0] < coordinates[1][0]-3:
            coordinates = coordinates[1:]
            continue

        if coordinates[len(coordinates)-1][0] > coordinates[1][0]+3:
            coordinates = coordinates[:len(coordinates)-1]
            continue

    coordinates.sort(key=lambda var: var[1])
    if abs(coordinates[0][0] - coordinates[1][0]) <= 8:
        centroid = (coordinates[1][0], coordinates[1][1])

        # Find out the state of lights
        # Using HSV Value to find out brightest circle
        hsv = cv2.cvtColor(tempImage, cv2.COLOR_BGR2HSV)

        HSValue = 0
        status = ''
        for k in range(len(coordinates)):
            x = coordinates[k][0]
            y = coordinates[k][1]

            if hsv[y][x][2] > HSValue:
                HSValue = hsv[y][x][2]

                if k == 0:
                    status = 'red'
                elif k == 1:
                    status = 'yellow'
                elif k == 2:
                    status = 'green'

        # output the required result
        detect_light = (centroid, status)
        return detect_light

    elif abs(coordinates[1][0] - coordinates[2][0]) <= 8:
        centroid = (coordinates[1][0], coordinates[1][1])

        # Find out the state of lights
        # Using HSV Value to find out brightest circle
        hsv = cv2.cvtColor(tempImage, cv2.COLOR_BGR2HSV)

        HSValue = 0
        status = ''
        for k in range(len(coordinates)):
            x = coordinates[k][0]
            y = coordinates[k][1]

            if hsv[y][x][2] > HSValue:
                HSValue = hsv[y][x][2]

                if k == 0:
                    status = 'red'
                elif k == 1:
                    status = 'yellow'
                elif k == 2:
                    status = 'green'

        # output the required result
        detect_light = (centroid, status)
        return detect_light


def yield_sign_detection(img_in, threshold=25, maxLineGap=5, tolerance=5):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    # tempImage = reduceBackgroundColor(np.copy(img_in))
    tempImage = np.copy(img_in)
    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
    edgeImage = canny_edge(gray, 25, 50)

    coordinates = detect_line(edgeImage, threshold, degree=180, minLineLength=35, maxLineGap=maxLineGap)
    coordinates = [lines for lines in coordinates if lines[4] > 70 and lines[4] < 110]

    # A yield sign basically need a set of 3 lines with similar length and theta approximately pi/3
    centroid = detect_yield_sign(coordinates, tolerance=tolerance)
    return centroid


def stop_sign_detection(img_in, threshold=10, maxLineGap=7):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    # tempImage = reduceBackgroundColor(np.copy(img_in))
    tempImage = np.copy(img_in)
    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
    edgeImage = canny_edge(gray, 25, 50)

    coordinates = detect_line(edgeImage, threshold, degree=180, minLineLength=15, maxLineGap=maxLineGap)
    coordinates = [lines for lines in coordinates if lines[4] > 25 and lines[4] <= 50]

    # A yield sign basically need a set of 3 lines with similar length and theta approximately 135 degree - 3/4 pi
    centroid = detect_stop_sign(coordinates)

    return centroid


def warning_sign_detection(img_in, threshold=45, minLineLength=30, maxLineGap=5):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    tempImage = np.copy(img_in)
    # reducedImage = reduceBackgroundColor(tempImage)
    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
    edgeImage = canny_edge(gray, 25, 75)

    coordinates = detect_line(edgeImage, threshold, degree=180, minLineLength=minLineLength, maxLineGap=maxLineGap)
    centroid = detect_warning_construction_sign(tempImage, coordinates, signToDetect="warning")

    return centroid


def construction_sign_detection(img_in, threshold=45, minLineLength=30, maxLineGap=5):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    tempImage = np.copy(img_in)
    # reducedImage = reduceBackgroundColor(tempImage)
    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
    edgeImage = canny_edge(gray, 25, 75)

    coordinates = detect_line(edgeImage, threshold, degree=180, minLineLength=minLineLength, maxLineGap=maxLineGap)
    centroid = detect_warning_construction_sign(tempImage, coordinates, signToDetect="construction")

    return centroid


def do_not_enter_sign_detection(img_in, minRadius=25, maxRadius=40):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """

    # Convert Input image to Edge image using Canny Edge detection
    # tempImage = reduceBackgroundColor(np.copy(img_in))
    tempImage = np.copy(img_in)
    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
    edgeImage = canny_edge(gray, 25, 50)

    # Adjust Input Radius List by convert [a,b) to [a,b]
    radiusList = list(range(minRadius, maxRadius, 1))
    radiusList.append(max(radiusList) + 1)

    # Find Circles using Hough Transform
    roundCoordinates = detect_circles(edgeImage, 25, radiusList, param1=25, param2=15)
    for i in roundCoordinates:
        # draw the outer circle
        cv2.circle(tempImage, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Find Lines using Hough Transform
    # Put a relatively high threshold to identify only two long lines in the circle
    lineCoordinates = detect_line(edgeImage, 20, degree=180, minLineLength=25, maxLineGap=10)

    centroid = detect_do_not_enter_sign(roundCoordinates, lineCoordinates)

    return centroid


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    tempImage = np.copy(img_in)
    sign_fns = [traffic_light_detection,
                stop_sign_detection,
                construction_sign_detection,
                do_not_enter_sign_detection,
                warning_sign_detection,
                yield_sign_detection]

    sign_labels = ['traffic', 'stop', 'construction', 'no_entry', 'warning', 'yield']
    radii_range = range(5, 15, 1)

    resultDict = {}

    for fn, name in zip(sign_fns, sign_labels):

        if name == 'traffic':
            coords = fn(tempImage, radii_range)
        else:
            coords = fn(tempImage)

        if coords is not None:
            resultDict[name] = coords
        print(resultDict)

    return resultDict


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    tempImage = np.copy(img_in)

    sign_fns = [traffic_light_detection,
                stop_sign_detection,
                construction_sign_detection,
                warning_sign_detection,
                yield_sign_detection,
                do_not_enter_sign_detection]

    sign_labels = ['traffic', 'stop', 'construction', 'warning', 'yield', 'no_entry']

    radii_range = range(10, 15, 1)
    resultDict = {}

    for fn, name in zip(sign_fns, sign_labels):

        if name == 'traffic':
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (1, 1), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 11, 7, 21)
            coords = fn(denoised, radii_range)
        elif name == 'stop':
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (5, 5), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
            coords = fn(denoised, threshold=10, maxLineGap=25)
        elif name == 'yield':
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (3, 3), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 4, 3, 7, 21)
            coords = fn(denoised, threshold=30, maxLineGap=10, tolerance=15)
        else:
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (3, 3), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
            coords = fn(denoised)

        if coords is not None:
            resultDict[name] = coords
        print(resultDict)

    return resultDict


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    tempImage = np.copy(img_in)

    sign_fns = [stop_sign_detection,
                yield_sign_detection,
                do_not_enter_sign_detection,
                construction_sign_detection,
                warning_sign_detection
                ]

    sign_labels = ['stop', 'yield', 'no_entry', 'construction', 'warning']

    resultDict = {}

    for fn, name in zip(sign_fns, sign_labels):

        if name == 'stop':
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (5, 5), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 15, 15, 7, 21)
            coords = fn(denoised, threshold=10, maxLineGap=20)

        elif name == 'yield':
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (7, 7), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
            coords = fn(denoised, threshold=25, maxLineGap=15, tolerance=15)

        elif name == 'no_entry':
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (9, 9), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 15, 10, 7, 21)
            coords = fn(denoised, minRadius=80, maxRadius=100)

        elif name == 'construction' or name == 'warning':
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (5, 5), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
            coords = fn(denoised, threshold=10, maxLineGap=10)

        else:
            # Code to adjust to adapt to noise image
            # Detect traffic lights
            blurred = cv2.GaussianBlur(tempImage, (9, 9), 0)
            denoised = cv2.fastNlMeansDenoisingColored(blurred, None, 15, 10, 7, 21)
            coords = fn(denoised)

        if coords is not None:
            resultDict[name] = coords
        print(resultDict)

    return resultDict

