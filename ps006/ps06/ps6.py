"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]

    X = []
    y = []

    for img in images_files:
        image = cv2.imread(os.path.join(folder, img))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize Image
        resizeGrayImage = cv2.resize(grayImage, tuple(size), interpolation=cv2.INTER_CUBIC)

        # Flatten the Image and add to List X
        X.append(resizeGrayImage.flatten())

        # Add Label into y at the same time
        strs = [s for s in img.split(".")[0] if s.isdigit()]
        y.append(int(''.join(strs)))

    return np.array(X), np.array(y)


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """

    trainObs = int(p * X.shape[0])
    randomOrder = np.random.permutation(X.shape[0])

    idxTrain, idxTest = randomOrder[:trainObs], randomOrder[trainObs:]
    Xtrain, Xtest = X[idxTrain], X[idxTest]
    ytrain, ytest = y[idxTrain], y[idxTest]

    return Xtrain, ytrain, Xtest, ytest


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    return np.mean(x, axis=0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """


    M = X.shape[0]
    normalizedX = np.subtract(X, get_mean_face(X))
    # sigma = (1/M) * (normalizedX.T.dot(normalizedX))
    sigma = (normalizedX.T.dot(normalizedX))
    eigenValues, eigenVectors = np.linalg.eigh(sigma)

    # reverse Array
    eigenValues = eigenValues[::-1]
    eigenVectors = eigenVectors.T[::-1]
    return eigenVectors[:k].T, eigenValues[:k]


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""

        for i in range(self.num_iterations):

            # 1. Renormalize the weights
            self.weights /= np.sum(self.weights)

            # 2. Instantiate the weak classifier h with the training data and labels.
            #    Train the classifier h
            #    Get predictions h(x)

            wk_clf = WeakClassifier(self.Xtrain, self.ytrain, self.weights)

            wk_clf.train()
            wk_results = [wk_clf.predict(x) for x in self.Xtrain]

            # 3.Find eps
            eps = sum([self.weights[i] if wk_results[i] != self.ytrain[i] else 0 for i in range(len(self.ytrain))])

            # 4.Calculate alpha
            alpha = 0.5 * np.log((1.0 - eps)/(eps + 1e-50))

            self.weakClassifiers.append(wk_clf)
            self.alphas.append(alpha)

            # 5.Update the weights
            if eps > self.eps:

                # Adjust effect
                for j in range(self.ytrain.shape[0]):
                    adj = np.exp(-1 * self.ytrain[j] * self.alphas[i-1] * self.weakClassifiers[i-1].predict(self.Xtrain[j]))
                    self.weights[j] = np.multiply(self.weights[j], adj)

            else:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        difference = np.subtract(self.predict(self.Xtrain), self.ytrain)

        return (difference == 0.0).sum(), (difference != 0.0).sum()

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        values = []
        for i in range(len(self.alphas)):
            predicates=[]
            for j in range(X.shape[0]):
                predicated = self.weakClassifiers[i].predict(X[j])
                predicates.append(predicated)

            predicates = np.asarray(predicates)
            values.append(self.alphas[i] * predicates)

        predictions = np.sign(np.sum(values, axis=0))
        return np.asarray(predictions.copy())


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)

        y, x = self.position
        h, w = self.size

        img[y: y + int(h/2), x: x + w] = 255
        img[y + int(h/2): y + h, x: x + w] = 126

        return img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)

        y, x = self.position
        h, w = self.size

        img[y: y + h, x: x + int(w/2)] = 255
        img[y: y + h, x + int(w/2): x + w] = 126

        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)

        y, x = self.position
        h, w = self.size

        img[y: y + h, x: x + w] = 255
        img[y + int(h/3): y + 2 * int(h/3), x: x + w] = 126

        return img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)

        y, x = self.position
        h, w = self.size

        # left half is white, right half is gray
        img[y: y + h, x: x + w] = 255
        img[y: y + h, x + int(w / 3): x + 2 * int(w / 3)] = 126

        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)

        y, x = self.position
        h, w = self.size

        # left half is white, right half is gray
        img[y: y + h, x: x + w] = 255
        img[y: y + int(h / 2), x: x + int(w / 2)] = 126
        img[y + int(h / 2): y + h, x + int(w / 2): x + w] = 126

        return img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """
        X = None

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """

        h, w = self.size
        ii = ii.astype(np.float32)

        if self.feat_type == (2, 1):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = self.position[0] + int(h / 2)-1
            x2 = x1 + w - 1
            A = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0] + int(h / 2)
            x1 = self.position[1]
            y2 = self.position[0] + int(h) - 1
            x2 = x1 + w - 1
            B = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            return A - B

        if self.feat_type == (1, 2):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = y1 + int(h) - 1
            x2 = x1 + int(w / 2) - 1
            A = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0]
            x1 = self.position[1] + int(w / 2)
            y2 = y1 + int(h) - 1
            x2 = self.position[1] + int(w) - 1
            B = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            return A - B

        if self.feat_type == (3, 1):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = self.position[0] + int(h / 3) - 1
            x2 = x1 + w - 1
            A = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0] + int(h / 3)
            x1 = self.position[1]
            y2 = self.position[0] + int(h / 3) + int(h / 3) - 1
            x2 = x1 + w - 1
            B = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0] + int(h / 3) + int(h / 3)
            x1 = self.position[1]
            y2 = self.position[0] + h - 1
            x2 = x1 + w - 1
            C = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            return A - B + C

        if self.feat_type == (1, 3):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = y1 + int(h) - 1
            x2 = x1 + int(w / 3) - 1
            A = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0]
            x1 = self.position[1] + int(w / 3)
            y2 = y1 + int(h) - 1
            x2 = self.position[1] + int(w / 3) + int(w / 3) - 1
            B = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0]
            x1 = self.position[1] + int(w / 3) + int(w / 3)
            y2 = self.position[0] + h - 1
            x2 = self.position[1] + int(w) - 1
            C = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            return A - B + C

        if self.feat_type == (2, 2):
            y1 = self.position[0]
            x1 = self.position[1]
            y2 = y1 + int(h / 2) - 1
            x2 = x1 + int(w / 2) - 1
            A = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0]
            x1 = self.position[1] + int(w / 2)
            y2 = y1 + int(h / 2) - 1
            x2 = self.position[1] + w - 1
            B = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0] + int(h / 2)
            x1 = self.position[1]
            y2 = self.position[0] + h - 1
            x2 = x1 + int(w / 2) - 1
            C = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            y1 = self.position[0] + int(h / 2)
            x1 = self.position[1] + int(w / 2)
            y2 = self.position[0] + h - 1
            x2 = self.position[1] + w - 1
            D = ii[y2, x2] - ii[y1 - 1, x2] - ii[y2, x1 - 1] + ii[y1 - 1, x1 - 1]

            return -A + B + C - D


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    returnImageSum = []

    for image in images:
        cumSum = np.cumsum(image , axis=0)
        cumSum = np.cumsum(cumSum, axis=1)

        returnImageSum.append(cumSum)

    return returnImageSum


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in sorted(FeatureTypes.items()):
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))

        # haarFeatures = np.sort(haarFeatures)
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))

        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='double') * 1.0 / (
                           2 * len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='double') * 1.0 / (
                           2 * len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):

            print("start - i = ", i + 1)
            # normalize weights
            weights = weights / np.sum(weights)

            # Instantiate classifier
            VJClassifier = VJ_Classifier(scores, self.labels, weights)
            VJClassifier.train()
            # Append classifier to self.classifiers
            self.classifiers.append(VJClassifier)

            # lowest error
            errors = VJClassifier.error
            # Update weights
            beta = errors / (1 - errors)

            for j in range(len(weights)):
                if int(self.labels[j]) == int(VJClassifier.predict(scores[j])):
                    weights[j] = weights[j] * beta
                else:
                    weights[j] = weights[j] * 1.0

            self.alphas.append(np.log(1 / beta))

            print("end - i = ", i + 1)

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        for clf in self.classifiers:
            feat_id = clf.feature
            hf = self.haarFeatures[feat_id]

            for x, im in enumerate(ii):
                scores[x, feat_id] = hf.evaluate(im)

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).

        for x in scores:

            left = sum(map(lambda idx: self.alphas[idx]*self.classifiers[idx].predict(x), range(len(self.classifiers))))
            right = sum(map(lambda idx: 0.5 * self.alphas[idx], range(len(self.alphas))))

            if left >= right:
                result.append(1)
            else:
                result.append(-1)

        return result

    def faceDetection(self, image, outputDir, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        grayImage = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)
        windowSize = (24, 24)
        filename = filename + ".png"

        # Feature Space
        pos = []
        X = []

        for x in range(grayImage.shape[1] - windowSize[1]):
            for y in range(grayImage.shape[0] - windowSize[0]):
                pos.append((x, y))
                X.append(grayImage[y: y + windowSize[1], x: x + windowSize[0]])

        pred = self.predict(X)
        pos = np.asarray(pos)
        pts = pos[np.where(np.asarray(pred) == 1)]

        pts_mean = np.mean(pts, axis=0)
        x_mean = int(pts_mean[0])
        y_mean = int(pts_mean[1])

        cv2.rectangle(image, tuple([x_mean, y_mean]), tuple([x_mean + windowSize[1], y_mean + windowSize[0]]), (0, 255, 255), 2)
        cv2.imwrite(os.path.join(outputDir, filename), image)