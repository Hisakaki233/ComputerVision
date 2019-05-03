import matplotlib.pyplot
import cv2
import numpy as np
import os

VID_DIR = "inputVideo"
OUT_DIR = "outputVideo"
OUT_IMG = "outImage"

actions = {'boxing': 1, 'handclapping': 2, 'handwaving': 3, 'jogging': 4, 'running': 5, 'walking': 6}


# Helper Functions are from previous Homework
def helper_video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None


def helper_video_create(video_name, predict, framesStart, frameLength, outVideo):

    # Calculate Frame Pairs
    framesEnd = dict()
    for act, values in framesStart.items():
        framesEnd[act] = [i + frameLength[act] for i in values]

    frames = list()
    for keys in framesStart.keys():
        for j in range(len(framesStart[keys])):
            frames.append((framesStart[keys][j], framesEnd[keys][j]))

    # Suppose to contain all 36 frames
    frames = sorted(frames, key=lambda x: x[0])
    y = sorted([1, 2, 3, 4, 5, 6] * 6)

    fps = 20

    video = os.path.join(VID_DIR, video_name)
    image_gen = helper_video_frame_generator(video)

    image = image_gen.__next__()
    h, w, d = image.shape

    out_path = os.path.join(OUT_DIR, outVideo)
    video_out = helper_mp4_video_writer(out_path, (w, h), fps)

    save_image_counter = 1

    frame_num = 1
    while image is not None:

        for i, (start, end) in enumerate(frames):
            if (frame_num >= start) & (frame_num <= end):
                helper_mark_location(image, i, y[i], predict[i])

                if frame_num == start + 10:
                    out_str = "Test_frame_output" + "-{}.png".format(frame_num)
                    helper_save_image(out_str, image)
                    save_image_counter += 1

                video_out.write(image)
                break

        image = image_gen.__next__()
        frame_num += 1

    video_out.release()


def helper_mp4_video_writer(filename, frame_size, fps=40):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def helper_save_image(filename, image):
    cv2.imwrite(os.path.join(OUT_IMG, filename), image)


def helper_mark_location(image, sample, result, predict):

    actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

    color1 = (205, 0, 0)
    color2 = (0, 0, 205)
    h, w, d = image.shape
    p1 = [int(w/10), int(h/5)]

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, "(Sample:{})".format(sample+1), (p1[0]-5, p1[1]+0), font, 0.3, color1, 1)
    cv2.putText(image, "(Actual:{})".format(actions[result-1]), (p1[0]-5, p1[1]+20), font, 0.3, color1, 1)

    if actions[result-1] == actions[predict-1]:
        cv2.putText(image, "(Prediction:{})".format(actions[predict-1]), (p1[0]+5, p1[1]+40), font, 0.3, color1, 1)
    else:
        cv2.putText(image, "(Prediction:{})".format(actions[predict-1]), (p1[0]+5, p1[1]+40), font, 0.3, color2, 1)


class MHIClassifier():

    def __init__(self, imageHuMoment, lableMatrix):
        self.classifier = cv2.ml.KNearest_create()
        self.X = imageHuMoment
        self.y = lableMatrix
        self.neighbours = list()

    def crossValidation(self):
        # train the model and predict the test y
        cnfmt = np.zeros((6, 6))
        for i in range(len(self.X)):
            j = list(range(0, i)) + list(range(i+1, self.X.shape[0]))
            # train dataset
            Xt = np.array(self.X[j])
            yt = np.array(self.y[j])

            # train knn classifier
            self.classifier.train(Xt, cv2.ml.ROW_SAMPLE, yt)

            Xte = np.array([self.X[i]])
            yte = np.array([self.y[i]])
            # predict test data labels, res is an (1,1) array, but only 1 element, so extract it
            retval, res, neighborResponses, dists = self.classifier.findNearest(Xte, 5)

            cnfmt[yte-1, int(res[0])-1] += 1

        self.confusionMatrix(cnfmt, actions=['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'])

    def train(self):
        # train dataset
        # train knn classifier
        self.classifier.train(self.X, cv2.ml.ROW_SAMPLE, self.y)

    def predict(self, X_test):
        # predict test data labels, res is an (1,1) array, but only 1 element, so extract it
        ret, results, neighbours, dist = self.classifier.findNearest(X_test.T, 5)

        self.y_hat = results[0, 0]
        self.neighbours.append(neighbours[0])

    def confusionMatrix(self, cm, actions):
        cm = (cm * 100 / cm.sum()).astype(np.uint) / 100.0
        th = cm.max() / 2.
        matplotlib.pyplot.figure(figsize=(8, 6.5))
        matplotlib.pyplot.imshow(cm, interpolation='nearest', cmap=matplotlib.pyplot.cm.Oranges)
        matplotlib.pyplot.title('Confusion matrices')
        matplotlib.pyplot.ylabel('Actual')
        matplotlib.pyplot.xlabel('Predicted')
        tm = np.arange(len(actions))
        matplotlib.pyplot.xticks(tm, actions)
        matplotlib.pyplot.yticks(tm, actions)
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.tight_layout()
        filename = 'confusion_matrices.png'
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                matplotlib.pyplot.text(j, i, cm[i, j], horizontalalignment="center",
                                       color="white" if cm[i, j] > th else "black")

        matplotlib.pyplot.savefig(os.path.join(OUT_IMG, filename))


class MHIModel():
    def __init__(self):
        '''
        self.frameLength = {'boxing': 36,
                            'handclapping': 27,
                            'handwaving': 48,
                            'jogging': 35,
                            'running': 25,
                            'walking': 45}
        '''

        self.frameLength = {'boxing': 30,
                            'handclapping': 25,
                            'handwaving': 50,
                            'jogging': 35,
                            'running': 25,
                            'walking': 45}

        self.frameLengthSet = [25, 30, 35, 40, 45, 50]

        # theta = [15, 15, 15, 15, 15, 15]
        # theta = [8, 8, 10, 25, 15, 35]
        # theta = [3, 3, 10, 30, 15, 45]
        # theta = [3, 3, 10, 30, 30, 30]

        # tau needs to be chosen and differs with actions
        # tau = [50, 50, 50, 50, 50, 50]
        # tau = [50, 30, 50, 35, 15, 55]
        # tau = [35, 15, 50, 40, 25, 65]
        # tau = [35, 20, 50, 45, 45, 45]

        self.theta = [8, 8, 35, 35, 25, 40]
        self.tau = [15, 18, 65, 55, 45, 65]

        # self.validateTheta = [17] * 6
        # self.validateTau = [40] * 6

        self.validateTheta = [15] * 6
        self.validateTau = [40] * 6

    def createBinaryMotionSignal(self, blurKernel, blurSigma, video_name, start, end, theta):

        video = os.path.join(VID_DIR, video_name)

        image_gen = helper_video_frame_generator(video)
        image = image_gen.__next__()

        frame = 0

        # Define my Return Data
        binaryImageSet = []
        imageBset = []

        while image is not None:

            # Start Processing Images
            if (frame >= start) and (frame <= end):
                # Create Image and ImageB for t and t+1
                image1 = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)
                image1 = cv2.GaussianBlur(image1, blurKernel, blurSigma)

                imageB = image_gen.__next__()

                if imageB is None:
                    break
                else:
                    image2 = cv2.cvtColor(np.copy(imageB), cv2.COLOR_BGR2GRAY)
                    image2 = cv2.GaussianBlur(image2, blurKernel, blurSigma)

                    # Binary Image Bt(x, y)
                    binaryImage = np.abs(cv2.subtract(image2, image1)) >= theta
                    binaryImage = binaryImage.astype(np.uint8)
                    # print(np.any(binaryImage > 0))

                    # "clean up" the difference Image by Applying morphologyEx()
                    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
                    kernel = np.ones((3, 3), np.uint8)
                    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)
                    imageBset.append(image2)
                    binaryImageSet.append(binaryImage)

                    # Assign Image B to Image A - use Copy to make sure they won't change together
                    image = np.copy(imageB)
                    frame += 1

            # move to next Image if not executed
            else:
                image = image_gen.__next__()
                frame += 1

        return binaryImageSet, imageBset

    def createMHI(self, start, end, binaryImageSet, tau):
        # Input shall be Binary Motion Signal from Function createBinaryMotionSignal
        # return MHI image

        Itau = np.zeros(binaryImageSet[0].shape, dtype=np.float)

        for i, image in enumerate(binaryImageSet):
            if i == (end - start):
                break
            mask1 = (image == 1)
            mask0 = (image == 0)
            Itau = tau * mask1 + np.clip(np.subtract(Itau, np.ones(Itau.shape)), 0, 255) * mask0

        return Itau.astype(np.uint8)

    def createHuMoment(self, image):
        # refer to paper
        # return 3 dictionary
        # derive all 7 mu's - exclude mu00, mu10, mu01, since mu10 = mu01 = 0
        pqpair = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]

        # calculate central moments upq
        u00 = image.sum()
        h, w = image.shape

        # Reshape to two dimension to secure broadcasting
        x_bar = np.sum(np.arange(w).reshape(1, -1) * image, keepdims=True) / u00
        y_bar = np.sum(np.arange(h).reshape(-1, 1) * image, keepdims=True) / u00

        mus = dict()
        eta = dict()
        for i, (p, q) in enumerate(pqpair):
            pq = str(p) + str(q)
            # Same idea to calculate all mu's
            diffx = np.arange(w).reshape(1, -1) - x_bar
            diffy = np.arange(h).reshape(-1, 1) - y_bar

            # calculate the central moments
            upq = np.sum((diffx ** p).reshape(1, -1) * (diffy ** q).reshape(-1, 1) * image, keepdims=True)
            mus[pq] = upq

            # calculate the scale invariance moments
            eta[pq] = mus[pq] / u00 ** (1 + (p + q) / 2)

        # Generate all invariant Hu Moment once have all eta
        momentHu = np.zeros(len(pqpair))

        momentHu[0] = eta['20'] + eta['02']
        momentHu[1] = (eta['20'] - eta['02']) ** 2 + 4 * eta['11'] ** 2
        momentHu[2] = (eta['30'] - 3 * eta['12']) ** 2 + (3 * eta['21'] - eta['03']) ** 2
        momentHu[3] = (eta['30'] + eta['12']) ** 2 + (eta['21'] + eta['03']) ** 2
        momentHu[4] = ((eta['30'] - 3 * eta['12']) * (eta['30'] + eta['12']) * ((eta['30'] + eta['12']) ** 2 - 3 * (eta['21'] + eta['03']) ** 2)
                       + (3 * eta['21'] - eta['03']) * (eta['21'] + eta['03']) * (3 * (eta['30'] + eta['12']) ** 2 - (eta['21'] + eta['03']) ** 2))
        momentHu[5] = ((eta['20'] - eta['02']) * ((eta['30'] + eta['12']) ** 2 - (eta['21'] + eta['03']) ** 2)
                       + 4 * eta['11'] * (eta['30'] + eta['12']) * (eta['21'] + eta['03']))
        momentHu[6] = ((3 * eta['21'] - eta['03']) * (eta['30'] + eta['12']) * ((eta['30'] + eta['12']) ** 2 - 3 * (eta['21'] + eta['03']) ** 2)
                       - (eta['30'] - 3 * eta['12']) * (eta['21'] + eta['03']) * (3 * (eta['30'] + eta['12']) ** 2 - (eta['21'] + eta['03']) ** 2))

        return mus, eta, momentHu

    def classifier(self):

        actions = [('boxing', 1), ('handclapping', 2), ('handwaving', 3), ('jogging', 4), ('running', 5), ('walking', 6)]

        video_name = {'boxing': "person15_boxing_d1_uncomp.avi",
                      'handclapping': "person15_handclapping_d1_uncomp.avi",
                      'handwaving': "person15_handwaving_d1_uncomp.avi",
                      'jogging': "person15_jogging_d1_uncomp.avi",
                      'running': "person15_running_d1_uncomp.avi",
                      'walking': "person15_walking_d1_uncomp.avi"}

        # person 15's training data
        # person15_boxing_d1		frames	1-150, 151-216, 217-290, 291-408
        # person15_handclapping_d1	frames	1-65, 66-164, 165-232, 233-312
        # person15_handwaving_d1	frames	1-136, 137-279, 280-440, 441-581
        # person15_jogging_d1		frames	1-75, 140-205, 240-300, 360-420
        # person15_running_d1		frames	1-50, 100-150, 180-225, 280-325
        # person15_walking_d1		frames	1-105, 215-338, 422-540, 630-741

        # frames = {'boxing': [(1, 150), (151, 216), (217, 290), (291, 408)],
        #           'handclapping': [(1, 65), (66, 164), (165, 232), (233, 312)],
        #           'handwaving': [(1, 136), (137, 279), (280, 440), (441, 581)],
        #           'jogging': [(1, 75), (140, 205), (240, 300), (360, 420)],
        #           'running': [(1, 50), (100, 150), (180, 225), (280, 325)],
        #           'walking': [(1, 105), (215, 338), (422, 540), (630, 741)]
        #           }

        framesStart = {'boxing': [1, 36, 72, 108, 160, 220],
                       'handclapping': [1, 27, 54, 81, 108, 135],
                       'handwaving': [1, 48, 96, 138, 160, 208],
                       'jogging': [10, 145, 155, 245, 255, 365],
                       'running': [10, 105, 115, 190, 195, 285],
                       'walking': [15, 220, 235, 430, 440, 640]}

        framesEnd = dict()
        for act, values in framesStart.items():
            framesEnd[act] = [i + self.frameLength[act] for i in values]

        frames = dict()
        for keys in framesStart.keys():
            frames[keys] = list()
            for j in range(len(framesStart[keys])):
                frames[keys].append((framesStart[keys][j], framesEnd[keys][j]))

        # choose theta for Binary Image Creation
        # theta = [15, 15, 15, 15, 15, 15]
        # theta = [8, 8, 10, 25, 15, 35]
        # theta = [3, 3, 10, 30, 15, 45]
        # theta = [3, 3, 10, 30, 30, 30]
        theta = self.theta

        # tau needs to be chosen and differs with actions
        # tau = [50, 50, 50, 50, 50, 50]
        # tau = [50, 30, 50, 35, 15, 55]
        # tau = [35, 15, 50, 40, 25, 65]
        # tau = [35, 20, 50, 45, 45, 45]
        tau = self.tau

        # create all the action MHI, MEI, and labels
        MotionHistoryImgs = []
        y = []

        frame_ids = [5, 10, 15]
        MotionEnergyImgs = []

        for i, (key, action) in enumerate(actions):
            print("key = " + key)

            for j, (start, end) in enumerate(frames[key]):

                # call createBinaryMotionSignal to build binary motion signal, prepare for MHI
                binaryImageSet, imageBset = self.createBinaryMotionSignal((5, 5), 0, video_name[key], start, end, theta[i])

                # MHI image and derive MEI based on MHI
                MHI = self.createMHI(start=start, end=end, binaryImageSet=binaryImageSet, tau=tau[i]).astype(np.float)

                # normalize the motion history image & prepare motion energy images
                cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)
                MEI = (255 * MHI > 0).astype(np.uint8)

                # save the sample MHI
                out_str = "MHI" + "-{}-{}.png".format(key, j)
                helper_save_image(out_str, MHI)

                out_str = "MEI" + "-{}-{}.png".format(key, j)
                helper_save_image(out_str, (255 * MEI))

                MotionHistoryImgs.append(MHI)
                MotionEnergyImgs.append(MEI)
                y.append(action)

            # Output some sample Binary Image & ImageB image
            for j in frame_ids:
                out_str = "binaryImageSet" + "-{}-{}.png".format(key, j)
                outBinaryImageSet = binaryImageSet[j]

                cv2.normalize(outBinaryImageSet, outBinaryImageSet, 0, 255, cv2.NORM_MINMAX)
                helper_save_image(out_str, outBinaryImageSet)

                out_str = "imageBset" + "-{}-{}.png".format(key, j)
                outImageBset = imageBset[j]
                helper_save_image(out_str, outImageBset)



        # call createHuMoment, create central moments and scale invariant moment based on paper
        mus = []
        etas = []
        HuMoment = []

        for MHI, MEI in zip(MotionHistoryImgs, MotionEnergyImgs):
            # print(MHI.shape)
            MHImus, MHIeta, MHIHuMoment = self.createHuMoment(MHI)
            MEImus, MEIeta, MEIHuMoment = self.createHuMoment(MEI)

            mus.append(np.append(MHImus, MEImus))
            etas.append(np.append(MHIeta, MEIeta))
            HuMoment.append(np.append(MHIHuMoment, MEIHuMoment))

        # now change cms and sims to be array, row:number of actions, column: number of moments
        # training Data, classify 6 actions

        y = np.array(y).astype(np.int).reshape(-1, 1)
        HuMoment = np.array(HuMoment).astype(np.float32)

        self.classifier = MHIClassifier(HuMoment, y)
        self.classifier.crossValidation()
        self.classifier.train()

    def recognition(self, video_name, outVideoName):

        actions = [('boxing', 1), ('handclapping', 2), ('handwaving', 3), ('jogging', 4), ('running', 5),
                   ('walking', 6)]

        # video_name = {'boxing': "person04_boxing_d1_uncomp.avi",
        #               'handclapping': "person14_handclapping_d1_uncomp.avi",
        #               'handwaving': "person07_handwaving_d1_uncomp.avi",
        #               'jogging': "person09_jogging_d1_uncomp.avi",
        #               'running': "person18_running_d1_uncomp.avi",
        #               'walking': "person14_walking_d1_uncomp.avi"}

        # Random Person's validation data
        # boxing - frame: 328
        # handclapping - frame: 524
        # handwaving - frame: 782
        # jogging - frame: 465
        # running -  frame: 375
        # walking - frame: 720
        # framesStart = {'boxing': [5, 36, 72, 108, 144, 180],
        #                'handclapping': [5, 27, 54, 81, 108, 135],
        #                'handwaving': [5, 48, 96, 138, 160, 208],
        #                'jogging': [1, 10, 150, 270, 280, 400],
        #                'running': [1, 15, 110, 223, 235, 330],
        #                'walking': [1, 15, 220, 235, 410, 610]}

        framesStart = {'boxing': [5, 36, 72, 108, 144, 180],
                       'handclapping': [i + 328 for i in [5, 27, 54, 81, 108, 135]],
                       'handwaving': [i + 852 for i in [5, 48, 96, 138, 160, 208]],
                       'jogging': [i + 1634 for i in [1, 10, 150, 270, 280, 400]],
                       'running': [i + 2099 for i in [1, 15, 110, 223, 235, 330]],
                       'walking': [i + 2474 for i in [1, 65, 220, 235, 410, 610]]
                       }

        # choose theta for Binary Image Creation
        # theta = [15, 15, 15, 15, 15, 15]
        # theta = [8, 8, 10, 25, 15, 35]
        # theta = [3, 3, 10, 30, 15, 45]
        # theta = [3, 3, 10, 30, 30, 30]
        theta = self.validateTheta

        # tau needs to be chosen and differs with actions
        # tau = [50, 50, 50, 50, 50, 50]
        # tau = [50, 30, 50, 35, 15, 55]
        # tau = [35, 15, 50, 40, 25, 65]
        # tau = [35, 20, 50, 45, 45, 45]
        # tau = [35, 20, 45, 35, 35, 45]
        tau = self.validateTau

        # create all the action MHI, MEI, and labels
        MotionHistoryImgs = []
        y = []
        y_hat = []

        frame_ids = [5, 10, 15]
        MotionEnergyImgs = []

        for i, (key, action) in enumerate(actions):
            print("key = " + key)

            for j, start in enumerate(framesStart[key]):
                frames = []
                y_sim = []
                y.append(action)
                for z in self.frameLengthSet:
                    framesEnd = framesStart[key][j] + z
                    frames.append((framesStart[key][j], framesEnd))

                for i, (start, end) in enumerate(frames):
                    # call createBinaryMotionSignal to build binary motion signal, prepare for MHI
                    binaryImageSet, imageBset = self.createBinaryMotionSignal((5, 5), 0, video_name, start, end,
                                                                              theta[i])

                    # MHI image and derive MEI based on MHI
                    MHI = self.createMHI(start=start, end=end, binaryImageSet=binaryImageSet, tau=tau[i]).astype(np.float)

                    # normalize the motion history image & prepare motion energy images
                    cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)
                    MEI = (255 * MHI > 0).astype(np.uint8)

                    MotionHistoryImgs.append(MHI)
                    MotionEnergyImgs.append(MEI)

                    MHImus, MHIeta, MHIHuMoment = self.createHuMoment(MHI)
                    MEImus, MEIeta, MEIHuMoment = self.createHuMoment(MEI)

                    HuMoment = np.append(MHIHuMoment, MEIHuMoment)

                    # now change cms and sims to be array, row:number of actions, column: number of moments
                    # training Data, classify 6 actions

                    # y = np.array(y).astype(np.int).reshape(-1, 1)
                    HuMoment = np.array(HuMoment).astype(np.float32)

                    self.classifier.predict(np.array(HuMoment).reshape(-1, 1))
                    y_sim.append(self.classifier.y_hat.astype(np.int8))

                y_hat.append(np.argmax(np.bincount(y_sim)))
                self.classifier.neighbours = list()

        acc = np.sum([1 for i in range(len(y)) if y[i] == y_hat[i]]) / len(y)
        print(y)
        print(y_hat)
        print("prediction accuracy is " + str(acc))

        print("----- Generating Testing Video ------")
        helper_video_create(video_name, y_hat, framesStart, self.frameLength, outVideoName)

        print("------ Creating Confusion Matrix ------------")
        confusionMatrix = np.zeros((6, 6))
        for (y, y_hat) in zip(y, y_hat):
            confusionMatrix[y-1, y_hat - 1] += 1

        confusionMatrix = (confusionMatrix * 100.0 / confusionMatrix.sum()).astype(np.uint) / 100.0
        th = confusionMatrix.max() / 2.
        matplotlib.pyplot.figure(figsize=(8, 6.5))
        matplotlib.pyplot.imshow(confusionMatrix, interpolation='nearest', cmap=matplotlib.pyplot.cm.Oranges)
        matplotlib.pyplot.title('Confusion matrices')
        matplotlib.pyplot.ylabel('Actual')
        matplotlib.pyplot.xlabel('Predicted')

        inAction = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        tm = np.arange(len(inAction))
        matplotlib.pyplot.xticks(tm, inAction)
        matplotlib.pyplot.yticks(tm, inAction)
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.tight_layout()

        for i in range(confusionMatrix.shape[0]):
            for j in range(confusionMatrix.shape[1]):
                matplotlib.pyplot.text(j, i, confusionMatrix[i, j], horizontalalignment="center",
                                       color="white" if confusionMatrix[i, j] > th else "black")

        matplotlib.pyplot.savefig(os.path.join(OUT_IMG, 'ValidationConfusionMatrix'))


if __name__=='__main__':
    print("Running....")
    cls = MHIModel()
    print("Training...")
    cls.classifier()
    print("Validating...")
    cls.recognition("testVideo.avi", "outputTestVideo.avi")