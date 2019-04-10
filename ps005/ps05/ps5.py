"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):

    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        # State vector(X) with the initial x and y values.
        self.state = np.array([[init_x], [init_y], [0.], [0.]])  # state

        # Covariance 4x4 array ( Sigma ) initialized with a diagonal matrix with some value.
        self.covariance = 1000 * np.eye(4)

        # 4x4 state transition matrix Dt
        dt = 1
        self.Dt = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # 2x4 measurement matrix Mt
        self.Mt = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # 4x4 process noise matrix Sigmadt
        self.dt = Q

        # 2x2 measurement noise matrix Sigmamt
        self.mt = R

    def predict(self):

        self.state = self.Dt.dot(self.state)
        self.covariance = self.Dt.dot(self.covariance).dot(self.Dt.transpose()) + self.dt

    def correct(self, meas_x, meas_y):

        # Kalman Gain K
        linearInverse = self.Mt.dot(self.covariance).dot(self.Mt.transpose()) + self.mt
        K = self.covariance.dot(self.Mt.transpose()).dot(np.linalg.inv(linearInverse))

        # Correction
        self.state = self.state + K.dot(np.array([[meas_x], [meas_y]]) - self.Mt.dot(self.state))
        self.covariance = (np.eye(4) - K.dot(self.Mt)).dot(self.covariance)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.

        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.template = template
        self.frame = frame

        # Gray Image
        self.modelGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float64)
        # self.modelGray = 0.12 * template[0] + 0.58 * template[1] + 0.3 * template[2]

        # Initialize your particles array. Read the docstring.
        self.particles = np.array([np.random.uniform(self.template_rect['x'],
                                                     self.template_rect['x'] + 2 * self.template_rect['w'],
                                                     self.num_particles),
                                   np.random.uniform(self.template_rect['y'],
                                                     self.template_rect['y'] + 2 * self.template_rect['h'],
                                                     self.num_particles)]).T

        # Initialize your weights array. Read the docstring.
        self.weights = np.array(np.ones(self.num_particles)/int(np.sum(self.num_particles)))

        # Initialize any other components you may need when designing your filter.
        self.avgSimilarity = 1e-1000
        self.weightsNew = True

    def particleInitial(self):
        # particleInitial particles follows normal distribution, this sigma_dyn is the control std
        noise = np.random.normal(0, self.sigma_dyn, self.particles.shape)
        particles = self.particles + noise

        # Cap and Limit value
        particles[:, 0] = np.clip(particles[:, 0], 0, self.frame.shape[1] - 1)
        particles[:, 1] = np.clip(particles[:, 1], 0, self.frame.shape[0] - 1)

        return particles

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        if template.shape != frame_cutout.shape:
            return 0.0
        else:
            tse = np.sum(np.subtract(template, frame_cutout, dtype=np.float64) ** 2)
            measureProb = np.exp(- (tse / float(template.shape[0] * template.shape[1])) / (2 * self.sigma_exp ** 2))
            return measureProb

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        j = np.random.choice(np.arange(self.num_particles), self.num_particles, True, p=self.weights.T)
        # resample the particles using the distribution of the weights
        return np.array(self.particles[j])

    def particle_filter(self, image):

        self.particles = self.resample_particles()
        particles = self.particleInitial()

        modelGrayTemp = [1] * len(particles)

        if particles.shape[1] == 2:

            positionX = particles[:, 0].astype(np.int)
            positionY = particles[:, 1].astype(np.int)

            h, w = self.modelGray.shape

            particleFrame = [1] * len(particles)

            for i in range(particles.shape[0]):
                particleFrame[i] = image[np.clip(positionY[i] - int(h/2), 0, image.shape[0]-1):np.clip(positionY[i] + h - int(h/2), 0, image.shape[0]-1),
                                         np.clip(positionX[i] - int(w/2), 0, image.shape[1]-1):np.clip(positionX[i] + w - int(w/2), 0, image.shape[1]-1)]


            self.particles = particles
            self.weights = np.array([self.get_error_metric(self.modelGray, np.array(particleFrame[i])) for i in range(len(particleFrame))])
            self.weights /= np.sum(self.weights)

        else:

            positionX = particles[:, 0].astype(np.int)
            positionY = particles[:, 1].astype(np.int)

            particleFrame = [1] * len(particles)

            for img in range(len(particleFrame)):
                modelGrayTemp[img] = cv2.resize(self.modelGray, None, fx=particles[img, 2], fy=particles[img, 2])

                h, w = modelGrayTemp[img].shape

                particleFrame[img] = image[np.clip(positionY[img] - int(h/2), 0, image.shape[0]-1):np.clip(positionY[img] + h - int(h/2), 0, image.shape[0]-1),
                                           np.clip(positionX[img] - int(w/2), 0, image.shape[1]-1):np.clip(positionX[img] + w - int(w/2), 0, image.shape[1]-1)]

            newWeights = np.array([self.get_error_metric(np.array(modelGrayTemp[i]), np.array(particleFrame[i])) for i in range(len(particleFrame))])
            if np.any(newWeights > 0.0):
                newWeights /= np.sum(newWeights)

            # if np.any(newWeights > self.avgSimilarity):
            if np.any(newWeights > self.avgSimilarity):
                non_zero_weights = newWeights[np.where(newWeights > 0)]
                self.particles = particles
                self.avgSimilarity = np.percentile(non_zero_weights, 75)
                self.weights = newWeights
                self.weightsNew = True
            else:
                self.weightsNew = False


        # print(self.weights)
        # self.weights = np.clip(self.weights, 1e-1000, np.max(self.weights))


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        # imageGray = 0.12 * frame[0] + 0.58 * frame[1] + 0.3 * frame[2]
        self.particle_filter(imageGray)


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.

        for p in self.particles:
            p = p[:2]
            cv2.circle(frame_in, tuple(p.astype(int)), 1, (0, 255, 255), -1)

        if self.particles.shape[1] <= 2:
            point1_y = (y_weighted_mean - self.modelGray.shape[0] / 2).astype(np.int)
            point1_x = (x_weighted_mean - self.modelGray.shape[1] / 2).astype(np.int)

            point2_y = (y_weighted_mean + self.modelGray.shape[0] / 2).astype(np.int)
            point2_x = (x_weighted_mean + self.modelGray.shape[1] / 2).astype(np.int)
        else:

            scalar = self.particles[np.argmax(self.weights), 2]
            point1_y = (y_weighted_mean - scalar * self.modelGray.shape[0] / 2).astype(np.int)
            point1_x = (x_weighted_mean - scalar * self.modelGray.shape[1] / 2).astype(np.int)

            point2_y = (y_weighted_mean + scalar * self.modelGray.shape[0] / 2).astype(np.int)
            point2_x = (x_weighted_mean + scalar * self.modelGray.shape[1] / 2).astype(np.int)

        cv2.rectangle(frame_in, (point1_x, point1_y), (point2_x, point2_y), (0, 255, 255), 2)

        weightedAverage = np.array((x_weighted_mean, y_weighted_mean))
        normalizedParticles = self.particles[:, :2] - weightedAverage
        dist = normalizedParticles[:, 0] ** 2 + normalizedParticles[:, 1] ** 2
        weighted_sum = np.sum(dist * self.weights)

        try:
            cv2.circle(frame_in, tuple(weightedAverage[:2].astype(np.int)), int(weighted_sum), (0, 0, 255), 2)
        except ValueError:
            pass


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def updateTemplate(self, image):

        # based on the resampled particles, only one state will be selected

        idx = np.random.choice(np.arange(0, self.num_particles), 1, p=self.weights)
        # print(self.particles[idx, 2])
        h, w = self.modelGray.shape
        # Subset Image
        Y_nbr = self.particles[idx][0][1].astype(np.int)
        x_nbr = self.particles[idx][0][0].astype(np.int)

        best = image[Y_nbr - int(h / 2):Y_nbr + h - int(h / 2),
                     x_nbr - int(w / 2):x_nbr + w - int(w / 2)]

        if self.particles.shape[1] <= 2:
            self.modelGray = self.alpha * best + (1.0 - self.alpha) * self.modelGray
        else:
            best = cv2.resize(best, self.modelGray.shape)
            self.modelGray = self.alpha * best + (1.0 - self.alpha) * self.modelGray


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        self.particle_filter(imageGray)
        self.updateTemplate(imageGray)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        # Initialize your particles array. Adding Scalar into existed model. Read the docstring.

        self.particles = np.array([np.random.uniform(self.template_rect['x'],
                                                     self.template_rect['x'] + 2 * self.template_rect['w'],
                                                     self.num_particles),
                                   np.random.uniform(self.template_rect['y'],
                                                     self.template_rect['y'] + 2 * self.template_rect['h'],
                                                     self.num_particles),
                                   np.random.uniform(0.9, 1.0, self.num_particles)]).T

    def particleInitial(self):
        # particleInitial particles follows normal distribution, this sigma_dyn is the control std
        noise = np.random.normal(0, self.sigma_dyn, (self.num_particles, 2))
        noiseScalar = np.random.normal(0, 0.01, (self.num_particles, 1))
        particles = self.particles + np.hstack((noise, noiseScalar))

        # Cap and Limit value
        particles[:, 0] = np.clip(particles[:, 0], 0, self.frame.shape[1] - 1)
        particles[:, 1] = np.clip(particles[:, 1], 0, self.frame.shape[0] - 1)
        particles[:, 2] = np.clip(particles[:, 2], 0.1, 1.00)

        return particles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        self.particle_filter(imageGray)
