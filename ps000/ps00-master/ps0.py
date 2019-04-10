"""
CS6476 Computer Vision
Problem Set 0 
Simple script to verify your installation
"""
import unittest

class TestInstall(unittest.TestCase):
    def setUp(self):
        pass

    def test_numpy(self):
        import numpy

    def test_cv2(self):
        import cv2

    def test_cv2_version(self):
        import cv2
        v = cv2.__version__.split(".")
        self.assertTrue(v[0] == '4' and v[1] == '0', 'Wrong OpenCV version. '
                                                     'Make sure you installed OpenCV 3.1.x.'
                                                     'Any other OpenCV versions i.e. 2.x are not supported.')

    def test_ORB(self):
        from cv2 import ORB_create
        test_orb = ORB_create()

    def test_load_mp4_videos(self):
        import cv2
        video = cv2.VideoCapture("turtle.mp4")
        if video.isOpened():
            okay, frame = video.read()
            self.assertTrue(okay, "Loading mp4 video failed")
        else:
            self.assertTrue(False, 'Video Failed to Open.')

    def test_show_image(self):
        import cv2
        video = cv2.VideoCapture("turtle.mp4")
        okay, frame = video.read()
        cv2.imshow('Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()
