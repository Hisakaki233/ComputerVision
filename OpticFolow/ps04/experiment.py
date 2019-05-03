"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "output"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    while(level > 0):

        u = 2 * ps4.expand_image(u)[:pyr[level-1].shape[0], :pyr[level-1].shape[1]]
        v = 2 * ps4.expand_image(v)[:pyr[level-1].shape[0], :pyr[level-1].shape[1]]

        level -= 1

    return u, v


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(filename, image)


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def helper_for_part_6(video_name, fps, frame_ids, output_video,
                      counter_init):

    # Initialize two videos - 1 frame difference
    image_gen = ps4.video_frame_generator(video_name)

    image1 = image_gen.__next__()
    image2 = image_gen.__next__()

    imageList = list()
    imageList.append(image1)
    imageList.append(image2)

    h, w, d = image1.shape

    video_out = mp4_video_writer(output_video, (w, h), fps)

    output_counter = counter_init
    frame_num = 1

    levels = 4  # TODO: Define the number of returnImg levels
    k_size = 135  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 25  # TODO: Select a sigma value if you are using a gaussian kernel

    interpolation = cv2.INTER_LINEAR  # You may try different values
    border_mode = cv2.BORDER_REFLECT  # You may try different values

    scale = 0.05
    stride = 10
    color = (0, 0, 255)

    image = image2
    while image is not None:

        print("Processing fame {}".format(frame_num))

        image1_blur = cv2.cvtColor(cv2.GaussianBlur(np.copy(imageList[-2]), (35, 35), 15), cv2.COLOR_BGR2GRAY)
        image2_blur = cv2.cvtColor(cv2.GaussianBlur(np.copy(imageList[-1]), (35, 35), 15), cv2.COLOR_BGR2GRAY)

        u, v = ps4.hierarchical_lk(image1_blur, image2_blur, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

        img_out = np.copy(imageList[-2])

        for y in range(0, v.shape[0], stride):

            for x in range(0, u.shape[1], stride):
                cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                           y + int(v[y, x] * scale)), color, 1)
                cv2.circle(img_out, (x + int(u[y, x] * scale),
                                     y + int(v[y, x] * scale)), 1, color, 1)

        frame_id = frame_ids[(output_counter - 1) % 2]

        if frame_num == frame_id:
            out_str = "output/ps4-6-{}.png".format(output_counter)
            save_image(out_str, img_out)
            output_counter += 1

        video_out.write(img_out)

        image = image_gen.__next__()
        imageList.append(image)

        frame_num += 1

    video_out.release()


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 40  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 5  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.
    # Smoothing

    tempImageA = np.copy(shift_0)
    tempImageB = np.copy(shift_r5_u5)

    smoothKernel = np.ones((7, 7), np.float32) / 49
    tempImageA = cv2.filter2D(tempImageA, -1, smoothKernel)
    tempImageB = cv2.filter2D(tempImageB, -1, smoothKernel)

    k_size = 64 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel

    u, v = ps4.optic_flow_lk(tempImageA, tempImageB, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    # Now let's try with ShiftR10. You may want to try smoothing the image at first
    # Optional: smooth the images if LK doesn't work well on raw images
    tempImageA = cv2.GaussianBlur(np.copy(shift_0), (11, 11), 9)
    tempImageB = cv2.GaussianBlur(np.copy(shift_r10), (11, 11), 9)

    k_size = 100 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel

    u, v = ps4.optic_flow_lk(tempImageA, tempImageB, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=2, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    # Now let's try with ShiftR20. You may want to try smoothing the image at first
    # Optional: smooth the images if LK doesn't work well on raw images
    tempImageA = cv2.GaussianBlur(np.copy(shift_0), (19, 19), 14)
    tempImageB = cv2.GaussianBlur(np.copy(shift_r20), (19, 19), 14)

    k_size = 121 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel

    u, v = ps4.optic_flow_lk(tempImageA, tempImageB, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)

    # Now let's try with ShiftR40. You may want to try smoothing the image at first
    # Optional: smooth the images if LK doesn't work well on raw images
    tempImageA = cv2.GaussianBlur(np.copy(shift_0), (21, 21), 17)
    tempImageB = cv2.GaussianBlur(np.copy(shift_r40), (21, 21), 17)

    k_size = 144 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel

    u, v = ps4.optic_flow_lk(tempImageA, tempImageB, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)
    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    # Pre-processing
    # smoothKernel = np.ones((15, 15), np.float32) / 225
    # yos_img_01 = cv2.filter2D(yos_img_01, -1, smoothKernel)
    # yos_img_02 = cv2.filter2D(yos_img_02, -1, smoothKernel)

    # Actual process
    levels = 1  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 121 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    # Pre-processing
    # smoothKernel = np.ones((15, 15), np.float32) / 225
    # yos_img_02 = cv2.filter2D(yos_img_02, -1, smoothKernel)
    # yos_img_03 = cv2.filter2D(yos_img_03, -1, smoothKernel)

    levels = 1  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 0 # TODO: Select the level number (or id) you wish to use
    k_size = 121 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 2  # TODO: Define the number ofreturnImg levels
    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel

    interpolation = cv2.INTER_LINEAR  # You may try different values
    border_mode = cv2.BORDER_REFLECT  # You may try different values

    shift_0_blur = cv2.GaussianBlur(np.copy(shift_0), (15, 15), 5)
    shift_r10_blur = cv2.GaussianBlur(np.copy(shift_r10), (15, 15), 5)

    u10, v10 = ps4.hierarchical_lk(shift_0_blur, shift_r10_blur, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=0.8, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.

    levels = 3  # TODO: Define the number of levels
    k_size = 45  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 21  # TODO: Select a sigma value if you are using a gaussian kernel

    shift_0_blur = cv2.GaussianBlur(np.copy(shift_0), (35, 35), 15)
    shift_r20_blur = cv2.GaussianBlur(np.copy(shift_r20), (35, 35), 15)

    u20, v20 = ps4.hierarchical_lk(shift_0_blur, shift_r20_blur, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=0.35, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    levels = 3  # TODO: Define the number of levels
    k_size = 65  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 25  # TODO: Select a sigma value if you are using a gaussian kernel

    shift_0_blur = cv2.GaussianBlur(np.copy(shift_0), (45, 45), 21)
    shift_r40_blur = cv2.GaussianBlur(np.copy(shift_r40), (45, 45), 21)

    u40, v40 = ps4.hierarchical_lk(shift_0_blur, shift_r40_blur, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=0.20, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 6  # TODO: Define the number of levels
    k_size = 75  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 35  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_LINEAR  # You may try different values
    border_mode = cv2.BORDER_REPLICATE  # You may try different values

    urban_img_01 = cv2.GaussianBlur(np.copy(urban_img_01), (45, 45), 21)
    urban_img_02 = cv2.GaussianBlur(np.copy(urban_img_02), (45, 45), 21)

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=0.20, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REPLICATE  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    levels = 3  # TODO: Define the number of levels
    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel

    interpolation = cv2.INTER_LINEAR  # You may try different values
    border_mode = cv2.BORDER_REFLECT  # You may try different values

    img_t00 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    img_t10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.

    img_t00_blur = cv2.GaussianBlur(img_t00, (15, 15), 5)
    img_t10_blur = cv2.GaussianBlur(img_t10, (15, 15), 5)

    returnImage = ps4.frame_interpolation(img_t00, img_t10, img_t00_blur, img_t10_blur,
                                          levels, k_size, k_type, sigma,
                                          interpolation, border_mode)

    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-1.png"), ps4.normalize_and_scale(returnImage))


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    levels = 4  # TODO: Define the number of levels
    k_size = 95  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 25  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_LINEAR  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    mc01 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc01.png'), 0) / 255.
    mc02 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc02.png'), 0) / 255.

    mc01blur = cv2.GaussianBlur(mc01, (35, 35), 17)
    mc02blur = cv2.GaussianBlur(mc02, (35, 35), 17)

    returnImage = ps4.frame_interpolation(mc01, mc02, mc01blur, mc02blur,
                                          levels, k_size, k_type, sigma,
                                          interpolation, border_mode)

    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-b-1.png"), ps4.normalize_and_scale(returnImage))

    levels = 6  # TODO: Define the number of levels
    k_size = 115  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 25  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_LINEAR  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    mc02 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc02.png'), 0) / 255.
    mc03 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc03.png'), 0) / 255.

    mc02blur = cv2.GaussianBlur(mc02, (45, 45), 21)
    mc03blur = cv2.GaussianBlur(mc03, (45, 45), 21)

    returnImage = ps4.frame_interpolation(mc02, mc03, mc02blur, mc03blur,
                                          levels, k_size, k_type, sigma,
                                          interpolation, border_mode)

    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-b-2.png"), ps4.normalize_and_scale(returnImage))


def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    print("\nPart 6:")

    my_video = "input_videos/ps4-my-video.mp4"  # Place your video in the input_video directory
    frame_ids = [155, 455]
    fps = 40

    helper_for_part_6(my_video, fps, frame_ids, "output/ps4-6-video.mp4", 1)


if __name__ == '__main__':
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
