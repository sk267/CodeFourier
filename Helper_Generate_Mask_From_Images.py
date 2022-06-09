import numpy as np
import cv2
from skimage.exposure import adjust_gamma

IMG_HEIGHT, IMG_WIDTH = 128, 128


def f2pd(fourier_array):
    # transoforms an fourier image into pixel domain in order
    # to display it
    # old:
    return (np.log(abs(fourier_array)))


def normalize_0_1(img):
    max = np.max(img)
    min = np.min(img)
    img = (img - min) / (max - min)
    return(img)


def combine_3_fourier_to_one_grayscale(im1, im2, im3):
    img_combined = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    im1 = f2pd(im1)  # * RGB_WEIGHTS[0]
    im2 = f2pd(im2)  # * RGB_WEIGHTS[1]
    im3 = f2pd(im3)  # * RGB_WEIGHTS[2]
    img_combined = im1 + im2 + im3

    # put values between 0 - 1:
    max_value = np.max(np.array(img_combined))
    img_combined /= max_value

    return img_combined


def generate_mask_from_images(
    img_filmed_complex_r,
    img_filmed_complex_g,
    img_filmed_complex_b,
    img_clean_complex_r,
    img_clean_complex_g,
    img_clean_complex_b
):
    # split to separate grayscale images

    img_filmed_fourier_combined = combine_3_fourier_to_one_grayscale(
        img_filmed_complex_r,
        img_filmed_complex_g,
        img_filmed_complex_b
    )

    img_clean_fourier_combined = combine_3_fourier_to_one_grayscale(
        img_clean_complex_r,
        img_clean_complex_g,
        img_clean_complex_b
    )

    DILATE_KERNEL_SIZE = (10, 10)

    # A - B (Filmed - Clean)
    fourier_mask = img_filmed_fourier_combined - \
        img_clean_fourier_combined
    # plt.imshow(fourier_mask, cmap="gray")
    # plt.show()

    # normalize image
    fourier_mask = normalize_0_1(fourier_mask)

    # Blur Image
    fourier_mask = cv2.blur(fourier_mask, (6, 6))

    # Enlarge Contrast
    fourier_mask = adjust_gamma(fourier_mask, 15)

    # Threshold Image
    fourier_mask = np.where(fourier_mask > 0.0005, 1, 0)

    # Dilate Image
    kernel = np.ones(DILATE_KERNEL_SIZE, np.float)
    fourier_mask = cv2.dilate(
        fourier_mask.astype("uint8"), kernel)

    return fourier_mask
