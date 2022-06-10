from Helper_Generate_Mask_From_Images import IMG_HEIGHT, IMG_WIDTH
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cv2 import log
import importlib

import Helper_Generate_Mask_From_Images

importlib.reload(Helper_Generate_Mask_From_Images)


# ppm: postprocessing model

def tf_inv_fourier_trans(img):
    return tf.math.real(tf.signal.ifft2d(img))


def soft_blending(clean, filmed, ones_t2c, u_net_output_t2c):
    # FORMEL: out * clean + (1 - out) * filmed
    # out == mask
    # t2c -> transfered to complex
    debug_softblending = False

    if debug_softblending:
        print("++++++++++++++++++++++++++++ Betrete Softblending!!")
        print("clean: ")
        print(clean[0][0])

        print("filmed: ")
        print(filmed[0][0])

        print("ones_t2c: ")
        print(ones_t2c[0][0])

        print("u_net_output_t2c: ")
        print(u_net_output_t2c[0][0])

    zw1 = tf.math.subtract(ones_t2c, u_net_output_t2c)
    zw1 = tf.math.multiply(zw1, filmed)

    if debug_softblending:
        print("zw1: ")
        print(zw1[0][0])
    zw2 = tf.multiply(u_net_output_t2c, clean)

    if debug_softblending:
        print("zw2: ")
        print(zw2[0][0])
    result = tf.math.add(zw1, zw2)

    if debug_softblending:
        print("result: ")
        print(result[0][0])
    return result


def do_image_blending_and_stack_grayscale_to_rgb(
    unet_output,
    img_clean_complex_r,
    img_clean_complex_g,
    img_clean_complex_b,
    img_filmed_complex_r,
    img_filmed_complex_g,
    img_filmed_complex_b,
):

    ones = tf.ones((128, 128))
    zeros = tf.zeros((128))
    ones_t2c = tf.complex(ones, zeros)

    # NUR ZUM TESTEN
    # unet_output = tf.multiply(unet_output, 0)
    # unet_output = tf.add(unet_output, 0.001)

    u_net_output_t2c = tf.complex(unet_output, zeros)

    img_processed_complex_fourier_r = soft_blending(
        img_clean_complex_r,
        img_filmed_complex_r,
        ones_t2c,
        u_net_output_t2c
    )

    img_processed_complex_fourier_g = soft_blending(
        img_clean_complex_g,
        img_filmed_complex_g,
        ones_t2c,
        u_net_output_t2c
    )

    img_processed_complex_fourier_b = soft_blending(
        img_clean_complex_b,
        img_filmed_complex_b,
        ones_t2c,
        u_net_output_t2c
    )

    # ----------- INVERSE FOURIER TRANSFORMATION -----------

    img_processed_r = tf_inv_fourier_trans(
        img_processed_complex_fourier_r
    )  # / 255
    # plt.imshow(img_processed_r, cmap="gray")
    # plt.show()

    img_processed_g = tf_inv_fourier_trans(
        img_processed_complex_fourier_g
    )  # / 255
    # plt.imshow(img_processed_g, cmap="gray")
    # plt.show()

    img_processed_b = tf_inv_fourier_trans(
        img_processed_complex_fourier_b
    )  # / 255
    # plt.imshow(img_processed_b, cmap="gray")
    # plt.show()

    img_processed_rgb = tf.stack(
        [img_processed_r, img_processed_g, img_processed_b], axis=-1)

    # plt.imshow(img_processed_rgb*255)
    # plt.show()

    img_processed_rgb = img_processed_rgb
    img_processed_rgb = Helper_Generate_Mask_From_Images.normalize_0_1(
        img_processed_rgb)

    img_processed_rgb = tf.reshape(
        img_processed_rgb, (1, IMG_WIDTH, IMG_HEIGHT, 3))

    return (img_processed_rgb, img_processed_r, u_net_output_t2c)
