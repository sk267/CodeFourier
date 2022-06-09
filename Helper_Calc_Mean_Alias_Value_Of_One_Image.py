# from Helper_Generate_Mask_From_Images import generate_mask_from_images
# from Helper_Do_Postprocessing_Stuff import do_image_blending_and_stack_grayscale_to_rgb
import importlib
import numpy as np
import matplotlib.pyplot as plt
from UNet_Fourier_Facilities import Fourier_Images
import tensorflow as tf
from pydoc import Helper

import Helper_Generate_Mask_From_Images
import Helper_Do_Postprocessing_Stuff

importlib.reload(Helper_Generate_Mask_From_Images)
importlib.reload(Helper_Do_Postprocessing_Stuff)


def calc_mean_alias_value_of_one_image(
        x_filmed,
        x_clean,
        show_intermediate_pics,
        IMG_WIDTH,
        IMG_HEIGHT,
        execute_postprocessing_model,
        execute_cnn_model,
        execute_UNet_model
):
    # print("betrete calc_mean_aias_value_of_one_image")

    # print("x_filmed: ")
    # plt.imshow(x_filmed)
    # plt.show()
    # print("x_clean: ")
    # plt.imshow(x_clean)
    # plt.show()

    fourier_handler = Fourier_Images(x_filmed, x_clean)

    img_filmed_r, img_filmed_g, img_filmed_b = fourier_handler.split_RGB_2_Grayscale(
        x_filmed)
    img_clean_r, img_clean_g, img_clean_b = fourier_handler.split_RGB_2_Grayscale(
        x_clean)

    img_filmed_complex_r = tf.signal.fft2d(img_filmed_r)
    img_filmed_complex_g = tf.signal.fft2d(img_filmed_g)
    img_filmed_complex_b = tf.signal.fft2d(img_filmed_b)

    img_clean_complex_r = tf.signal.fft2d(img_clean_r)
    img_clean_complex_g = tf.signal.fft2d(img_clean_g)
    img_clean_complex_b = tf.signal.fft2d(img_clean_b)

    differenzbild_fourier_px = Helper_Generate_Mask_From_Images.generate_mask_from_images(
        img_filmed_complex_r, img_filmed_complex_g, img_filmed_complex_b, img_clean_complex_r, img_clean_complex_g,  img_clean_complex_b
    )
    # plt.imsave(".\\tmp\\img_filmed_fourier_combined.png",
    #            img_filmed_fourier_combined, cmap="gray")
    # plt.imsave(".\\tmp\\img_clean_fourier_combined.png",
    #            img_clean_fourier_combined, cmap="gray")
    if show_intermediate_pics:
        plt.imshow(differenzbild_fourier_px, cmap="gray")
        plt.show()

    differenzbild_fourier_px = differenzbild_fourier_px.reshape(
        1, IMG_WIDTH, IMG_HEIGHT, 1)

    del(fourier_handler)

    # ----------- HIER STARTET U-NET-MODEL -----------

    u_net_output = execute_UNet_model(
        differenzbild_fourier_px, training=True)

    # ----------- ALPHA-BLENDING -----------

    # FORMEL: (1 - out) * clean + out * filmed
    # t2c -> transfered to complex

    if show_intermediate_pics:
        print("x_clean: ")
        plt.imshow(x_clean)
        plt.show()

        print("x_filmed: ")
        plt.imshow(x_filmed)
        plt.show()

    # u_net_output = np.zeros((IMG_WIDTH, IMG_HEIGHT))



    # if show_intermediate_pics:
    #     print("u_net_output test: ")
    #     plt.imshow(u_net_output, cmap="gray")
    #     plt.show()

    if show_intermediate_pics:
        print("u_net_output orig: ")
        plt.imshow(u_net_output, cmap="gray")
        plt.show()

    x_clean = x_clean.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3))
    x_filmed = x_filmed.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3))

    u_net_output = tf.reshape(
        u_net_output, shape=(IMG_WIDTH, IMG_HEIGHT))

    img_clean_complex_r = tf.reshape(
        img_clean_complex_r, (IMG_WIDTH, IMG_HEIGHT))
    img_clean_complex_g = tf.reshape(
        img_clean_complex_g, (IMG_WIDTH, IMG_HEIGHT))
    img_clean_complex_b = tf.reshape(
        img_clean_complex_b, (IMG_WIDTH, IMG_HEIGHT))
    img_filmed_complex_r = tf.reshape(
        img_filmed_complex_r, (IMG_WIDTH, IMG_HEIGHT))
    img_filmed_complex_g = tf.reshape(
        img_filmed_complex_g, (IMG_WIDTH, IMG_HEIGHT))
    img_filmed_complex_b = tf.reshape(
        img_filmed_complex_b, (IMG_WIDTH, IMG_HEIGHT))

    # ------------ postprocessing Model aufrufen --------

    image_processed_rgb, img_processed_r, u_net_output_complex = Helper_Do_Postprocessing_Stuff.do_image_blending_and_stack_grayscale_to_rgb(
        u_net_output,
        img_clean_complex_r,
        img_clean_complex_g,
        img_clean_complex_b,
        img_filmed_complex_r,
        img_filmed_complex_g,
        img_filmed_complex_b,
    )

    if show_intermediate_pics:
        print("image_processed_rgb: ")
        # print(np.array(image_processed_rgb).reshape(IMG_HEIGHT, IMG_WIDTH, 3))
        plt.imshow(np.array(image_processed_rgb).reshape(
            IMG_HEIGHT, IMG_WIDTH, 3))
        plt.show()

    # ----------- Feed multiple Buckets in CNN for predicting amount of alias -----------

    def create_multiple_buckets_to_feed_them_into_CNN(tensor):

        BATCH_SIZE_CNN = 1
        NUM_BOXES = 30
        CROP_SIZE = (60, 60)

        boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
        box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,
                                        maxval=BATCH_SIZE_CNN, dtype=tf.int32)

        output = tf.image.crop_and_resize(
            tensor, boxes, box_indices, CROP_SIZE)
        return output

    cnn_input = create_multiple_buckets_to_feed_them_into_CNN(
        image_processed_rgb)

    # print("cnn_input: ")
    # print(cnn_input)

    # ------- Make prediction --------

    y_pred = execute_cnn_model(cnn_input)

    y_pred = tf.math.reduce_mean(
        y_pred, keepdims=True
    )

    # print("Ende von calc_mean_aias_value_of_one_image:")
    # print("y_pred: ")
    # print(y_pred)

    return y_pred, differenzbild_fourier_px, u_net_output, image_processed_rgb
