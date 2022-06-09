import tensorflow as tf

# ppm: postprocessing model


def tf_inv_fourier_trans(img):
    # return tf.math.round(tf.math.real(tf.signal.ifft2d(img)))
    return tf.math.real(tf.signal.ifft2d(img))


def create_postprocessing_model():

    ppm_input_img_clean = tf.keras.Input(shape=(128, 128, 3), batch_size=1)
    ppm_input_img_filmed = tf.keras.Input(shape=(128, 128, 3), batch_size=1)
    ppm_input_unet_output = tf.keras.Input(shape=(128, 128), batch_size=1)

    ppm_input_img_clean_complex_r = tf.keras.Input(
        shape=(128, 128), batch_size=1, dtype=tf.complex64)
    ppm_input_img_clean_complex_g = tf.keras.Input(
        shape=(128, 128), batch_size=1, dtype=tf.complex64)
    ppm_input_img_clean_complex_b = tf.keras.Input(
        shape=(128, 128), batch_size=1, dtype=tf.complex64)

    ppm_input_img_filmed_complex_r = tf.keras.Input(
        shape=(128, 128), batch_size=1, dtype=tf.complex64)
    ppm_input_img_filmed_complex_g = tf.keras.Input(
        shape=(128, 128), batch_size=1, dtype=tf.complex64)
    ppm_input_img_filmed_complex_b = tf.keras.Input(
        shape=(128, 128), batch_size=1, dtype=tf.complex64)

    ones = tf.ones((128, 128))
    zeros = tf.zeros((128))
    ones_t2c = tf.complex(ones, zeros)
    u_net_output_t2c = tf.complex(ppm_input_unet_output, zeros)

    # FORMEL: (1 - out) * clean + out * filmed
    # out == mask
    # t2c -> transfered to complex

    # # NUR ZUM TESTEN:
    # u_net_output_t2c = tf.complex(ones, zeros)

    def soft_blending(clean, filmed, ones_t2c=ones_t2c, u_net_output_t2c=u_net_output_t2c):
        print("++++++++++++++++++++++++++++ Betrete Softblenging!!")
        zw1 = tf.math.subtract(ones_t2c, u_net_output_t2c)
        zw1 = tf.math.multiply(zw1, clean)

        zw2 = tf.multiply(u_net_output_t2c, filmed)
        return tf.math.add(zw1, zw2)

    img_processed_complex_fourier_r = soft_blending(
        ppm_input_img_clean_complex_r, ppm_input_img_filmed_complex_r)

    img_processed_complex_fourier_g = soft_blending(
        ppm_input_img_clean_complex_g, ppm_input_img_filmed_complex_g)

    img_processed_complex_fourier_b = soft_blending(
        ppm_input_img_clean_complex_b, ppm_input_img_filmed_complex_b)

    # # img_processed_px_fourier = tf.math.log(                     # zum anschauen
    # #     tf.math.abs(img_processed_complex_fourier_r))

    # ----------- INVERSE FOURIER TRANSFORMATION -----------

    img_processed_r = tf_inv_fourier_trans(
        img_processed_complex_fourier_r
    ) / 255
    # img_processed_r = tf.math.divide(img_processed_r, 255)

    img_processed_g = tf_inv_fourier_trans(
        img_processed_complex_fourier_g
    ) / 255
    # img_processed_g = tf.math.divide(img_processed_g, 255)

    img_processed_b = tf_inv_fourier_trans(
        img_processed_complex_fourier_b
    ) / 255
    # img_processed_b = tf.math.divide(img_processed_b, 255)

    # HIERHER
    # 3 Einzelkanäle zu einem RGB Bild
    # img_processed_r = tf.math.log(tf.math.abs(img_processed_r)) #/255
    # img_processed_g = tf.math.log(tf.math.abs(img_processed_g)) #/255
    # img_processed_b = tf.math.log(tf.math.abs(img_processed_b)) #/255

    img_processed_rgb = tf.stack(
        [img_processed_r, img_processed_g, img_processed_b], axis=-1)

    postprocess_model = tf.keras.Model(inputs=[
        ppm_input_img_clean,
        ppm_input_img_filmed,
        ppm_input_unet_output,
        ppm_input_img_clean_complex_r,
        ppm_input_img_clean_complex_g,
        ppm_input_img_clean_complex_b,
        ppm_input_img_filmed_complex_r,
        ppm_input_img_filmed_complex_g,
        ppm_input_img_filmed_complex_b
    ], outputs=[img_processed_rgb, img_processed_r, u_net_output_t2c], name="postprocessing_model")

    # postprocess_model.summary()
    return postprocess_model
