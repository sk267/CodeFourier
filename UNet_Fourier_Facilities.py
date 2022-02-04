import tensorflow as tf
import numpy as np
import statistics
import cv2
from skimage.exposure import adjust_gamma
import matplotlib.pyplot as plt


class UNet_Label_Gen():
    def __init__(self):
        self.model = tf.keras.models.load_model(
            "./../Code/models/single_rgb_image")

    def get_mean_alias_value_for_whole_picture(self, img, skip_every_nth=1, n_channels=3):
        self.BUCKET_SIZE = 60
        self.steps_horizontal = img.shape[0] / self.BUCKET_SIZE
        self.steps_vertikal = img.shape[1] / self.BUCKET_SIZE
        alias_values = []
        for y in range(int(self.steps_vertikal)):
            if y % skip_every_nth != 0:
                continue
            for x in range(int(self.steps_horizontal)):
                if x % skip_every_nth != 0:
                    continue
                bucket = img[y*self.BUCKET_SIZE:(y+1)*self.BUCKET_SIZE, x *
                             self.BUCKET_SIZE:(x+1)*self.BUCKET_SIZE]
                bucket = np.asarray(bucket).reshape(
                    1, self.BUCKET_SIZE, self.BUCKET_SIZE, n_channels) / 255
                prediction = self.model.predict(bucket)
                alias_values.append(prediction[0][0])
        return(1 - statistics.mean(alias_values))


class Fourier_Images():
    def __init__(self, img_filmed, img_aligned):
        self.image_filmed = img_filmed
        self.img_aligned = img_aligned
        self.RGB_WEIGHTS = [0.299, 0.587, 0.114]
        self.IMG_HEIGHT = img_filmed.shape[0]
        self.IMG_WIDTH = img_filmed.shape[1]

        self.img_filmed_fourier_combined = 0
        self.img_clean_fourier_combined = 0

        self.img_filmed_fourier_r = 0
        self.img_filmed_fourier_g = 0
        self.img_filmed_fourier_b = 0

        self.img_clean_fourier_r = 0
        self.img_clean_fourier_g = 0
        self.img_clean_fourier_b = 0

        self.fourier_mask = 0

    def f2pd(self, fourier_array):
        # transoforms an fourier image into pixel domain in order
        # to display it
        return (np.log(abs(fourier_array)))

    def normalize_0_1(self, img):
        max = np.max(img)
        min = np.min(img)
        img = (img - min) / (max - min)
        return(img)

    def split_RGB_2_Grayscale(self, img):
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        return (img_r, img_g, img_b)

    def grayscale_2_Fourier(self, img):
        return(np.fft.fftshift(np.fft.fft2(img)))

    def inverse_fourier(self, fourier_array):
        return(abs(np.fft.ifft2(fourier_array)))

    def combine_3_fourier_to_one_grayscale(self, im1, im2, im3):
        img_combined = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH))
        im1 = self.f2pd(im1) * self.RGB_WEIGHTS[0]
        im2 = self.f2pd(im2) * self.RGB_WEIGHTS[1]
        im3 = self.f2pd(im3) * self.RGB_WEIGHTS[2]
        img_combined = im1 + im2 + im3
        return (img_combined)

    def generate_mask_from_images(self):
        # split to separate grayscale images
        img_filmed_r, img_filmed_g, img_filmed_b = self.split_RGB_2_Grayscale(
            self.image_filmed)
        img_clean_r, img_clean_g, img_clean_b = self.split_RGB_2_Grayscale(
            self.img_aligned)

        # transform to fourier
        self.img_filmed_fourier_r = self.grayscale_2_Fourier(img_filmed_r)
        self.img_filmed_fourier_g = self.grayscale_2_Fourier(img_filmed_g)
        self.img_filmed_fourier_b = self.grayscale_2_Fourier(img_filmed_b)

        self.img_clean_fourier_r = self.grayscale_2_Fourier(img_clean_r)
        self.img_clean_fourier_g = self.grayscale_2_Fourier(img_clean_g)
        self.img_clean_fourier_b = self.grayscale_2_Fourier(img_clean_b)

        self.img_filmed_fourier_combined = self.combine_3_fourier_to_one_grayscale(
            self.img_filmed_fourier_r, self.img_filmed_fourier_g, self.img_filmed_fourier_b
        )

        self.img_clean_fourier_combined = self.combine_3_fourier_to_one_grayscale(
            self.img_clean_fourier_r, self.img_clean_fourier_g, self.img_clean_fourier_b
        )

        DILATE_KERNEL_SIZE = (10, 10)

        # A - B (Filmed - Clean)
        self.fourier_mask = self.img_filmed_fourier_combined - \
            self.img_clean_fourier_combined
        # plt.imshow(fourier_mask, cmap="gray")
        # plt.show()

        # normalize image
        self.fourier_mask = self.normalize_0_1(self.fourier_mask)

        # Blur Image
        self.fourier_mask = cv2.blur(self.fourier_mask, (6, 6))

        # Enlarge Contrast
        self.fourier_mask = adjust_gamma(self.fourier_mask, 15)

        # Threshold Image
        self.fourier_mask = np.where(self.fourier_mask > 0.0005, 1, 0)

        # Dilate Image
        kernel = np.ones(DILATE_KERNEL_SIZE, np.float)
        self.fourier_mask = cv2.dilate(
            self.fourier_mask.astype("uint8"), kernel)

        return self.fourier_mask

    def replace_masked_sections_and_return_resulting_img(self):

        # replace masked sections
        img_filmed_fourier_processed_r = self.img_filmed_fourier_r.copy()
        img_filmed_fourier_processed_g = self.img_filmed_fourier_g.copy()
        img_filmed_fourier_processed_b = self.img_filmed_fourier_b.copy()

        for y in range(self.IMG_HEIGHT):
            for x in range(self.IMG_WIDTH):
                if self.fourier_mask[y][x] == 1:
                    img_filmed_fourier_processed_r[y][x] = self.img_clean_fourier_r[y][x]
                    img_filmed_fourier_processed_g[y][x] = self.img_clean_fourier_g[y][x]
                    img_filmed_fourier_processed_b[y][x] = self.img_clean_fourier_b[y][x]

        # Inverse fourier transformation
        inv_img_processed_r = self.inverse_fourier(
            img_filmed_fourier_processed_r)
        inv_img_processed_g = self.inverse_fourier(
            img_filmed_fourier_processed_g)
        inv_img_processed_b = self.inverse_fourier(
            img_filmed_fourier_processed_b
        )

        # Concatenate them to one image
        inv_img_processed_rgb = np.zeros((self.IMG_WIDTH, self.IMG_HEIGHT, 3))
        inv_img_processed_rgb[:, :, 0] = inv_img_processed_r[:, :]
        inv_img_processed_rgb[:, :, 1] = inv_img_processed_g[:, :]
        inv_img_processed_rgb[:, :, 2] = inv_img_processed_b[:, :]
        inv_img_processed_rgb = inv_img_processed_rgb.astype(int)

        return inv_img_processed_rgb
