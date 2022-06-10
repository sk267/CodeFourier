import numpy as np


def calc_white_amount_of_unet_out(img_in):
    print("############## betrete calc_white_amount_of_unet_out")
    mask = np.array(img_in)
    return np.mean(mask)
