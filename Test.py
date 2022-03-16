from cProfile import label
from unittest import result
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from UNet_Fourier_Facilities import UNet_Label_Gen, Fourier_Images
from skimage.io import imread
import os

from UNet_Fourier_Facilities_Fake import UNet_Label_Gen_Fake

print(5+3j)

# img = imread("./Tests/orig.jpg")

# FILMED_PATH = "D:\\Main\\MA_PROGR\\Data\\Train\\UNet_Train\\water\\filmed_small"
# CLEAN_ALIGNED_PATH = "D:\\Main\\MA_PROGR\\Data\\Train\\UNet_Train\\water\\clean_aligned_small"

# img_filmed = imread(f"{FILMED_PATH}\{os.listdir(FILMED_PATH)[0]}")
# img_aligned = imread(
#     f"{CLEAN_ALIGNED_PATH}\{os.listdir(CLEAN_ALIGNED_PATH)[0]}")


# labelGenFake = UNet_Label_Gen_Fake()
# fourier_handler = Fourier_Images(img_filmed, img_aligned)


# for i in range(20):
#     a = labelGenFake.get_decreasing_alias_value()
#     print(a)
# mask = fourier_handler.generate_mask_from_images()
# processed_image = fourier_handler.replace_masked_sections_and_return_resulting_img()

# plt.imshow(mask, cmap="gray")
# plt.show()
# plt.imshow(processed_image)
# plt.show()

# print(labelGen.get_mean_alias_value_for_whole_picture(processed_image))
