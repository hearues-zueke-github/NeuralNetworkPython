#! /usr/bin/python2.7

from __init__ import *

from PIL import Image

if __name__ == "__main__":
    file_name1 = "networks/binary_adder_5_bits_145/sqrt_mean_squared_error.png"
    file_name2 = "networks/binary_adder_5_bits_144/sqrt_mean_squared_error.png"
    file_name3 = "networks/binary_adder_5_bits_147/sqrt_mean_squared_error.png"
    file_name4 = "networks/binary_adder_5_bits_146/sqrt_mean_squared_error.png"

    img0 = Image.open(file_name1)
    img1 = Image.open(file_name2)
    img2 = Image.open(file_name3)
    img3 = Image.open(file_name4)

    img01 = utils.concat_images_horizontal(img0, img1)
    img23 = utils.concat_images_horizontal(img2, img3)

    img0123 = utils.concat_images_vertical(img01, img23)

    # img0123.show()
    img0123.save("networks/concat_sqrt_mean_square_error.png")
