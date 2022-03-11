#! /usr/bin/python2.7

from __init__ import *

from PIL import Image

if __name__ == "__main__":
    imgs = [Image.open("pictures/image_{}.png".format(i)) for i in xrange(1, 5)]
    for i, img in enumerate(imgs):
        # img.show()
        print("#{}: img.size = {}".format(i, img.size))

    img01 = utils.concat_images_horizontal(imgs[0], imgs[1])
    img23 = utils.concat_images_horizontal(imgs[2], imgs[3])

    img0123 = utils.concat_images_vertical(img01, img23)

    img0123.show()
