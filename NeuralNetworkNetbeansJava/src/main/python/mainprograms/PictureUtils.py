#! /usr/bin/python2.7
# created: 06.05.2016

from __init__ import *

# maybe not needed self implementation!
def shrink_picture():
    pass

import Image

img = Image.new( 'RGB', (255,255), "black") # create a new black image
pixels = img.load() # create the pixel map

for i in range(img.size[0]):    # for every pixel:
    for j in range(img.size[1]):
        pixels[i, j] = (i, j, 0) # set the colour accordingly

img.show()

img = img.resize((100, 100), PIL.Image.ANTIALIAS)

img.show()
