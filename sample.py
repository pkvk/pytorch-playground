""" Manipulate data sample
"""

import os
import tempfile
from subprocess import check_output

from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance
import numpy as np
# import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from utee import selector


model_raw, ds_fetcher, is_imagenet = selector.select('mnist')
ds_val = ds_fetcher(batch_size=10, train=False, val=True)

def _sweep(ar, threshold=4):
    min1, max1 = -1, -1
    for i, v in enumerate(ar.transpose()):
        if min1 == -1 and v.max() < threshold:
            pass  # No information yet
        elif min1 == -1:
            min1 = i  # Information detected, set beginning
        elif v.max() > threshold:
            max1 = i
    return min1, max1

# Returns ROI box (left, upper, right, lower)
def find_roi(ar, constrain=False):
    # Looking for x values
    min1, max1 = _sweep(ar)

    max_w, max_h = ar.shape

    # Looking for y values
    min2, max2 = _sweep(ar.transpose())

    d = max(max1 - min1, max2 - min2)  # Square width
    cen_x, cen_y = (max1 + min1)/2, (max2 + min2)/2
    d2 = d / 2.0
    # print(d, cen_x, cen_y, d2)

    box = (int(round(cen_x - d2)), int(round(cen_y - d2)),
           int(round(cen_x + d2)), int(round(cen_y + d2)))

    if constrain:
        # Crop within the image boundaries
        return (max(0, box[0]), max(0, box[1]),
                min(max_w, box[2]), min(max_h, box[3]))
    else:
        return box

def convert(im):
    """ Converts an image to a 28x28 MNIST format.
        Returns an array of pixels
    """
    im = im.convert(mode="L")  # Convert to black and white

    # Detect image data, crop, and resize
    pix = np.array(im)
    # im = Image.fromarray(np.uint8(pix))
    box = find_roi(pix)

    im = im.crop(box)
    # im = ImageOps.invert(im)
    # im.show()

    im = im.resize((20, 20))
    # im.show()

    # Create a blank white image 28x28
    new = Image.new('L', (28, 28), 'white')
    new.paste(im, (4, 4))

    return new

def recognize(im):
    img = convert(im)

    x = torch.tensor(np.array(img))
    x = x.unsqueeze(0)
    x = x.type(torch.FloatTensor).cuda()
    x = x / 255.0
    # x = 1 - x / 255.0

    print(x.size())
    output = model_raw(x)
    _, predicted = torch.max(output.data, 1)
    print("output", output)
    print(predicted.item())

    r = str(list(output.cpu().squeeze().detach().numpy()))
    print("returning", r)
    return r


if __name__ == '__main__':
    filename = 'Data/sample3.png'
    print(recognize(Image.open(filename)))
