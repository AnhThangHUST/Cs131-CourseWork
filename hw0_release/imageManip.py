import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    pass
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    pass
    out = 0.5 * np.square(image)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    pass
    out = np.sum(image, axis=2) / 3
    ### END YOUR CODE

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    
    ### YOUR CODE HERE
    pass
    mapToNum = {"R":"0", "G":"1", "B":"2"}
    image[:,:,mapToNum[channel]] = 0
    ### END YOUR CODE
    out = image
    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    pass
    mapToNum = {"L":"0", "A":"1", "B":"2"}
    lab[:,:,mapToNum[channel]] = 0
    ### END YOUR CODE
    
    out = lab
    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    pass
    mapToNum = {'H': 0, 'S': 1, 'V': 2}
    hsv[:, :, mapToNum[channel]] = 0
    out = hsv
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    pass
    image1 = rgb_decomposition(image1, channel1)
    image1 = image1[:,0 : image1.shape[1]/2,: ]
    image2 = rgb_decomposition(image2, channel2)
    image2 = image2[:, image2.shape[1]/2 : ,: ]
    out[:, 0 : image1.shape[1]/2,: ] = image1
    out[:, image2.shape[1]/2 : ,: ] = image2
    ### END YOUR CODE

    return out
