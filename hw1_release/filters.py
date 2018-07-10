import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ## get origin of kernel
    delta_r = int(Hk/2)
    delta_h = int(Wk/2)
    for r_im in range (0,Hi):
        for c_im in range (0,Wi):
            for r_ker in range (0,Hk):
                for c_ker in range (0,Wk):
                    if (-1<r_im-r_ker+delta_r<Hi and -1<c_im-c_ker+delta_h<Wi):
                        out[r_im][c_im] += kernel[r_ker][c_ker]*image[r_im - r_ker + delta_r][c_im - c_ker+delta_h]

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    pass
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    delta_r = int(Hk/2)
    delta_h = int(Wk/2)
    image = zero_pad(image, delta_r, delta_h)
    for r_im in range(0,Hi):
        for c_im in range(0,Wi):
            out[r_im][c_im] = np.sum(kernel*image[r_im:r_im+Hk,c_im:c_im+Wk])
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    out = np.zeros_like(f)
    Hf,Wf = f.shape
    Hg,Wg = g.shape
    delta_r = int(Hg/2)
    delta_h = int(Wg/2)
    f = zero_pad(f, delta_r, delta_h)
    for r_im in range(0,Hf):
        for c_im in range(0,Wf):
            out[r_im][c_im] = np.sum(g*f[r_im:r_im+Hg,c_im:c_im+Wg])
    
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    g_mean = np.sum(g)/np.size(g)
    g = g - g_mean
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    g = normalize(g)
    out = np.zeros_like(f)
    Hf,Wf = f.shape
    Hg,Wg = g.shape
    delta_r = int(Hg/2)
    delta_h = int(Wg/2)
    f = zero_pad(f, delta_r, delta_h)
    for r_im in range(0,Hf):
        for c_im in range(0,Wf):
            # normalize image patch
            im_patch = f[r_im : r_im+Hg, c_im : c_im+Wg]
            nml_im_patch = normalize(im_patch)
            out[r_im][c_im] = np.sum(g * nml_im_patch)
    ### END YOUR CODE

    return out

def normalize(f):
    f_mean = np.sum(f)/np.size(f)
    f = (f-f_mean)/np.std(f)
    return f