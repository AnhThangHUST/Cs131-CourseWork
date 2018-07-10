import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    pass
    for r_im in range(0,Hi):
        for c_im in range(0, Wi):
            out[r_im][c_im] +=np.sum(kernel*padded[r_im:r_im+Hk,c_im:c_im+Wk])
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))
    k = int((size-1)/2)
    
    ### YOUR CODE HERE
    pass
    for i in range(0,size):
        for j in range(0,size):
            kernel[i][j] = np.exp(-((i-k)*(i-k)+(j-k)*(j-k))/(2*sigma*sigma)) * 1/(2*np.pi*sigma*sigma)
            pass
    ### END YOUR CODE
    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    pass
    #forward derivative
    derivative_hor = np.array([-0.5,0,0.5]).reshape(1,3)
    out = conv(img, derivative_hor)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    pass
    #forward derivative
    derivative_ver = np.array([-0.5,0,0.5]).reshape(3,1)
    out = conv(img, derivative_ver)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    pass
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx*Gx + Gy*Gy)
    theta = np.arctan2(Gx,Gy)
    theta = 180/np.pi *theta
    theta[np.where(theta<0)] += 360
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    minG = np.min(G)
    out = np.copy(G)
    G = np.pad(G,((1,1),(1,1)),'constant', constant_values = minG)
    #Slow Way
    for i in range(1,H+1):
        for j in range(1,W+1):
            k = theta[i-1][j-1]
            if (k==0 or k==180):
                if np.max([G[i][j], G[i][j+1],G[i][j-1]]) != G[i][j]:
                    out[i-1][j-1] = 0
            if (k==45 or k==225):
                if np.max([G[i][j], G[i-1][j+1],G[i+1][j-1]]) != G[i][j]:
                    out[i-1][j-1] = 0
            if (k==90 or k==270):
                if np.max([G[i][j], G[i-1][j],G[i+1][j]]) != G[i][j]:
                    out[i-1][j-1] = 0
            if (k==135 or k==315):
                if np.max([G[i][j], G[i+1][j+1],G[i-1][j-1]]) != G[i][j]:
                    out[i-1][j-1] = 0
    #Fast Way
    #tup1 = np.where((theta==0)|(theta==180))
    #tup2 = np.where((theta==90)|(theta==270))
    #tup3 = np.where((theta==45)|(theta==225))
    #tup4 = np.where((theta==135)|(theta==315))
    ### END YOUR CODE
    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    pass
    strong_edges = img>high
    weak_edges = (img < high) & (img > low)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)

    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    pass
    edges = np.copy(strong_edges)
    for x,y in indices:
        for neighbor in get_neighbors(y,x,H,W):
            if weak_edges[neighbor] >0:
                edges[neighbor] = weak_edges[neighbor]
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE
    pass
    img = conv(img,gaussian_kernel(kernel_size, sigma))
    G,theta = gradient(img)
    G = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(G,high,low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return accumulator, rhos, thetas
