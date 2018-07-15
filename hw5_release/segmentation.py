import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    prev_assignments = np.zeros(N)
    for n in range(num_iters):
        ### YOUR CODE HERE
        pass
        #2. Tinh ra xem trung tam nao la gan nhat vs diem do
        for i in range(N):
            diff = centers-features[i]
            distances = np.linalg.norm(diff, axis = 1)
            assignments[i] = np.argmin(distances)
        #3. Tinh toan lai trung tam cua cac cum
        for j in range(k):
            points_belong_to_cluster = features[np.where(assignments==j)]
            # centers luc sau co the khong phai la 1 diem  cua feature nua
            if len(points_belong_to_cluster)!=0:
                centers[j] = np.mean(points_belong_to_cluster, axis=0)
        #4. Neu assignments khong co j thay doi thi return khoi vong lap
        if (np.array_equal(prev_assignments, assignments)):
            break
        prev_assignments = np.copy(assignments)
        ### END YOUR CODE
    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    # ta implement tuong tu ham tren vi da fast nhat co the roi, thay moi np.where
    prev_assignments = np.zeros(N)
    for n in range(num_iters):
        ### YOUR CODE HERE
        #centers : K*D, feature: N*D
        square_center = np.square(centers)
        mask_center = np.sum(square_center, axis = 1).reshape(k,1)
        mask_center = np.repeat(mask_center, N, axis = 1)
        square_feature = np.square(features)
        mask_feature = np.sum(square_feature, axis = 1).reshape(1,N)
        mask_feature = np.repeat(mask_feature, k, axis = 0)
        all_distances = mask_center + mask_feature - 2 * np.dot(centers,features.T)
        all_distances[all_distances==0] = np.max(all_distances)
        assignments = np.argmin(all_distances, axis = 0)
        for j in range(k):
            points_belong_to_cluster = features[np.where(assignments==j)]
            # centers luc sau co the khong phai la 1 diem  cua feature nua
            if len(points_belong_to_cluster)!=0:
                centers[j] = np.mean(points_belong_to_cluster, axis=0)
        if (np.array_equal(prev_assignments, assignments)):
            break
        prev_assignments = np.copy(assignments)
    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N
    distances = np.zeros_like(assignments)
    while n_clusters > k:
        ### YOUR CODE HERE
        # Luu y la ta can tinh khoang cach giua cac cap trung tam(centroid) center de xem cac cluster gan nhau nhat
        all_distances = np.zeros((n_clusters,n_clusters))
        for i in range(n_clusters):
            #1. tinh khoang cach giua cac cluster voi nhau
            all_distances[i] = np.linalg.norm(centers - centers[i], axis = 1)
            # gia su ta da co bang all distance N*N chi co distance giua 1 diem vs chinh no =0 thi
        all_distances[all_distances==0] = np.max(all_distances)

        #2. hop cac cluster gan nhau nhat
        min_index = np.argmin(all_distances)
        #index_cluster1< index_cluster2
        index_cluster1 = np.argmin(all_distances) // n_clusters
        index_cluster2 = np.argmin(all_distances) % n_clusters 
        assignments[assignments==index_cluster2] = index_cluster1
        assignments[assignments>index_cluster2] -= 1
        # tinh lai centroid cua cac cluster and number of cluster
        centers[index_cluster1] = np.mean(features[assignments==index_cluster1], axis = 0)
        centers = np.delete(centers, index_cluster2, axis = 0)
        n_clusters -= 1
        ### END YOUR CODE
        
    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    pass
    features = img.reshape((H*W,C))
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    pass
    features[:,:C] = color.reshape((-1,C))
    features[:,C] = np.mgrid[:H, :W][0].reshape((H*W))
    features[:,C+1] = np.mgrid[:H, :W][1].reshape((H*W))
    features -= np.mean(features, axis=0)
    features /= np.std(features, axis=0)
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features
    

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
