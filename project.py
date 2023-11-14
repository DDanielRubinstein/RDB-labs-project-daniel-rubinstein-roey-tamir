import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import random as r 

# Creating a cache for storing the MSE values
mse_cache = {}

# Load the data
def import_csv(filename):
    """
    Import data from a CSV file, convert it to a NumPy array, and perform angle-based sorting and index shifting.
    
    Input:
        filename (str): The path to the CSV file containing points in a specific format.
        
    Output:
        points (numpy.ndarray): An array containing the sorted and shifted points.
        
    The function does the following:
    1. Reads the CSV and extracts points as tuples.
    2. Converts the list of tuples to a NumPy array.
    3. Computes the mean point in the XY plane.
    4. Sorts the points based on their angles with respect to the mean point.
    5. Shifts the indices of the points based on their proximity to the mean x-coordinate.
    """
        
    points = []
    data = pd.read_csv(filename)
    for i, row in data.iterrows():
        point = (row['X'].split()[0], row['X'].split()[1], row['X'].split()[2])
        points.append(point)
        
    points = np.array(points).astype(float)

    # "ANGLE-SORTING"
    # Compute the mean of the points in the XY plane
    mean_point = np.mean(points[:, :2], axis=0)

    # Compute the angle for each point with respect to the mean point
    angles = [math.atan2(p[1] - mean_point[1], p[0] - mean_point[0]) for p in points[:, :2]]
    
    # Sort the points based on these angles
    sorted_indices = np.argsort(angles)
    points = points[sorted_indices]

    # SHIFTING
    # Compute the mean of the x-coordinates
    mean_x = np.mean(points[:, 0])

    # Find the point closest to the mean x-coordinate
    start_index = np.argmin(np.abs(points[:, 0] - mean_x))

    # Create an array of indices
    indices = np.arange(len(points))

    # Shift the indices
    shifted_indices = (indices + start_index - 1) % len(points)

    # Reorder the points based on the shifted indices
    points = points[shifted_indices]
    
    return points

# Synthetic data generation
# NOTE: this function is not a required part of the project, and is simply a mean to demonstrate the result of running the algorithm - as was used throughtout the semester in our reports.
"""
def generate_3d_rectangle(num_points=1000):
    # Define the corners of the bottom rectangle (floor)
    lower_left = np.array([1, 1, 0])
    upper_right = np.array([5, 3, 0])
    
    # Define the corners of the top rectangle (ceiling)
    height = 4  # Arbitrary height
    lower_left_top = lower_left + np.array([0, 0, height])
    upper_right_top = upper_right + np.array([0, 0, height])

    # Generate points for the bottom and top rectangles (similar to 2D version)
    def generate_rectangle_points(ll, ur, num_points):
        edges = [
            np.linspace(ll, [ur[0], ll[1], ll[2]], num_points // 4, endpoint=False),
            np.linspace([ur[0], ll[1], ll[2]], ur, num_points // 4, endpoint=False),
            np.linspace(ur, [ll[0], ur[1], ur[2]], num_points // 4, endpoint=False),
            np.linspace([ll[0], ur[1], ur[2]], ll, num_points // 4, endpoint=True)
        ]
        return np.concatenate(edges)

    bottom_points = generate_rectangle_points(lower_left, upper_right, num_points // 2)
    top_points = generate_rectangle_points(lower_left_top, upper_right_top, num_points // 2)

    # Add some noise to the points
    points = np.concatenate([bottom_points, top_points])

    # Assuming a threshold to differentiate between bottom and upper points
    threshold_z = 2  # Example threshold value

    # Split points into bottom and upper based on the z-coordinate
    bott = points[points[:, 2] < threshold_z]
    upp = points[points[:, 2] >= threshold_z]

    # Generate noise for bottom and upper points
    noise_bottom = np.random.uniform(-0.75, 4, bott.shape[0])
    noise_upper = np.random.uniform(0.75, 4, upp.shape[0])

    # Add the noise to the respective points
    bott[:, 2] += noise_bottom
    upp[:, 2] += noise_upper
    points = np.concatenate([bott, upp])
    points = points + np.random.normal(scale=0.05, size=points.shape)
    # "ANGLE MANNER"
    # Compute the mean of the points in the XY plane
    mean_point = np.mean(points[:, :2], axis=0)

    # Compute the angle for each point with respect to the mean point
    angles = [math.atan2(p[1] - mean_point[1], p[0] - mean_point[0]) for p in points[:, :2]]
    
    # Sort the points based on these angles
    sorted_indices = np.argsort(angles)
    points = points[sorted_indices]
    # SHIFTING
    # Compute the mean of the x-coordinates
    mean_x = np.mean(points[:, 0])

    # Find the point closest to the mean x-coordinate
    start_index = np.argmin(np.abs(points[:, 0] - mean_x))

    # Create an array of indices
    indices = np.arange(len(points))

    # Shift the indices
    shifted_indices = (indices + start_index - 1) % len(points)

    # Reorder the points based on the shifted indices
    points = points[shifted_indices]
    
    return points
"""

# Euclidean distance squered
def euc_dist(a, b):
    """
    Calculate the squared Euclidean distance between two points in a 2D space.
    
    Input:
        a (tuple): The coordinates (x, y) of the first point.
        b (tuple): The coordinates (x, y) of the second point.
        
    Output:
        float: The squared Euclidean distance between points a and b.
    """

    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def fit_parabola(x, y):
    """
    Fit a parabola (quadratic polynomial) to the given data points using least squares linear regression.
    
    Parameters:
    x (numpy.ndarray): The x-coordinates of the data points.
    y (numpy.ndarray): The y-coordinates of the data points.
    
    Returns:
    numpy.ndarray: Coefficients [a, b, c] of the fitted parabola y = ax^2 + bx + c.
    """

    # Construct the design matrix with columns [x^2, x, 1]
    X = np.vstack([x**2, x, np.ones(len(x))]).T

    # Solve the least squares problem
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]

    return np.array([a, b, c])

def generate_parab(u, v, distances):
    """
    Generate a parabolic function based on a given range of indices and distances.
    
    Input:
        start (int): The starting index for the subset of distances.
        end (int): The ending index for the subset of distances.
        distances (numpy.ndarray): An array containing distances.
        
    Output:
        parab (numpy.poly1d): A parabolic function fitted to the subset of distances.
        
    The function also uses a cache to store previously calculated parabolas to avoid redundant calculations.
    """

    # Calculate x and y based on start and end indices
    x = np.array(range(u, v))
    y = distances[u:v]
    return np.poly1d(fit_parabola(x, y))

def MSE(u, v, distances):
    """
    Calculate the Mean Squared Error (MSE) between the observed distances and a parabolic fit.
    
    Input:
        start (int): The starting index for the subset of distances.
        end (int): The ending index for the subset of distances.
        distances (numpy.ndarray): An array containing distances.
        parab (numpy.poly1d): A parabolic function to compare against.
        
    Output:
        mse_value (float): The calculated MSE value.
        
    The function uses a cache to store previously calculated MSE values to avoid redundant calculations.
    """
        
    # Use u and v as the unique key for caching
    key = (u, v)
    
    # Check if the result is already in the cache
    if key in mse_cache:
        return mse_cache[key]
    
    # Calculate x and y based on start and end indices
    x = np.array(range(u, v))
    y = distances[u:v]
    
    # If not in cache, calculate the MSE and store in cache
    parab = generate_parab(u, v, distances)
    mse_value = np.sum((y - parab(x))**2)
    mse_cache[key] = mse_value
    
    return mse_value

# Find the path of length k
def segment(n, k, distances):
    """
    Perform dynamic programming to segment a set of distances into 'k' segments, each represented by a parabola.
    
    Input:
        n (int): The total number of points.
        k (int): The number of segments.
        distances (numpy.ndarray): An array containing distances.
        
    Output:
        path (list): A list of indices that represent the best segmentation.
        
    The function utilizes dynamic programming to efficiently find the best segmentation based on MSE.
    """
        
    D = np.full((n+1, k+1), np.inf)
    P = np.full((n+1, k+1), -1)

    D[0, 0] = 0

    # The dynammic programming is done efficiently using the calculations of previous length
    for length in range(1, k + 1):
        # Iterate over each possible ending point of the segment
        for v in range(length, n + 1):
            min_dist = np.inf
            min_index = -1
            
            # Iterate over each possible starting point of the segment
            for u in range(v - length + 1):
                if v - u < 3: # Skip if not enough points to fit a parabola
                    continue

                # Generate parabola (if didn't already) for the segment and calculate its MSE (weight)
                mse = MSE(u, v, distances)
                
                # always keep the best
                if D[u, length - 1] + mse < min_dist:
                    min_dist = D[u, length - 1] + mse
                    min_index = u

            # Update phase in the DP    
            D[v, length] = min_dist
            P[v, length] = min_index

    # Reconstruction phase
    path = []
    current = P[n,k]
    for i in range(k, 0, -1):
        path.append(current - 1)
        current = P[current, i]
    path.reverse()
    return path

def ransac_z_fit(floorCeil, points, iterations=300, threshold=0.5):
    """
    Perform modified-RANSAC to find the most frequent z-coordinate within a given threshold with priority to high (ceil) and low (floor) points.
    
    Input:
        floorCeil: -1 if supposed to estimate the floor, 1 if supposed to estimate the ceil.
        points (numpy.ndarray): An array containing 3D points.
        iterations (int): The number of iterations for RANSAC. Default is 300.
        threshold (float): The distance threshold for inliers. Default is 0.5.
        
    Output:
        best_z (float): The z-coordinate with the most inliers (also considering priorities).
    """

    # We can get rid of at least half of the points (the higher or the lower half given that we are looking for the floor or the ceiling, respectively). 
    
    med = np.median(points[:, 2])
    bott = points[points[:, 2] < med]
    upp = points[points[:, 2] >= med]

    best_z = None
    best_inliers = -np.inf
    for _ in range(iterations):
        # Randomly sample a point
        if(floorCeil == 1):
            sample = points[np.random.choice(upp.shape[0], 1, replace=False)]
        else:
            sample = points[np.random.choice(bott.shape[0], 1, replace=False)]

        sample_z = floorCeil * sample[0, 2]
        
        # Count inliers within the threshold
        inliers = np.sum(np.abs(floorCeil * points[:, 2] - sample_z) < threshold) * sample_z # give weight according to height
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_z = sample_z
    
    return best_z

def plot_parab(range_start, range_end, distances):
    """
    Generate x and y values to plot a parabolic function based on a given range and distances.
    
    Input:
        range_start (int): The starting index for the range of distances.
        range_end (int): The ending index for the range of distances.
        distances (numpy.ndarray): An array containing distances.
        
    Output:
        x (numpy.ndarray): The x values for plotting.
        Y (numpy.ndarray): The y values for plotting, based on the parabolic fit.
    """
        
    x = np.arange(range_start, range_end + 1)
    parab = generate_parab(range_start, range_end, distances)
    Y = parab(x)
    return x, Y

def bounding_box_3d(points):
    """
    Compute and visualize a 3D bounding box for a set of points.
    
    Input:
        points (numpy.ndarray): An array containing 3D points.
        
    Output:
        rectangle_3d (list): A list of tuples representing the 3D bounding box corners.
        
    The function performs the following tasks:
    1. Estimates a 2D bounding box in the XY plane using a dynamic programming approach.
    2. Visualizes the 2D bounding box and the segmented parabolas.
    3. Estimates the floor and ceiling heights using RANSAC.
    4. Constructs and returns the 3D bounding box.
    """

    n = len(points)
    k = 4

    # Calculate the mean of the XY coordinates
    c_mean = points[:, :2].mean(axis=0)
    # Compute distances from each point to the mean point
    distances = [euc_dist(points[i, :2], c_mean) for i in range(n)]
    # Segment the distances into k segments using the dynamic programming approach
    path = segment(n, k, distances)
    path_points = points[path]

    # Define the corners of the 2D rectangle in XY plane
    sorted_indices = np.argsort(path_points[:, 0])
    path_x = path_points[sorted_indices]
    sorted_indices = np.argsort(path_points[:, 1])
    path_y = path_points[sorted_indices]
    avg = lambda x, y: (x+y)/2
    lower_left = (avg(path_x[0][0], path_x[1][0]), avg(path_y[0][1], path_y[1][1]))
    lower_right = (avg(path_x[2][0], path_x[3][0]), avg(path_y[0][1], path_y[1][1]))
    upper_right = (avg(path_x[2][0], path_x[3][0]), avg(path_y[2][1], path_y[3][1]))
    upper_left = (avg(path_x[0][0], path_x[1][0]), avg(path_y[2][1], path_y[3][1]))

    rectangle = [lower_left, lower_right, upper_right, upper_left]

    # Visualization of 2D bounding box estimation
    plt.scatter(points[:, 0], points[:, 1], s=5)
    # Red edges (estimation from the parabolas themselves)
    plt.plot(points[path, 0], points[path, 1], color='red', linestyle = '--')
    plt.plot([points[path[-1], 0], points[path[0], 0]], [points[path[-1], 1], points[path[0], 1]], color='red', linestyle='--')
    # Green rectangle (returned rectangle - "mean" rectangle derived from the red shape)
    rectangle_points = np.array(rectangle + [lower_left]) 
    plt.plot(rectangle_points[:, 0], rectangle_points[:, 1], color='green')
    plt.show()

    # Plotting the signal function
    plt.scatter(np.linspace(1, n, n), distances)

    # Plotting the parabola for the first segment
    x, y = plot_parab(0, path[0], distances)
    plt.plot(x, y)

    # Plotting the parabolas for the segments defined by the path
    for i in range(len(path) - 1):
        x, y = plot_parab(path[i], path[i + 1], distances)
        plt.plot(x, y)

    # Plotting the last parabola
    x, y = plot_parab(path[-1], len(distances) - 1, distances)
    plt.plot(x, y)
    plt.show()

    # RANSAC-based estimation of floor and ceiling heights
    floor_z = ransac_z_fit(-1, points[points[:, 2] < np.median(points[:, 2])])
    ceiling_z = ransac_z_fit(1, points[points[:, 2] > np.median(points[:, 2])])

    # Construct 3D bounding box
    rectangle_3d = [
        (lower_left[0], lower_left[1], floor_z),
        (lower_right[0], lower_right[1], floor_z),
        (upper_right[0], upper_right[1], floor_z),
        (upper_left[0], upper_left[1], floor_z),
        (lower_left[0], lower_left[1], ceiling_z),
        (lower_right[0], lower_right[1], ceiling_z),
        (upper_right[0], upper_right[1], ceiling_z),
        (upper_left[0], upper_left[1], ceiling_z)
    ]

    return rectangle_3d


# NOTE: End of auxiliary functions, start of execution.

# Please note that we left an option to test the algorithm on synthetic data, 
# to do so, replace the import function with the commented example below.
points_3d = import_csv("map.csv") # generate_3d_rectangle(amount_of_points)
estimated_bounding_box = bounding_box_3d(points_3d)


# Visualisation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', marker='o', s=40, label="Points")
for i in range(4):
    ax.plot([estimated_bounding_box[i][0], estimated_bounding_box[i+4][0]], 
            [estimated_bounding_box[i][1], estimated_bounding_box[i+4][1]], 
            [estimated_bounding_box[i][2], estimated_bounding_box[i+4][2]], c='green')
for i in range(4):
    ax.plot([estimated_bounding_box[i][0], estimated_bounding_box[(i+1)%4][0]], 
            [estimated_bounding_box[i][1], estimated_bounding_box[(i+1)%4][1]], 
            [estimated_bounding_box[i][2], estimated_bounding_box[(i+1)%4][2]], c='green')
for i in range(4):
    ax.plot([estimated_bounding_box[i+4][0], estimated_bounding_box[(i+1)%4+4][0]], 
            [estimated_bounding_box[i+4][1], estimated_bounding_box[(i+1)%4+4][1]], 
            [estimated_bounding_box[i+4][2], estimated_bounding_box[(i+1)%4+4][2]], c='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Estimated 3D room")
ax.legend()
plt.show()