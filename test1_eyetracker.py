import cv2
import random
import math
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import sys
import time
import threading

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

try:
    import matplotlib
    matplotlib.use('TkAgg')  # must be set before any pyplot import; TkAgg is
                             # compatible with Tkinter which is already running
    import gaze_viz_3d
    VIZ_3D_AVAILABLE = True
except ImportError:
    VIZ_3D_AVAILABLE = False
    print("gaze_viz_3d module not found. 3D gaze visualization will be disabled.")

ray_lines = [] 
model_centers = []
max_rays = 100
prev_model_center_avg = (320,240)
max_observed_distance = 0  # Initialize adaptive radius
cached_threshold_idx = None   # 0/1/2 once a good threshold level is found per eye
_suppress_internal_windows = False  # set True in dual mode to hide internal imshows

# Function to detect available cameras
def detect_cameras(max_cams=10):
    available_cameras = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Crop the image to maintain a specific aspect ratio (width:height) before resizing.
def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

# Apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# Finds a square area of dark pixels in the image
def get_darkest_area(image):
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None

    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            current_sum = 0
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point

# Mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)

def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

# Returns the largest contour that is not extremely long or tall
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    return [largest_contour] if largest_contour is not None else []

#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

# Process frames for pupil detection
def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window, state=None):
    # ---------------------------------------------------------------------------
    # Resolve state: use the passed-in per-eye dict when available (dual mode),
    # otherwise fall back to the legacy module-level globals (single-eye mode).
    # ---------------------------------------------------------------------------
    if state is not None:
        _ray_lines            = state['ray_lines']
        _model_centers        = state['model_centers']
        _stored_intersections = state['stored_intersections']
        _max_observed_distance = state['max_observed_distance']
        _prev_model_center_avg = state['prev_model_center_avg']
        _cached_threshold_idx  = state['cached_threshold_idx']
    else:
        global ray_lines, model_centers, stored_intersections
        global max_observed_distance, prev_model_center_avg, cached_threshold_idx
        _ray_lines            = ray_lines
        _model_centers        = model_centers
        _stored_intersections = stored_intersections
        _max_observed_distance = max_observed_distance
        _prev_model_center_avg = prev_model_center_avg
        _cached_threshold_idx  = cached_threshold_idx

    direction = None
    center_x = None
    center_y = None

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_image = cv2.dilate(thresholded_image_medium, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["relaxed", "medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = [] #holds an array of the best contour points from the fitting process
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    final_goodness = 0
    
    #iterate through binary images and see which fits the ellipse best.
    # Caching: if a threshold level worked last frame, try it first and break
    # immediately on success (1 iteration instead of 3). If it misses, fall
    # through to the remaining indices so the cache can be re-established.
    if _cached_threshold_idx is not None:
        others = [j for j in range(1, 4) if j != _cached_threshold_idx + 1]
        search_order = [_cached_threshold_idx + 1] + others
    else:
        search_order = list(range(1, 4))

    best_thresh_i = None

    for i in search_order:
        # Dilate the binary image
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        #initialize variables
        center_x, center_y = None, None

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            center_x, center_y = map(int, ellipse[0]) 
            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image
            best_thresh_i = i
            if _cached_threshold_idx is not None:
                # Cached index worked — no need to evaluate the other two
                break

    # Persist the winning threshold level so next frame starts here directly
    if best_thresh_i is not None:
        _cached_threshold_idx = best_thresh_i - 1

    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    final_rotated_rect = None

    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0] > 5):
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse

        # Store the new ray in the list
        _ray_lines.append(final_rotated_rect)
        # **Prune rays if list exceeds max_rays**
        if len(_ray_lines) > max_rays:
            num_to_remove = len(_ray_lines) - max_rays
            _ray_lines = _ray_lines[num_to_remove:]  # Keep only the last `max_rays` elements

    model_center_average = (320,240)

    model_center = compute_average_intersection(frame, _ray_lines, _stored_intersections, 5, 1500, 5)
    if model_center is not None:
        model_center_average = update_and_average_point(_model_centers, model_center, 200)

    if model_center_average[0] == 320:
        model_center_average = _prev_model_center_avg
    if model_center_average[0] != 0:
        _prev_model_center_avg = model_center_average
    
    # Example safety check
    if center_x is None or center_y is None or model_center_average[0] is None or model_center_average[1] is None:
        # Write state back before early return
        if state is not None:
            state['ray_lines']             = _ray_lines
            state['model_centers']         = _model_centers
            state['stored_intersections']  = _stored_intersections
            state['max_observed_distance'] = _max_observed_distance
            state['prev_model_center_avg'] = _prev_model_center_avg
            state['cached_threshold_idx']  = _cached_threshold_idx
        else:
            ray_lines             = _ray_lines
            model_centers         = _model_centers
            stored_intersections  = _stored_intersections
            max_observed_distance = _max_observed_distance
            prev_model_center_avg = _prev_model_center_avg
            cached_threshold_idx  = _cached_threshold_idx
        return final_rotated_rect, None

    # Calculate the distance only if model_centers has at least 100 values
    if len(_model_centers) >= 100 and center_x is not None:
        distance = math.sqrt((center_x - model_center_average[0]) ** 2 + (center_y - model_center_average[1]) ** 2)
        if distance > _max_observed_distance:
            _max_observed_distance = distance
            
    _max_observed_distance = 202

    # Draw reference lines/ellipses
    cv2.circle(frame, model_center_average, int(_max_observed_distance), (255, 50, 50), 2)  # Draw eye sphere (circle)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)  # Draw eye center

    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)  # # Draw line from eye center to ellipse center
        
    cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2) #draw final ellipse on image

    # Calculate the extended endpoint of gaze line
    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        # Compute the vector from model_center_average to center_x, center_y
        dx = center_x - model_center_average[0]
        dy = center_y - model_center_average[1]

        # Scale the vector by 1.2x
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)

        # Draw the extended gaze line
        cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3) 

    if render_cv_window:
        cv2.imshow("Best Thresholded Image Contours on Frame", frame)

    if GL_SPHERE_AVAILABLE:
        gl_image = gl_sphere.update_sphere_rotation(center_x, center_y, model_center_average[0], model_center_average[1])

    # Call the function
    center, direction = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1])

    if center is not None and direction is not None:
        origin_text = f"Origin: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        dir_text    = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"

        # Set bottom-left corner for drawing text
        text_origin = (12, frame.shape[0] - 38)  # 40 pixels from bottom
        text_dir    = (12, frame.shape[0] - 13)  # 15 pixels from bottom
        text_origin2 = (10, frame.shape[0] - 40)  # 40 pixels from bottom
        text_dir2    = (10, frame.shape[0] - 15)  # 15 pixels from bottom

        # Draw shadow text on the frame
        cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        # Draw text on the frame
        cv2.putText(frame, origin_text, text_origin2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, dir_text, text_dir2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if center is not None and direction is not None:
        print(f"Sphere Center:   ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"Gaze Direction:  ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})")
    else:
        print("No valid intersection found.")

    if not _suppress_internal_windows:
        cv2.imshow("Frame with Ellipse and Rays", frame)

    if GL_SPHERE_AVAILABLE:
        if gl_image is not None:
            blended = cv2.addWeighted(frame, 0.6, gl_image, 0.4, 0)
            cv2.imshow("Eye Tracker + Sphere", blended)

    # ---------------------------------------------------------------------------
    # Write local state variables back to wherever they came from
    # ---------------------------------------------------------------------------
    if state is not None:
        state['ray_lines']             = _ray_lines
        state['model_centers']         = _model_centers
        state['stored_intersections']  = _stored_intersections
        state['max_observed_distance'] = _max_observed_distance
        state['prev_model_center_avg'] = _prev_model_center_avg
        state['cached_threshold_idx']  = _cached_threshold_idx
    else:
        ray_lines             = _ray_lines
        model_centers         = _model_centers
        stored_intersections  = _stored_intersections
        max_observed_distance = _max_observed_distance
        prev_model_center_avg = _prev_model_center_avg
        cached_threshold_idx  = _cached_threshold_idx

    return final_rotated_rect, direction

def update_and_average_point(point_list, new_point, N):
    """
    Adds a new point to the list, keeps only the last N points, 
    and returns the average of those points.
    
    Parameters:
    - point_list: Global list storing past points [(x1, y1), (x2, y2), ...]
    - new_point: Tuple (x, y) representing the new point to add.
    - N: Maximum number of points to keep in the list.
    
    Returns:
    - (avg_x, avg_y): The average point as a tuple of integers.
    - None if the list is empty.
    """
    point_list.append(new_point)  # Add new point

    if len(point_list) > N:
        point_list.pop(0)  # Remove the oldest point to maintain size N

    if not point_list:
        return None  # No points available

    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))

    return (avg_x, avg_y)

def draw_orthogonal_ray(image, ellipse, length=100, color=(0, 255, 0), thickness=1):
    """
    Draws a ray passing through the center of an ellipse orthogonally to its major axis.
    
    Parameters:
    - image: The OpenCV image to draw on.
    - ellipse: A tuple ((cx, cy), (major_axis, minor_axis), angle) representing the fitted ellipse.
    - length: Length of the ray to draw on each side of the ellipse center.
    - color: Color of the line in BGR format (default: green).
    - thickness: Thickness of the line (default: 2).
    """

    (cx, cy), (major_axis, minor_axis), angle = ellipse
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Compute the normal vector at the ellipse center (perpendicular to surface)
    normal_dx = (minor_axis / 2) * np.cos(angle_rad)  # Minor axis component
    normal_dy = (minor_axis / 2) * np.sin(angle_rad)

    # Compute start and end points of the orthogonal ray
    pt1 = (int(cx - length * normal_dx / (minor_axis / 2)), int(cy - length * normal_dy / (minor_axis / 2)))
    pt2 = (int(cx + length * normal_dx / (minor_axis / 2)), int(cy + length * normal_dy / (minor_axis / 2)))

    # Draw the ray
    cv2.line(image, pt1, pt2, color, thickness)

    return image 

stored_intersections = []  # Stores all past intersections

def compute_average_intersection(frame, ray_lines_arg, stored_intersections_arg, N, M, spacing):
    """
    Selects N random lines from the list, highlights them in red on the frame,
    computes their intersections, stores them, and prunes stored intersections when exceeding M.

    Parameters:
    - frame: The OpenCV frame to draw on.
    - ray_lines_arg: List of ellipse tuples ((cx, cy), (major_axis, minor_axis), angle).
    - stored_intersections_arg: Per-eye list of stored intersections (passed in directly).
    - N: Number of random lines to select for intersection calculation.
    - M: Maximum number of stored intersections before pruning.

    Returns:
    - (avg_x, avg_y): Average intersection point of selected lines.
    """
    if len(ray_lines_arg) < 2 or N < 2:
        return (0, 0)  # Need at least 2 lines to find intersections

    # Get frame dimensions dynamically
    height, width = frame.shape[:2]

    # Select N unique random lines
    selected_lines = random.sample(ray_lines_arg, min(N, len(ray_lines_arg)))

    intersections = []

    # Compute intersections for each pair of selected lines
    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]

        angle1 = line1[2]  # Extract angle from ellipse tuple
        angle2 = line2[2]  # Extract angle from ellipse tuple

        if abs(angle1 - angle2) >= 2:  # Ensure lines differ by at least 2 degrees
            intersection = find_line_intersection(line1, line2)
            
            # Ensure the intersection is within the frame bounds before adding
            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                intersections.append(intersection)
                stored_intersections_arg.append(intersection)  # Store valid intersections

    # Prune intersections if stored list exceeds M
    if len(stored_intersections_arg) > M:
        stored_intersections_arg[:] = prune_intersections(stored_intersections_arg, M)

    if not intersections:
        return None  # No valid intersections found

    # Compute the average intersection point
    avg_x = np.mean([pt[0] for pt in stored_intersections_arg])
    avg_y = np.mean([pt[1] for pt in stored_intersections_arg])

    return (int(avg_x), int(avg_y))

#Removes the oldest intersections to ensure only the last M intersections remain.
def prune_intersections(intersections, maximum_intersections):

    if len(intersections) <= maximum_intersections:
        return intersections  # No need to prune if within the limit

    # Keep only the last M intersections
    pruned_intersections = intersections[-maximum_intersections:]

    return pruned_intersections

def find_line_intersection(ellipse1, ellipse2):
    """
    Computes the intersection of two lines that are orthogonal to the surface of given ellipses.
    
    Parameters:
    - ellipse1, ellipse2: Ellipse tuples ((cx, cy), (major_axis, minor_axis), angle).
    
    Returns:
    - (x, y): Intersection point of the two lines, or None if parallel.
    """

    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2

    # Convert angles to radians
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)

    # Compute direction vectors for the two lines
    dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)

    # Line equations in parametric form:
    # (x1, y1) + t1 * (dx1, dy1) = (x2, y2) + t2 * (dx2, dy2)
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])

    # Solve for t1, t2 using linear algebra (if the determinant is nonzero)
    if np.linalg.det(A) == 0:
        return None  # Lines are parallel and do not intersect

    t1, t2 = np.linalg.solve(A, B)

    # Compute intersection point
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1

    return (int(intersection_x), int(intersection_y))

def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    """Compute 3D gaze direction from pupil and sphere center screen coordinates.
    Returns:
        sphere_center (np.ndarray): 3D position of the sphere center in world space
        gaze_direction (np.ndarray): Normalized 3D direction vector from sphere center
    """

    # Get viewport dimensions
    viewport_width = screen_width
    viewport_height = screen_height

    # Define camera and projection settings
    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0

    # Camera position is fixed at z = 3
    camera_position = np.array([0.0, 0.0, 3.0])

    # Compute size of far plane in world units
    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    # Convert screen (x, y) to normalized device coordinates [-1, 1]
    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height

    # Project pupil center to far plane coordinates in world space
    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    # Compute ray direction from camera to far plane point
    ray_origin = camera_position
    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    # Sphere radius and center offset
    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    # Compute intersection with sphere
    origin = ray_origin
    direction = -ray_direction
    L = origin - sphere_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        # Compute the closest point to the sphere (tangent point approximation)
        t = -np.dot(direction, L) / np.dot(direction, direction)
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        target_direction = intersection_local / np.linalg.norm(intersection_local)
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        t = None
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        if t is None:
            return None, None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    t = None
    if t1 > 0 and t2 > 0:
        t = min(t1, t2)
    elif t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    if t is None:
        return None, None

    # Final intersection point
    intersection_point = origin + t * direction

    # Convert to local space relative to sphere center
    intersection_local = intersection_point - sphere_center
    target_direction = intersection_local / np.linalg.norm(intersection_local)

    # Local green ring direction
    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    # Compute rotation to align local +Z to target
    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        return sphere_center, circle_local_center

    rotation_axis /= rotation_axis_norm
    dot = np.dot(circle_local_center, target_direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)

    # Rotation matrix from axis-angle
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t_ = 1 - c
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_*x_*x_ + c, t_*x_*y_ - s*z_, t_*x_*z_ + s*y_],
        [t_*x_*y_ + s*z_, t_*y_*y_ + c, t_*y_*z_ - s*x_],
        [t_*x_*z_ - s*y_, t_*y_*z_ + s*x_, t_*z_*z_ + c]
    ])

    # Rotate +Z vector to get gaze direction
    gaze_local = np.array([0.0, 0.0, inner_radius])
    gaze_rotated = rotation_matrix @ gaze_local
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    # Write to file (overwrite every frame)
    file_path = "gaze_vector.txt"

    def is_file_available(path):
        try:
            with open(path, "a"):
                return True
        except IOError:
            return False

    if is_file_available(file_path):
        try:
            with open(file_path, "w") as f:
                all_values = np.concatenate((sphere_center, gaze_rotated))
                csv_line = ",".join(f"{v:.6f}" for v in all_values)
                f.write(csv_line + "\n")
        except Exception as e:
            print("Write error:", e)
    else:
        print("File is currently in use. Skipping write.")

    return sphere_center, gaze_rotated

# Finds the pupil in an individual frame and returns the center point
def process_frame(frame):

    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)

    #find the darkest point
    darkest_point = get_darkest_area(frame)

    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    
    #take the three images thresholded at different levels and process them
    final_rotated_rect, _ = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)
    
    return final_rotated_rect


# Process video from the selected camera
def process_camera():
    global selected_camera
    cam_index = int(selected_camera.get())

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 0)
        process_frame(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# Process a selected video file
def process_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])

    if not video_path:
        return  # User canceled selection

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    

# GUI for selecting camera or video (single-source, kept for reference)
def selection_gui():
    global selected_camera
    cameras = detect_cameras()

    root = tk.Tk()
    root.title("Select Input Source")
    tk.Label(root, text="Orlosky Eye Tracker 3D", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(root, text="Select Camera:").pack(pady=5)

    selected_camera = tk.StringVar()
    selected_camera.set(str(cameras[0]) if cameras else "No cameras found")

    camera_dropdown = ttk.Combobox(root, textvariable=selected_camera,
                                   values=[str(cam) for cam in cameras])
    camera_dropdown.pack(pady=5)

    tk.Button(root, text="Start Camera",
              command=lambda: [root.destroy(), process_camera()]).pack(pady=5)
    tk.Button(root, text="Browse Video",
              command=lambda: [root.destroy(), process_video()]).pack(pady=5)

    if GL_SPHERE_AVAILABLE:
        app = gl_sphere.start_gl_window()

    root.mainloop()


# ---------------------------------------------------------------------------
# Per-eye isolated state containers
# ---------------------------------------------------------------------------

def _make_eye_state():
    """Create a fresh per-eye state dict."""
    return {
        'ray_lines':             [],
        'model_centers':         [],
        'stored_intersections':  [],
        'max_observed_distance': 0,
        'prev_model_center_avg': (320, 240),
        'cached_threshold_idx':  None,
    }


# ---------------------------------------------------------------------------
# Threaded per-eye worker — no global lock, truly parallel
# ---------------------------------------------------------------------------

def _thread_process_eye(cap, eye_state, results, eye_key, mirror_frame=None):
    """Read one frame from cap (or use mirror_frame), run the full detection
    pipeline with the eye's own isolated state dict, and store results.

    Because process_frames() now reads/writes only the passed-in state dict
    instead of module-level globals, both eye threads can run concurrently
    with no locking whatsoever.
    """
    frame_key = eye_key + '_frame'
    dir_key   = eye_key + '_dir'
    ret_key   = eye_key + '_ret'

    results[frame_key] = None
    results[dir_key]   = None
    results[ret_key]   = False

    # --- Frame capture ---
    if mirror_frame is not None:
        frame = cv2.flip(mirror_frame, 1)
        ret   = True
    else:
        ret, frame = cap.read()

    if not ret or frame is None:
        return

    results[ret_key] = True

    # --- Pre-processing (identical to process_frame, but passes eye_state) ---
    frame_disp = crop_to_aspect_ratio(frame.copy())
    darkest_pt = get_darkest_area(frame_disp)
    if darkest_pt is None:
        return

    gray = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY)
    dpv  = gray[darkest_pt[1], darkest_pt[0]]

    th_s = mask_outside_square(apply_binary_threshold(gray, dpv,  5), darkest_pt, 250)
    th_m = mask_outside_square(apply_binary_threshold(gray, dpv, 15), darkest_pt, 250)
    th_r = mask_outside_square(apply_binary_threshold(gray, dpv, 25), darkest_pt, 250)

    # --- Detection: isolated state, no globals touched ---
    _, direction = process_frames(
        th_s, th_m, th_r, frame_disp, gray, darkest_pt, False, False,
        state=eye_state)

    results[frame_key] = frame_disp
    results[dir_key]   = direction


# ---------------------------------------------------------------------------
# Dual tracking main loop
# ---------------------------------------------------------------------------

def run_dual_tracking(src_left, src_right=None, mirror_mode=False):
    """Open two video sources and process both eyes concurrently each frame."""
    cap_l = cv2.VideoCapture(src_left)
    cap_r = cv2.VideoCapture(src_right if (src_right and not mirror_mode) else src_left)

    if not cap_l.isOpened():
        print(f"Error: Could not open left source: {src_left}")
        return
    if not cap_r.isOpened():
        print(f"Error: Could not open right source: {src_right}")
        return

    # Fresh isolated state for each eye
    eye_state_left  = _make_eye_state()
    eye_state_right = _make_eye_state()

    global _suppress_internal_windows
    _suppress_internal_windows = True

    cv2.namedWindow("Left Eye - Gaze",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Eye - Gaze", cv2.WINDOW_NORMAL)

    if VIZ_3D_AVAILABLE:
        gaze_viz_3d.start()

    viz_update_interval = 3
    frame_idx = 0

    while True:
        results = {}

        if mirror_mode:
            # Read the single source once and hand the same raw frame to both threads
            ret_pre, frame_pre = cap_l.read()
            mirror_src = frame_pre if ret_pre else None
        else:
            mirror_src = None

        t_l = threading.Thread(
            target=_thread_process_eye,
            args=(cap_l, eye_state_left,  results, 'l'),
            kwargs={'mirror_frame': mirror_src if mirror_mode else None},
            daemon=True,
        )
        t_r = threading.Thread(
            target=_thread_process_eye,
            args=(cap_r, eye_state_right, results, 'r'),
            kwargs={'mirror_frame': mirror_src if mirror_mode else None},
            daemon=True,
        )

        t_l.start()
        t_r.start()
        t_l.join()
        t_r.join()

        if not results.get('l_ret') and not results.get('r_ret'):
            break

        # Display (main thread only — required on most platforms)
        frame_l = results.get('l_frame')
        frame_r = results.get('r_frame')

        if frame_l is not None:
            cv2.imshow("Left Eye - Gaze",  frame_l)
        if frame_r is not None:
            cv2.imshow("Right Eye - Gaze", frame_r)

        # Optional 3-D gaze visualisation
        l_dir = results.get('l_dir')
        r_dir = results.get('r_dir')
        if (VIZ_3D_AVAILABLE
                and l_dir is not None
                and r_dir is not None
                and frame_idx % viz_update_interval == 0):
            intersection = gaze_viz_3d.compute_gaze_intersection(
                gaze_viz_3d.LEFT_EYE_POS,  l_dir,
                gaze_viz_3d.RIGHT_EYE_POS, r_dir,
            )
            gaze_viz_3d.update(l_dir, r_dir, intersection)

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    if VIZ_3D_AVAILABLE:
        gaze_viz_3d.stop()


# ---------------------------------------------------------------------------
# Dual-source selection GUI
# ---------------------------------------------------------------------------

def dual_selection_gui():
    cameras = detect_cameras()
    root = tk.Tk()
    root.title("Dual Eye Tracker Configuration")

    tk.Label(root, text="Left Eye Source:",
             font=("Arial", 10, "bold")).pack(pady=5)
    sel_l = tk.StringVar(value=str(cameras[0]) if cameras else "0")
    ttk.Combobox(root, textvariable=sel_l,
                 values=[str(c) for c in cameras]).pack()

    tk.Label(root, text="Right Eye Source:",
             font=("Arial", 10, "bold")).pack(pady=5)
    sel_r = tk.StringVar(value=str(cameras[1]) if len(cameras) > 1 else "1")
    ttk.Combobox(root, textvariable=sel_r,
                 values=[str(c) for c in cameras]).pack()

    def start_streams():
        src_l = "http://10.159.65.65:8080?action=stream"
        src_r = "http://10.159.65.65:8081?action=stream"
        root.destroy()
        run_dual_tracking(src_l, src_r, mirror_mode=False)

    def start_videos():
        src_l = filedialog.askopenfilename(title="Select LEFT Video",
                                           filetypes=[("Video", "*.mp4")])
        if not src_l:
            return
        src_r = filedialog.askopenfilename(title="Select RIGHT Video",
                                           filetypes=[("Video", "*.mp4")])
        if not src_r:
            return
        root.destroy()
        run_dual_tracking(src_l, src_r, mirror_mode=False)

    def start_mirrored():
        src = filedialog.askopenfilename(title="Select Single Video to Mirror",
                                         filetypes=[("Video", "*.mp4")])
        if not src:
            return
        root.destroy()
        run_dual_tracking(src_left=src, mirror_mode=True)

    tk.Button(root, text="Start 2 Streams",
              command=start_streams).pack(pady=10)
    tk.Button(root, text="Start 2 Videos",
              command=start_videos).pack(pady=5)
    tk.Button(root, text="Start 1 Video (Mirrored)",
              command=start_mirrored, bg="lightblue").pack(pady=10)

    root.mainloop()


# Run GUI to select camera or video
if __name__ == "__main__":
    dual_selection_gui()