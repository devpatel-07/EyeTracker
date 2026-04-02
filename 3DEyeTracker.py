import cv2
import random
import math
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import sys
import time
import matplotlib.pyplot as plt

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")


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
    spacing = int(len(all_contours)/25) 
    # Temporary array for result
    filtered_points = []
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
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
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  
        
        # Check if angle is oriented towards centroid
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
        cv2.ellipse(image, ellipse, color, 2)  
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
        return [0, 0]  
    
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
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  
    
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
        return ellipse_goodness  
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    #skew of the ellipse
    axes_lengths = ellipse[1]  
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

def compute_gaze_intersection(left_eye_center, left_gaze_dir, right_eye_center, right_gaze_dir):
    origin_delta = left_eye_center - right_eye_center
    
    dot_left_left = np.dot(left_gaze_dir, left_gaze_dir)   
    dot_left_right = np.dot(left_gaze_dir, right_gaze_dir)
    dot_right_right = np.dot(right_gaze_dir, right_gaze_dir) 
    
    dot_left_delta = np.dot(left_gaze_dir, origin_delta)
    dot_right_delta = np.dot(right_gaze_dir, origin_delta)
    
    denominator = dot_left_left * dot_right_right - dot_left_right * dot_left_right
    
    if abs(denominator) < 1e-6:
        return (left_eye_center + right_eye_center) / 2 + left_gaze_dir * 1000 

    dist_left = (dot_left_right * dot_right_delta - dot_right_right * dot_left_delta) / denominator
    dist_right = (dot_left_left * dot_right_delta - dot_left_right * dot_left_delta) / denominator
    
    closest_point_left = left_eye_center + dist_left * left_gaze_dir
    closest_point_right = right_eye_center + dist_right * right_gaze_dir
    
    gaze_point_3d = (closest_point_left + closest_point_right) / 2
    
    return gaze_point_3d

class EyeTracker:
    def __init__(self, name, camera_position_3d):
        self.name = name
        self.camera_position = np.array(camera_position_3d)
        
        self.ellipses = [[0, 0, 0]] * 10 
        self.counter = 0                 
        self.eye_centers = []            
        self.rays = []                   
        self.indexCounter = 0            
        self.arraySize = 1500

        self.prev_model_center_avg = (320, 240)
        self.max_observed_distance = 0  

    def eyecenter_estimation(self, frame):
        # Build rays for all ellipses
        current_rays = []
        for (cx, cy, angle_deg) in self.ellipses:
            a = np.deg2rad(angle_deg)
            dx, dy = -np.sin(a), np.cos(a)
            current_rays.append((cx, cy, dx, dy))

        # Compute intersections across ALL pairs in current frame
        intersections = []
        for i in range(len(current_rays)):
            for j in range(i + 1, len(current_rays)):
                x1, y1, ddx1, ddy1 = current_rays[i]
                x2, y2, ddx2, ddy2 = current_rays[j]

                v1 = np.array([ddx1, ddy1])
                v2 = np.array([ddx2, ddy2])
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

                if abs(cos_theta) > np.cos(np.deg2rad(2)):
                    continue

                A = np.array([[ddx1, -ddx2], [ddy1, -ddy2]])
                B = np.array([x2 - x1, y2 - y1])

                try:
                    t1, _ = np.linalg.solve(A, B)
                except np.linalg.LinAlgError:
                    continue

                intersectionX = x1 + t1 * ddx1
                intersectionY = y1 + t1 * ddy1
                intersections.append((intersectionX, intersectionY))

        if not intersections:
            return None

        avg_x = int(np.mean([pt[0] for pt in intersections]))
        avg_y = int(np.mean([pt[1] for pt in intersections]))

        est_center = (avg_x, avg_y)

        # Draw rays from each ellipse center to estimated eye center
        for (cx, cy, _, __) in current_rays:
            line = ((int(cx), int(cy)), est_center)
            if line not in self.rays:
                self.rays.append(line)

        if len(self.rays) > 10:
            self.rays = self.rays[-10:]

        for ellipse_center, intersection in self.rays:
            cv2.line(frame, ellipse_center, intersection, (255, 0, 255), 1)

        return est_center

    def compute_gaze_vector(self, x, y, center_x, center_y, screen_width=640, screen_height=480):
        viewport_width = screen_width
        viewport_height = screen_height
        fov_y_deg = 45.0
        aspect_ratio = viewport_width / viewport_height
        far_clip = 100.0

        camera_position = self.camera_position

        fov_y_rad = np.radians(fov_y_deg)
        half_height_far = np.tan(fov_y_rad / 2) * far_clip
        half_width_far = half_height_far * aspect_ratio

        ndc_x = (2.0 * x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * y) / viewport_height

        far_x = ndc_x * half_width_far
        far_y = ndc_y * half_height_far
        far_z = camera_position[2] - far_clip
        far_point = np.array([far_x, far_y, far_z])

        ray_origin = camera_position
        ray_direction = far_point - camera_position
        ray_direction /= np.linalg.norm(ray_direction)
        ray_direction = -ray_direction

        inner_radius = 1.0 / 1.05
        sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
        sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
        sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0]) + camera_position

        origin = ray_origin
        direction = -ray_direction
        L = origin - sphere_center

        a = np.dot(direction, direction)
        b = 2 * np.dot(direction, L)
        c = np.dot(L, L) - inner_radius**2

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
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

        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        target_direction = intersection_local / np.linalg.norm(intersection_local)

        circle_local_center = np.array([0.0, 0.0, inner_radius])
        circle_local_center /= np.linalg.norm(circle_local_center)

        rotation_axis = np.cross(circle_local_center, target_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm < 1e-6:
            return sphere_center, circle_local_center

        rotation_axis /= rotation_axis_norm
        dot = np.dot(circle_local_center, target_direction)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)

        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        t_ = 1 - c
        x_, y_, z_ = rotation_axis

        rotation_matrix = np.array([
            [t_*x_*x_ + c, t_*x_*y_ - s*z_, t_*x_*z_ + s*y_],
            [t_*x_*y_ + s*z_, t_*y_*y_ + c, t_*y_*z_ - s*x_],
            [t_*x_*z_ - s*y_, t_*y_*z_ + s*x_, t_*z_*z_ + c]
        ])

        gaze_local = np.array([0.0, 0.0, inner_radius])
        gaze_rotated = rotation_matrix @ gaze_local
        gaze_rotated /= np.linalg.norm(gaze_rotated)

        return sphere_center, gaze_rotated

    # Process frames for pupil detection
    def process_frames(self, thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point):
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] 
        final_contours = [] 
        goodness = 0 
        final_goodness = 0
        
        #initialize variables
        center_x, center_y = None, None

        #iterate through binary images and see which fits the ellipse best
        for i in range(1,4):
            # Dilate the binary image
            dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
            
            # Find contours
            contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

            if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
                current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], False)
                ellipse = cv2.fitEllipse(reduced_contours[0])
                center_x, center_y = map(int, ellipse[0]) 
                    
                #in total pixels, first element is pixel total, next is ratio
                total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, False)                 
                final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]

            if final_goodness > 0 and final_goodness > goodness: 
                goodness = final_goodness
                final_contours = reduced_contours

        final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
        
        final_rotated_rect = None
        model_center_average = (320, 240)

        if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0]) > 5:
            ellipse = cv2.fitEllipse(final_contours[0])
            final_rotated_rect = ellipse
            
            #Storing information from each pupil ellipse
            (c_x, c_y), _, ellipse_angle = ellipse
            self.ellipses[self.counter % 10] = [c_x, c_y, ellipse_angle]

            frame_height, frame_width = frame.shape[0:2]
            boundary_center = (frame_width//2 - 100, frame_height//2 - 100)
            boundary_radius = int(frame_height * 0.5)

            #checks for if there has been at least 3 frames
            if self.counter >= 2:
                #finds eye center estimate from 3 most recent frames
                eye_center = self.eyecenter_estimation(frame)
                
                #updates list of past 1500 eye center estimates
                if eye_center is not None:
                    d_squared = (eye_center[0] - boundary_center[0])**2 + (eye_center[1] - boundary_center[1])**2
                    if d_squared < boundary_radius**2:
                        if (len(self.eye_centers) < self.arraySize):
                            self.eye_centers.append(eye_center)
                        else:
                            self.eye_centers[self.indexCounter % self.arraySize] = eye_center
                            self.indexCounter += 1

                #display average eye center estimate
                if len(self.eye_centers) > 0:
                    x_estimate = sum([x[0] for x in self.eye_centers if x]) // len(self.eye_centers)
                    y_estimate = sum([y[1] for y in self.eye_centers if y]) // len(self.eye_centers)
                    model_center_average = (x_estimate, y_estimate)
            
            #track frames
            self.counter += 1

        if model_center_average[0] == 320:
            model_center_average = self.prev_model_center_avg
        if model_center_average[0] != 0:
            self.prev_model_center_avg = model_center_average
        
        # Example safety check
        if center_x is None or center_y is None or model_center_average[0] is None or model_center_average[1] is None:
            return None, None

        # Draw reference lines/ellipses
        cv2.circle(frame, model_center_average, int(202), (255, 50, 50), 2)  # Draw eye sphere (circle)
        cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)  # Draw eye center

        if final_rotated_rect is not None:
            cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)  # # Draw line from eye center to ellipse center
            cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2) #draw final ellipse on image

        cv2.putText(frame, self.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"{self.name} Feed", frame)

        return self.compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1])

    # Finds the pupil in an individual frame and returns the center point
    def process_frame(self, frame):
        # Crop and resize frame
        frame = crop_to_aspect_ratio(frame)
        
        # FLIP REMOVED: frame = cv2.flip(frame, 0) was causing the upside-down issue

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
        return self.process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point)


def run_dual_tracking(src_left, src_right=None, mirror_mode=False):
    cap_left = cv2.VideoCapture(src_left)
    
    if not mirror_mode:
        cap_right = cv2.VideoCapture(src_right)

    tracker_left = EyeTracker(name="Left Eye", camera_position_3d=[-0.1, 0.0, 3.0])
    tracker_right = EyeTracker(name="Right Eye", camera_position_3d=[0.1, 0.0, 3.0])

    # --- SETUP MATPLOTLIB 3D ---
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Real-Time 3D Gaze Vectors")
    plt.show(block=False)  # <--- ADD THIS LINE

    while True:
        ret_l, frame_l = cap_left.read()
        
        if not ret_l:
            print("Video feed ended or disconnected.")
            break

        if mirror_mode:
            # We horizontally mirror the video to act as the right eye
            frame_r = cv2.flip(frame_l, 1) 
            ret_r = True
        else:
            ret_r, frame_r = cap_right.read()
            if not ret_r:
                print("Right video feed ended.")
                break

        left_data = tracker_left.process_frame(frame_l.copy())
        right_data = tracker_right.process_frame(frame_r.copy())

        # Unpack the tuples first
        lCenter, lDirection = left_data
        rCenter, rDirection = right_data

        # Safely check if the data exists using 'is not None'
        if lCenter is not None and rCenter is not None:
            intersection_3d = compute_gaze_intersection(lCenter, lDirection, rCenter, rDirection)
            
            # --- UPDATE MATPLOTLIB 3D ---
            ax.clear()
            
            # Vectors from eye centers to intersection
            A = intersection_3d - lCenter
            B = intersection_3d - rCenter

            # Draw vectors
            ax.quiver(*lCenter, *A, color='r', linewidth=2, label='Left Gaze')
            ax.quiver(*rCenter, *B, color='b', linewidth=2, label='Right Gaze')

            # Draw points
            ax.scatter(*lCenter, color='red', s=50, label='Left Eye (Pa)')
            ax.scatter(*rCenter, color='blue', s=50, label='Right Eye (Pb)')
            ax.scatter(*intersection_3d, color='green', s=50, label='Intersection (Pi)')

            # Labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Fixed limits (prevents jumping). Scaled down for eye tracking units.
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-5, 4)
            
            ax.legend()
            
            fig.canvas.draw()
            fig.canvas.flush_events()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): cv2.waitKey(0)

    cap_left.release()
    if not mirror_mode:
        cap_right.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)


def dual_selection_gui():
    cameras = detect_cameras()
    root = tk.Tk()
    root.title("Dual Eye Tracker Configuration")
    
    tk.Label(root, text="Left Eye Source:", font=("Arial", 10, "bold")).pack(pady=5)
    selected_left = tk.StringVar(value=str(cameras[0]) if cameras else "0")
    ttk.Combobox(root, textvariable=selected_left, values=[str(c) for c in cameras]).pack()

    tk.Label(root, text="Right Eye Source:", font=("Arial", 10, "bold")).pack(pady=5)
    selected_right = tk.StringVar(value=str(cameras[1]) if len(cameras)>1 else "1")
    ttk.Combobox(root, textvariable=selected_right, values=[str(c) for c in cameras]).pack()

    def start_cameras():
        src_l = int(selected_left.get())
        src_r = int(selected_right.get())
        root.destroy()
        run_dual_tracking(src_l, src_r, mirror_mode=False)

    def start_videos():
        src_l = filedialog.askopenfilename(title="Select LEFT Video", filetypes=[("Video", "*.mp4;*.avi")])
        if not src_l: return
        src_r = filedialog.askopenfilename(title="Select RIGHT Video", filetypes=[("Video", "*.mp4;*.avi")])
        if not src_r: return
        root.destroy()
        run_dual_tracking(src_l, src_r, mirror_mode=False)
        
    def start_mirrored_video():
        src = filedialog.askopenfilename(title="Select Single Video to Mirror", filetypes=[("Video", "*.mp4;*.avi")])
        if not src: return
        root.destroy()
        run_dual_tracking(src_left=src, mirror_mode=True)

    tk.Button(root, text="Start 2 Webcams", command=start_cameras).pack(pady=10)
    tk.Button(root, text="Start 2 Videos", command=start_videos).pack(pady=5)
    tk.Button(root, text="Start 1 Video (Mirrored)", command=start_mirrored_video, bg="lightblue").pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    dual_selection_gui()