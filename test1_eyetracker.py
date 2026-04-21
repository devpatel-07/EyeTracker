# Before cv2: OpenCV may load Qt; matplotlib must match (not TkAgg vs Qt).
import matplotlib
matplotlib.use("QtAgg")

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
import gaze_world

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

try:
    import gaze_viz_3d
    VIZ_3D_AVAILABLE = True
except ImportError:
    VIZ_3D_AVAILABLE = False
    print("gaze_viz_3d module not found. 3D gaze visualization will be disabled.")

try:
    import serial as _pyserial
    SERIAL_AVAILABLE = True
except ImportError:
    _pyserial = None
    SERIAL_AVAILABLE = False

ray_lines = []
model_centers = []
max_rays = 100
prev_model_center_avg = (320, 240)
max_observed_distance = 0
cached_threshold_idx = None
_suppress_internal_windows = False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def detect_cameras(max_cams=10):
    available_cameras = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))


def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = int(darkestPixelValue) + int(addedThreshold)
    threshold = max(0, min(255, threshold))
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image


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
                    current_sum += int(gray[y + dy, x + dx])
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point


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

    all_contours = np.concatenate(contours[0], axis=0)
    spacing = int(len(all_contours) / 25)
    filtered_points = []
    centroid = np.mean(all_contours, axis=0)
    cos_threshold = np.cos(np.radians(60))

    for i in range(0, len(all_contours), 1):
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]

        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        vec_to_centroid = centroid - current_point

        if np.dot(vec_to_centroid, (vec1 + vec2) / 2) >= cos_threshold:
            filtered_points.append(current_point)

    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))


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


def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, color, 2)
        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image


def check_contour_pixels(contour, image_shape, debug_mode_on):
    if len(contour) < 5:
        return [0, 0]

    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)

    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)

    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10)
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4)

    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)

    absolute_pixel_total_thick = np.sum(overlap_thick > 0)
    total_border_pixels = np.sum(contour_mask > 0)
    ratio_under_ellipse = np.sum(overlap_thin > 0) / total_border_pixels if total_border_pixels > 0 else 0

    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]


def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0, 0, 0]
    if len(contour) < 5:
        print("length of contour was 0")
        return 0

    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, (255), -1)

    ellipse_area = np.sum(mask == 255)
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))

    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness

    ellipse_goodness[0] = covered_pixels / ellipse_area
    ellipse_goodness[2] = min(ellipse[1][1] / ellipse[1][0], ellipse[1][0] / ellipse[1][1])

    return ellipse_goodness


# ---------------------------------------------------------------------------
# Main per-frame processing
# ---------------------------------------------------------------------------

stored_intersections = []


def process_frames(thresholded_image_strict, thresholded_image_medium,
                   thresholded_image_relaxed, frame, gray_frame,
                   darkest_point, debug_mode_on, render_cv_window, state=None):
    # --- Resolve state: per-eye dict (dual mode) or module globals (single mode) ---
    if state is not None:
        _ray_lines             = state['ray_lines']
        _model_centers         = state['model_centers']
        _stored_intersections  = state['stored_intersections']
        _max_observed_distance = state['max_observed_distance']
        _prev_model_center_avg = state['prev_model_center_avg']
        _cached_threshold_idx  = state['cached_threshold_idx']
    else:
        global ray_lines, model_centers, stored_intersections
        global max_observed_distance, prev_model_center_avg, cached_threshold_idx
        _ray_lines             = ray_lines
        _model_centers         = model_centers
        _stored_intersections  = stored_intersections
        _max_observed_distance = max_observed_distance
        _prev_model_center_avg = prev_model_center_avg
        _cached_threshold_idx  = cached_threshold_idx

    global max_rays

    direction = None
    center_x = None
    center_y = None

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict]
    name_array = ["relaxed", "medium", "strict"]
    final_contours = []
    goodness = 0
    final_goodness = 0
    best_thresh_i = None
    final_image = image_array[0]

    gray_copies = [gray_frame.copy(), gray_frame.copy(), gray_frame.copy()]

    # Build search order: try cached threshold first if available
    if _cached_threshold_idx is not None:
        others = [j for j in range(3) if j != _cached_threshold_idx]
        search_order = [_cached_threshold_idx] + others
    else:
        search_order = list(range(3))

    for idx in search_order:
        dilated_image = cv2.dilate(image_array[idx], kernel, iterations=2)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        center_x, center_y = None, None

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            center_x, center_y = map(int, ellipse[0])

            if debug_mode_on:
                cv2.imshow(name_array[idx] + " threshold", gray_copies[idx])

            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)
            cv2.ellipse(gray_copies[idx], ellipse, (255, 0, 0), 2)

            final_goodness = current_goodness[0] * total_pixels[0] * total_pixels[0] * total_pixels[1]

        if final_goodness > 0 and final_goodness > goodness:
            goodness = final_goodness
            final_contours = reduced_contours
            final_image = dilated_image
            best_thresh_i = idx
            # If cached index worked, skip the rest
            if _cached_threshold_idx is not None and idx == _cached_threshold_idx:
                break

    # Update cache
    if best_thresh_i is not None:
        _cached_threshold_idx = best_thresh_i

    # Optimize contours and fit final ellipse
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]

    final_rotated_rect = None
    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0]) > 5:
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse

        _ray_lines.append(final_rotated_rect)
        if len(_ray_lines) > max_rays:
            _ray_lines = _ray_lines[-max_rays:]

    # Compute model center (eye center) from ray intersections
    model_center_average = (320, 240)
    model_center = compute_average_intersection(frame, _ray_lines, _stored_intersections, 5, 1500, 5)
    if model_center is not None:
        model_center_average = update_and_average_point(_model_centers, model_center, 200)

    if model_center_average[0] == 320:
        model_center_average = _prev_model_center_avg
    if model_center_average[0] != 0:
        _prev_model_center_avg = model_center_average

    # --- Write state back (helper for early return) ---
    def _write_state_back():
        if state is not None:
            state['ray_lines']             = _ray_lines
            state['model_centers']         = _model_centers
            state['stored_intersections']  = _stored_intersections
            state['max_observed_distance'] = _max_observed_distance
            state['prev_model_center_avg'] = _prev_model_center_avg
            state['cached_threshold_idx']  = _cached_threshold_idx
        else:
            global ray_lines, model_centers, stored_intersections
            global max_observed_distance, prev_model_center_avg, cached_threshold_idx
            ray_lines             = _ray_lines
            model_centers         = _model_centers
            stored_intersections  = _stored_intersections
            max_observed_distance = _max_observed_distance
            prev_model_center_avg = _prev_model_center_avg
            cached_threshold_idx  = _cached_threshold_idx

    # Safety check
    if (center_x is None or center_y is None
            or model_center_average[0] is None or model_center_average[1] is None):
        _write_state_back()
        return final_rotated_rect, None

    # Adaptive distance (currently fixed)
    if len(_model_centers) >= 100 and center_x is not None:
        distance = math.sqrt((center_x - model_center_average[0]) ** 2
                             + (center_y - model_center_average[1]) ** 2)
        if distance > _max_observed_distance:
            _max_observed_distance = distance
    _max_observed_distance = 202

    # --- Drawing ---
    cv2.circle(frame, model_center_average, int(_max_observed_distance), (255, 50, 50), 2)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)

    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)

    if final_rotated_rect is not None:
        cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2)

    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        dx = center_x - model_center_average[0]
        dy = center_y - model_center_average[1]
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)
        cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3)

    if render_cv_window:
        cv2.imshow("Best Thresholded Image Contours on Frame", frame)

    # GL sphere (main thread only, single-eye mode only)
    gl_image = None
    if GL_SPHERE_AVAILABLE and state is None:
        try:
            gl_image = gl_sphere.update_sphere_rotation(
                center_x, center_y, model_center_average[0], model_center_average[1])
        except Exception:
            gl_image = None

    # Compute gaze vector
    center, direction = compute_gaze_vector(
        center_x, center_y, model_center_average[0], model_center_average[1])

    if center is not None and direction is not None:
        origin_text = f"Origin: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        dir_text    = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"

        text_origin = (12, frame.shape[0] - 38)
        text_dir    = (12, frame.shape[0] - 13)
        text_origin2 = (10, frame.shape[0] - 40)
        text_dir2    = (10, frame.shape[0] - 15)

        cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, origin_text, text_origin2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, dir_text, text_dir2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if center is not None and direction is not None:
        print(f"Sphere Center:   ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"Gaze Direction:  ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})")
    else:
        print("No valid intersection found.")

    if not _suppress_internal_windows:
        cv2.imshow("Frame with Ellipse and Rays", frame)

    if GL_SPHERE_AVAILABLE and gl_image is not None:
        blended = cv2.addWeighted(frame, 0.6, gl_image, 0.4, 0)
        cv2.imshow("Eye Tracker + Sphere", blended)

    _write_state_back()
    return final_rotated_rect, direction


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def update_and_average_point(point_list, new_point, N):
    point_list.append(new_point)
    if len(point_list) > N:
        point_list.pop(0)
    if not point_list:
        return None
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)


def draw_orthogonal_ray(image, ellipse, length=100, color=(0, 255, 0), thickness=1):
    (cx, cy), (major_axis, minor_axis), angle = ellipse
    angle_rad = np.deg2rad(angle)
    normal_dx = (minor_axis / 2) * np.cos(angle_rad)
    normal_dy = (minor_axis / 2) * np.sin(angle_rad)
    pt1 = (int(cx - length * normal_dx / (minor_axis / 2)),
           int(cy - length * normal_dy / (minor_axis / 2)))
    pt2 = (int(cx + length * normal_dx / (minor_axis / 2)),
           int(cy + length * normal_dy / (minor_axis / 2)))
    cv2.line(image, pt1, pt2, color, thickness)
    return image


def compute_average_intersection(frame, ray_lines_arg, stored_intersections_arg, N, M, spacing):
    if len(ray_lines_arg) < 2 or N < 2:
        return (0, 0)

    height, width = frame.shape[:2]
    selected_lines = random.sample(ray_lines_arg, min(N, len(ray_lines_arg)))
    intersections = []

    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]
        angle1 = line1[2]
        angle2 = line2[2]

        if abs(angle1 - angle2) >= 2:
            intersection = find_line_intersection(line1, line2)
            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                intersections.append(intersection)
                stored_intersections_arg.append(intersection)

    if len(stored_intersections_arg) > M:
        stored_intersections_arg[:] = stored_intersections_arg[-M:]

    if not intersections:
        return None

    avg_x = np.mean([pt[0] for pt in stored_intersections_arg])
    avg_y = np.mean([pt[1] for pt in stored_intersections_arg])
    return (int(avg_x), int(avg_y))


def prune_intersections(intersections, maximum_intersections):
    if len(intersections) <= maximum_intersections:
        return intersections
    return intersections[-maximum_intersections:]


def find_line_intersection(ellipse1, ellipse2):
    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2

    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)

    dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)

    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])

    if np.linalg.det(A) == 0:
        return None

    t1, _ = np.linalg.solve(A, B)
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1
    return (int(intersection_x), int(intersection_y))


def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    viewport_width = screen_width
    viewport_height = screen_height
    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0
    camera_position = np.array([0.0, 0.0, 3.0])

    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height

    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    origin = camera_position
    direction = -ray_direction
    L = origin - sphere_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius ** 2

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        t = -np.dot(direction, L) / np.dot(direction, direction)
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        nrm = np.linalg.norm(intersection_local)
        if nrm < 1e-9:
            return None, None
        target_direction = intersection_local / nrm
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

    # Compute rotation from +Z to target direction
    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        return sphere_center, circle_local_center

    rotation_axis /= rotation_axis_norm
    dot = np.clip(np.dot(circle_local_center, target_direction), -1.0, 1.0)
    angle_rad = np.arccos(dot)

    c_a = np.cos(angle_rad)
    s_a = np.sin(angle_rad)
    t_ = 1 - c_a
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_ * x_ * x_ + c_a,      t_ * x_ * y_ - s_a * z_, t_ * x_ * z_ + s_a * y_],
        [t_ * x_ * y_ + s_a * z_, t_ * y_ * y_ + c_a,      t_ * y_ * z_ - s_a * x_],
        [t_ * x_ * z_ - s_a * y_, t_ * y_ * z_ + s_a * x_, t_ * z_ * z_ + c_a],
    ])

    gaze_local = np.array([0.0, 0.0, inner_radius])
    gaze_rotated = rotation_matrix @ gaze_local
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    # Write gaze to file
    file_path = "gaze_vector.txt"
    try:
        with open(file_path, "w") as f:
            all_values = np.concatenate((sphere_center, gaze_rotated))
            csv_line = ",".join(f"{v:.6f}" for v in all_values)
            f.write(csv_line + "\n")
    except Exception as e:
        print("Write error:", e)

    return sphere_center, gaze_rotated


# ---------------------------------------------------------------------------
# Single-eye entry points
# ---------------------------------------------------------------------------

def process_frame(frame):
    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]

    th_strict  = mask_outside_square(apply_binary_threshold(gray_frame, darkest_pixel_value, 5),  darkest_point, 250)
    th_medium  = mask_outside_square(apply_binary_threshold(gray_frame, darkest_pixel_value, 15), darkest_point, 250)
    th_relaxed = mask_outside_square(apply_binary_threshold(gray_frame, darkest_pixel_value, 25), darkest_point, 250)

    final_rotated_rect, _ = process_frames(
        th_strict, th_medium, th_relaxed, frame, gray_frame, darkest_point, False, False)
    return final_rotated_rect


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


def process_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_path:
        return
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


def selection_gui():
    global selected_camera
    cameras = detect_cameras()

    root = tk.Tk()
    root.title("Select Input Source")
    tk.Label(root, text="Orlosky Eye Tracker 3D", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(root, text="Select Camera:").pack(pady=5)

    selected_camera = tk.StringVar()
    selected_camera.set(str(cameras[0]) if cameras else "No cameras found")
    ttk.Combobox(root, textvariable=selected_camera, values=[str(cam) for cam in cameras]).pack(pady=5)

    tk.Button(root, text="Start Camera", command=lambda: [root.destroy(), process_camera()]).pack(pady=5)
    tk.Button(root, text="Browse Video",  command=lambda: [root.destroy(), process_video()]).pack(pady=5)

    if GL_SPHERE_AVAILABLE:
        gl_sphere.start_gl_window()

    root.mainloop()


# ---------------------------------------------------------------------------
# Per-eye state factory
# ---------------------------------------------------------------------------

def _make_eye_state():
    return {
        'ray_lines':             [],
        'model_centers':         [],
        'stored_intersections':  [],
        'max_observed_distance': 0,
        'prev_model_center_avg': (320, 240),
        'cached_threshold_idx':  None,
    }


# ---------------------------------------------------------------------------
# Threaded per-eye worker
# ---------------------------------------------------------------------------

def _thread_process_eye(cap, eye_state, results, eye_key, mirror_frame=None):
    frame_key = eye_key + '_frame'
    dir_key   = eye_key + '_dir'
    ret_key   = eye_key + '_ret'

    results[frame_key] = None
    results[dir_key]   = None
    results[ret_key]   = False

    if mirror_frame is not None:
        frame = cv2.flip(mirror_frame, 1)
        ret = True
    else:
        ret, frame = cap.read()

    if not ret or frame is None:
        return

    results[ret_key] = True

    frame_disp = crop_to_aspect_ratio(frame.copy())
    darkest_pt = get_darkest_area(frame_disp)
    if darkest_pt is None:
        return

    gray = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY)
    dpv = gray[darkest_pt[1], darkest_pt[0]]

    th_s = mask_outside_square(apply_binary_threshold(gray, dpv, 5),  darkest_pt, 250)
    th_m = mask_outside_square(apply_binary_threshold(gray, dpv, 15), darkest_pt, 250)
    th_r = mask_outside_square(apply_binary_threshold(gray, dpv, 25), darkest_pt, 250)

    _, direction = process_frames(
        th_s, th_m, th_r, frame_disp, gray, darkest_pt, False, False, state=eye_state)

    results[frame_key] = frame_disp
    results[dir_key]   = direction


# ---------------------------------------------------------------------------
# Dual tracking main loop
# ---------------------------------------------------------------------------

def run_dual_tracking(src_left, src_right=None, mirror_mode=False):
    cap_l = cv2.VideoCapture(src_left)
    cap_r = cv2.VideoCapture(src_right if (src_right and not mirror_mode) else src_left)

    if not cap_l.isOpened():
        print(f"Error: Could not open left source: {src_left}")
        return
    if not cap_r.isOpened():
        print(f"Error: Could not open right source: {src_right}")
        return

    eye_state_left  = _make_eye_state()
    eye_state_right = _make_eye_state()
    gaze_world.reset_smoothing()

    global _suppress_internal_windows
    _suppress_internal_windows = True

    cv2.namedWindow("Left Eye - Gaze",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Eye - Gaze", cv2.WINDOW_NORMAL)

    if GL_SPHERE_AVAILABLE:
        gl_sphere.start_gl_window()
    if VIZ_3D_AVAILABLE:
        gaze_viz_3d.start()

    viz_update_interval = 3
    frame_idx = 0

    try:
        while True:
            results = {}

            if mirror_mode:
                ret_pre, frame_pre = cap_l.read()
                mirror_src = frame_pre if ret_pre else None
            else:
                mirror_src = None

            t_l = threading.Thread(
                target=_thread_process_eye,
                args=(cap_l, eye_state_left, results, 'l'),
                kwargs={'mirror_frame': mirror_src if mirror_mode else None},
                daemon=True)
            t_r = threading.Thread(
                target=_thread_process_eye,
                args=(cap_r, eye_state_right, results, 'r'),
                kwargs={'mirror_frame': mirror_src if mirror_mode else None},
                daemon=True)

            t_l.start()
            t_r.start()
            t_l.join()
            t_r.join()

            if not results.get('l_ret') and not results.get('r_ret'):
                break

            # Display (main thread — required on most platforms)
            frame_l_disp = results.get('l_frame')
            frame_r_disp = results.get('r_frame')

            if frame_l_disp is not None:
                frame_l_disp = cv2.rotate(frame_l_disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imshow("Left Eye - Gaze", frame_l_disp)
            if frame_r_disp is not None:
                frame_r_disp = cv2.rotate(frame_r_disp, cv2.ROTATE_90_CLOCKWISE)
                cv2.imshow("Right Eye - Gaze", frame_r_disp)

            l_direction = results.get('l_dir')
            r_direction = results.get('r_dir')

            # --- World-space gaze ---
            if l_direction is not None and r_direction is not None:
                result = gaze_world.compute_world_gaze_projected(l_direction, r_direction)
                if result is not None:
                    tx, ty, tz = result['target_world']
                    print(f"Gaze target (world): "
                          f"X={tx:+.2f}  Y={ty:.2f}  Z={tz:+.2f} m  "
                          f"| height={result['height']:.2f} m  "
                          f"| dist={result['distance']:.2f} m  "
                          f"| miss={result['miss_distance'] * 100:.1f} cm")

                    if VIZ_3D_AVAILABLE and frame_idx % viz_update_interval == 0:
                        gaze_viz_3d.update(
                            result['left_origin_world'],
                            result['left_direction_world'],
                            result['right_origin_world'],
                            result['right_direction_world'],
                            result['target_world'])

            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

    finally:
        _suppress_internal_windows = False
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

    tk.Label(root, text="Left Eye Source:", font=("Arial", 10, "bold")).pack(pady=5)
    sel_l = tk.StringVar(value=str(cameras[0]) if cameras else "0")
    ttk.Combobox(root, textvariable=sel_l, values=[str(c) for c in cameras]).pack()

    tk.Label(root, text="Right Eye Source:", font=("Arial", 10, "bold")).pack(pady=5)
    sel_r = tk.StringVar(value=str(cameras[1]) if len(cameras) > 1 else "1")
    ttk.Combobox(root, textvariable=sel_r, values=[str(c) for c in cameras]).pack()

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

    tk.Button(root, text="Start 2 Streams", command=start_streams).pack(pady=10)
    tk.Button(root, text="Start 2 Videos",  command=start_videos).pack(pady=5)
    tk.Button(root, text="Start 1 Video (Mirrored)", command=start_mirrored, bg="lightblue").pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    dual_selection_gui()