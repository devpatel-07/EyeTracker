import cv2
import numpy as np

# --- 1. CONSTANTS & GLOBALS ---
EYE_RADIUS_MM = 12.0
# At 3-5mm distance, we'll set a fixed target Z for the eyeball surface
# In a fixed mount, this would be a calibrated constant.
SURFACE_DISTANCE_MM = 5.0 

# Storage for Eye Center Estimation (from block 1)
ellipses_history = [[0,0,0], [0,0,0], [0,0,0]] 
counter = 0
eye_centers_2d = []

# --- 2. THE MATH (From your notes & block 2) ---

def solve_ray_sphere_intersection(O, d, C, r):
    """Exactly follows: || O + td - C ||^2 = r^2"""
    oc = O - C
    a = np.dot(d, d)
    b = 2.0 * np.dot(oc, d)
    c = np.dot(oc, oc) - r**2
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return O + t * d
    return None

def eyecenter_estimation_2d(ellipses):
    """Intersects the minor axes of 3 ellipses to find the 2D center of rotation."""
    pts = []
    for i in range(3):
        cx, cy, angle_deg = ellipses[i]
        a = np.deg2rad(angle_deg)
        # Minor axis direction (perpendicular to major)
        dx, dy = -np.sin(a), np.cos(a)
        pts.append((cx, cy, dx, dy))

    intersections = []
    pairs = [(0,1), (1,2), (0,2)]
    for i, j in pairs:
        x1, y1, dx1, dy1 = pts[i]
        x2, y2, dx2, dy2 = pts[j]
        A = np.array([[dx1, -dx2], [dy1, -dy2]])
        B = np.array([x2 - x1, y2 - y1])
        if abs(np.linalg.det(A)) > 1e-3:
            t1, _ = np.linalg.solve(A, B)
            intersections.append((x1 + t1*dx1, y1 + t1*dy1))
    
    if not intersections: return None
    return np.mean(intersections, axis=0)

# --- 3. IMAGE PROCESSING (From block 1) ---

def get_pupil_ellipse(frame):
    """Manual CV to find the pupil without MediaPipe."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find darkest point
    min_val, _, min_loc, _ = cv2.minMaxLoc(cv2.blur(gray, (5,5)))
    
    # Threshold around the darkest point
    _, thresh = cv2.threshold(gray, min_val + 20, 255, cv2.THRESH_BINARY_INV)
    
    # Masking to local area (250px square like your code)
    mask = np.zeros_like(thresh)
    cv2.rectangle(mask, (min_loc[0]-125, min_loc[1]-125), (min_loc[0]+125, min_loc[1]+125), 255, -1)
    thresh = cv2.bitwise_and(thresh, mask)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            return cv2.fitEllipse(largest)
    return None

# --- 4. MAIN LOOP ---

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Camera Intrinsics
    f = w  # Focal length approximation
    cx, cy = w / 2.0, h / 2.0
    O = np.array([0.0, 0.0, 0.0])

    # 1. Find Pupil Ellipse
    ellipse = get_pupil_ellipse(frame)
    
    if ellipse:
        (ex, ey), (ma, mi), angle = ellipse
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
        
        # 2. Update Eyecenter Estimation (Running 3-frame window)
        ellipses_history[counter % 3] = [ex, ey, angle]
        counter += 1
        
        if counter >= 3:
            center_2d = eyecenter_estimation_2d(ellipses_history)
            if center_2d is not None:
                # Store and average for stability
                eye_centers_2d.append(center_2d)
                if len(eye_centers_2d) > 50: eye_centers_2d.pop(0)
                
                avg_c2d = np.mean(eye_centers_2d, axis=0)
                cv2.circle(frame, (int(avg_c2d[0]), int(avg_c2d[1])), 4, (255, 255, 0), -1)

                # 3. CONVERT TO 3D MM SPACE
                # Eye center C
                C_z = SURFACE_DISTANCE_MM + EYE_RADIUS_MM
                C = np.array([
                    (avg_c2d[0] - cx) * C_z / f,
                    (avg_c2d[1] - cy) * C_z / f,
                    C_z
                ])

                # Ray d pointing at the current pupil pixel
                d = np.array([ex - cx, ey - cy, f])
                d /= np.linalg.norm(d)

                # 4. SOLVE INTERSECTION P
                P = solve_ray_sphere_intersection(O, d, C, EYE_RADIUS_MM)

                if P is not None:
                    # 5. GAZE VECTOR g = (P - C) / ||P - C||
                    g = (P - C) / np.linalg.norm(P - C)

                    # 6. VISUALIZATION (Reprojection)
                    # Extend vector by 40mm
                    line_len = 40.0
                    g_end_3d = P + (g * line_len)
                    
                    # Project back to pixels
                    end_u = int((g_end_3d[0] * f / g_end_3d[2]) + cx)
                    end_v = int((g_end_3d[1] * f / g_end_3d[2]) + cy)
                    
                    # Draw gaze line from pupil to projected endpoint
                    cv2.line(frame, (int(ex), int(ey)), (end_u, end_v), (255, 0, 0), 3)
                    
                    # Output
                    cv2.putText(frame, f"Gaze: [{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}]", 
                                (20, 40), 1, 1.2, (255, 255, 0), 2)

    cv2.imshow('Merged Gaze Tracker (Macro)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()