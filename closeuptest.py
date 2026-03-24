import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# --- PHYSICAL CONSTANTS ---
EYE_RADIUS_MM = 12.0
AVG_PUPIL_DISTANCE_MM = 63.0 # Average distance between human eyes

def solve_ray_sphere_intersection(O, d, C, r):
    oc = O - C
    a = np.dot(d, d)
    b = 2.0 * np.dot(oc, d)
    c = np.dot(oc, oc) - r**2
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return O + t * d
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Camera Intrinsics
    f = w # Focal length approximation
    cx, cy = w / 2, h / 2
    O = np.array([0.0, 0.0, 0.0])

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        
        # --- DYNAMIC DISTANCE CALCULATION ---
        # Get left and right iris centers (468 and 473)
        l_iris = mesh[468]
        r_iris = mesh[473]
        
        # Calculate pixel distance between eyes
        dx_px = (l_iris.x - r_iris.x) * w
        dy_px = (l_iris.y - r_iris.y) * h
        dist_px = np.sqrt(dx_px**2 + dy_px**2)
        
        # Formula: Z = (Physical_Dist * Focal_Length) / Pixel_Dist
        # This updates ESTIMATED_DISTANCE_MM in real-time as you move closer/further
        current_z = (AVG_PUPIL_DISTANCE_MM * f) / dist_px

        # --- EYE GEOMETRY (Left Eye) ---
        pupil_2d = np.array([l_iris.x * w, l_iris.y * h])
        
        # Eye corners for center estimation
        c1, c2 = mesh[33], mesh[133]
        center_2d = np.array([((c1.x + c2.x)/2)*w, ((c1.y + c2.y)/2)*h])
        
        # 3. 3D Eyeball Center C
        # Note: We subtract Cx/Cy to center the coordinate system on the lens
        C = np.array([
            (center_2d[0] - cx) * current_z / f,
            (center_2d[1] - cy) * current_z / f,
            current_z + EYE_RADIUS_MM
        ])
        
        # 4. Ray direction d (from notes: d = (x, y, f))
        d = np.array([pupil_2d[0] - cx, pupil_2d[1] - cy, f])
        d = d / np.linalg.norm(d)
        
        # 5. Intersection P
        P = solve_ray_sphere_intersection(O, d, C, EYE_RADIUS_MM)
        
        if P is not None:
            # Gaze Vector g = (P - C) / ||P - C||
            g = (P - C) / np.linalg.norm(P - C)
            
            # --- VISUALIZATION ---
            # Project 3D points back to 2D for drawing
            g_end_3d = P + (g * 50.0) 
            end_2d = ( (g_end_3d[0:2] * f) / g_end_3d[2] ) + [cx, cy]
            
            cv2.line(frame, tuple(pupil_2d.astype(int)), tuple(end_2d.astype(int)), (255, 0, 0), 2)
            cv2.putText(frame, f"Dist: {int(current_z)}mm", (20, 40), 2, 0.7, (0,255,0), 1)
            cv2.putText(frame, f"Vector: [{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}]", (20, 70), 2, 0.7, (255,255,0), 1)

    cv2.imshow('Close-up Gaze Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()