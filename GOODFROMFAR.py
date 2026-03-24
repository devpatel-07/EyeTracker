import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- FIXED CONSTANTS ---
EYE_RADIUS_MM = 12.0  # r = 12mm from notes
# Change this to roughly how far your face is from the webcam!
# 450mm to 600mm is standard laptop viewing distance.
ESTIMATED_DISTANCE_MM = 500.0  

def get_camera_matrix(width, height):
    focal_length = width 
    center_x = width / 2.0
    center_y = height / 2.0
    return center_x, center_y, focal_length

def solve_ray_sphere_intersection(O, d, C, r):
    """
    Exactly matches notes: (d.d)t^2 + 2d(O-C)t + ||O-C||^2 - r^2 = 0
    """
    oc = O - C
    
    a = np.dot(d, d)
    b = 2.0 * np.dot(oc, d)
    c = np.dot(oc, oc) - r**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant > 0:
        # Subtract sqrt to get the surface point closest to the camera
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t > 0:
            return O + t * d
    return None

def main():
    cap = cv2.VideoCapture(0)
    
    # Camera origin (O)
    O = np.array([0.0, 0.0, 0.0])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        Cx, Cy, f = get_camera_matrix(w, h)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. Observed 2D pupil point: P_img = (u,v)
            left_iris = landmarks[468] 
            u, v = left_iris.x * w, left_iris.y * h
            
            # Find 2D eye center using corners
            corner_in = landmarks[133]
            corner_out = landmarks[33]
            eye_center_x_2d = ((corner_in.x + corner_out.x) / 2.0) * w
            eye_center_y_2d = ((corner_in.y + corner_out.y) / 2.0) * h
            
            # 2. 3D Eye Center (C)
            # Unprojecting using standard pinhole model at assumed distance
            C_x = (eye_center_x_2d - Cx) * ESTIMATED_DISTANCE_MM / f
            C_y = (eye_center_y_2d - Cy) * ESTIMATED_DISTANCE_MM / f
            # Eyeball center is shifted back from the surface by radius 'r'
            C_z = ESTIMATED_DISTANCE_MM + EYE_RADIUS_MM 
            C = np.array([C_x, C_y, C_z])
            
            # 3. Ray direction (d) -> matching notes: d = (x, y, f)
            d = np.array([u - Cx, v - Cy, f])
            d = d / np.linalg.norm(d) # Normalize
            
            # 4. Ray Equation -> R(t) = O + td to find true 3D pupil center (P)
            P = solve_ray_sphere_intersection(O, d, C, EYE_RADIUS_MM)
            
            if P is not None:
                # 5. Gaze Vector: g = (P - C) / ||P - C||
                g = (P - C) / np.linalg.norm(P - C)
                
                # --- Drawing / Visualization ---
                cv2.circle(frame, (int(u), int(v)), 3, (0, 255, 0), -1)
                cv2.circle(frame, (int(eye_center_x_2d), int(eye_center_y_2d)), 2, (0, 0, 255), -1)
                
                # Project the vector forward by 60mm to make it visible
                gaze_end_3d = P + (g * 60.0) 
                
                # Map back to screen space for drawing: X_pixel = fx(X/Z) + Cx
                end_x_2d = int((gaze_end_3d[0] * f / gaze_end_3d[2]) + Cx)
                end_y_2d = int((gaze_end_3d[1] * f / gaze_end_3d[2]) + Cy)
                
                cv2.line(frame, (int(u), int(v)), (end_x_2d, end_y_2d), (255, 0, 0), 2)
                
                text = f"Vector g: [{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}]"
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('3D Gaze Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()