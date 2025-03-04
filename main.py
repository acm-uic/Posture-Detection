import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime



# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

def calculate_angle(a, b, c):
    """
    Calculate angle between three points
    Args:
        a, b, c: Points in format [x, y]
    Returns:
        Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def analyze_side_posture(landmarks):
    """
    Analyze posture from side view
    Returns angles for neck and back alignment
    """
    # Get relevant landmarks for side profile
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
           landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
    
    # Calculate angles
    neck_angle = calculate_angle(ear, shoulder, hip)
    back_angle = calculate_angle(shoulder, hip, knee)
    
    return neck_angle, back_angle
    

#calutating start time
starttime = time.time()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect pose
    results = pose.process(image)
    
    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)
        
        # Analyze posture
        landmarks = results.pose_landmarks.landmark
        neck_angle, back_angle = analyze_side_posture(landmarks)
        
        #finding current time
        currenttime = time.time()
        durationtime = currenttime - starttime

        # Display angles
        cv2.putText(image, f'Neck angle: {neck_angle:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Back angle: {back_angle:.1f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"Duration: {durationtime:.2f} secs", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Basic posture feedback
        if 80 <= neck_angle <= 100:
            neck_status = "Good neck alignment"
        else:
            neck_status = "Adjust neck position"
            
        if 160 <= back_angle <= 180:
            back_status = "Good back alignment"
        else:
            back_status = "Adjust back position"
            
        cv2.putText(image, neck_status, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, back_status, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow('Side Profile Pose Detection', image)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
