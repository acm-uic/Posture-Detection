import os
import cv2
import mediapipe as mp
import numpy as np
from groq import Groq
from drawing import draw_landmarks
from dotenv import load_dotenv

load_dotenv()
# Get a groq api key from https://console.groq.com/keys for free as assign in .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    print("Failed to open camera")
    exit()


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

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def is_side_profile(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_difference = abs(left_shoulder.x - right_shoulder.x)

    if shoulder_difference < 0.05:
        return True
    return False


def degreeFromLeftShoulder(landmark):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    shoulder_midpointX = (right_shoulder.x + left_shoulder.x) / 2
    shoulder_midpointY = (right_shoulder.y + left_shoulder.y) / 2

    LeftshoulderPoint = [left_shoulder.x, left_shoulder.y]
    shoulderMidPoint = [shoulder_midpointX, shoulder_midpointY]
    nosePoint = [nose.x, nose.y]

    print(calculate_angle(LeftshoulderPoint, shoulderMidPoint, nosePoint))


def analyze_side_posture(landmarks):
    """
    Analyze posture from side view
    Returns angles for neck and back alignment
    """
    # Get relevant landmarks for side profile
    shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ]
    hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
    ]
    knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
    ]
    ear = [
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,
    ]

    # Calculate angles
    neck_angle = calculate_angle(ear, shoulder, hip)
    back_angle = calculate_angle(shoulder, hip, knee)

    return neck_angle, back_angle

posture_data = []
MAX_FRAMES = 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect pose
    results = pose.process(image)

    # Black screen to put wireframe on
    wireImage = np.zeros_like(frame)

    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Filtered connections
    connections = frozenset(
        [
            (9, 10),
            (11, 12),
            (11, 23),
            (12, 24),
            (23, 24),
        ]
    )

    if results.pose_landmarks:
        # Draw pose landmarks on camera
        draw_landmarks(image, results.pose_landmarks, connections)

        # Draw pose landmarks on black screen
        draw_landmarks(wireImage, results.pose_landmarks, connections)

        # Analyze posture
        landmarks = results.pose_landmarks.landmark

        if is_side_profile(landmarks):
            side_profile_status = "Side profile: Detected"
            neck_angle, back_angle = analyze_side_posture(landmarks)
            frame_data={
                "neck_angle":neck_angle,
                "back_angle":back_angle,
                "timestamp": cv2.getTickCount() / cv2.getTickFrequency()
            }
            posture_data.append(frame_data)

            # Display angles
            cv2.putText(
                image,
                f"Neck angle: {neck_angle:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                f"Back angle: {back_angle:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                image,
                f"Frames collected: {len(posture_data)}/{MAX_FRAMES}",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            if len(posture_data) >= MAX_FRAMES:
                break

            # Basic posture feedback
            if 80 <= neck_angle <= 100:
                neck_status = "Good neck alignment"
            else:
                neck_status = "Adjust neck position"

            if 160 <= back_angle <= 180:
                back_status = "Good back alignment"
            else:
                back_status = "Adjust back position"
            

            cv2.putText(
                image,
                neck_status,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                back_status,
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        else:
            side_profile_status = "Side profile: Not detected"
        cv2.putText(
            image,
            side_profile_status,
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    # Display the image
    cv2.imshow("Side Profile Pose Detection", image)
    cv2.imshow("WireFrame", wireImage)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()

def getting_llm_response(query) -> str:
    chat_completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        max_completion_tokens=2049,
        messages=[
            {
                "role":"system",
                "content":"You are a Posture specialist, and can analyse a patients neck and back angle to generate a posture report. You will be given the neck and back angle as input. You are very knowledgeable.You also provide tips on how to fix the posture, if posture is bad."
            },
            {
                "role":"user",
                "content":query
            }
        ]
    )
    return chat_completion.choices[0].message.content

if posture_data:
    # Calculate averages
    avg_neck_angle = sum(d["neck_angle"] for d in posture_data) / len(posture_data)
    avg_back_angle = sum(d["back_angle"] for d in posture_data) / len(posture_data)
    duration = posture_data[-1]["timestamp"] - posture_data[0]["timestamp"]
    perfect_neck_count = sum(1 for d in posture_data if 80 <= d["neck_angle"] <= 100)
    perfect_back_count = sum(1 for d in posture_data if 160 <= d["back_angle"] <= 180)
    total_frames = len(posture_data)

    # Format data for LLM
    llm_input = f"""
    This is the patient's posture details:
    - Average Neck Angle: {avg_neck_angle:.1f} degrees
    - Average Back Angle: {avg_back_angle:.1f} degrees
    """

    print("Data to send to LLM:")
    # print(llm_input)
    print(getting_llm_response(llm_input))
    print(llm_input)
else:
    print("No posture data collected.")