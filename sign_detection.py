import cv2
import mediapipe as mp

print("Program is running...")
print("Press 'q' to exit")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

detector = HandLandmarker.create_from_options(options)

# Define connections (links between landmarks)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Little finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect_for_video(mp_image, frame_count)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Convert landmarks to coordinates
            points = []
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                points.append((x, y))
            
            # Draw lines first
            for connection in HAND_CONNECTIONS:
                start_point = points[connection[0]]
                end_point = points[connection[1]]
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            
            # Draw points over the lines
            for point in points:
                cv2.circle(frame, point, 5, (255, 0, 0), -1)
        
        cv2.putText(
            frame,
            f"Detected {len(results.hand_landmarks)} hand(s)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    cv2.imshow("Hand Detection - Press Q to Exit", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed successfully")


