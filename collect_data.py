import cv2
import mediapipe as mp
import numpy as np
import os
import time

print("=== Sign Language Data Collection System ===")

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

# Create data folder
if not os.path.exists('data'):
    os.makedirs('data')

# Action name (letter or word)
action = input("Enter action name (example: alef, baa, hello): ")
action_path = os.path.join('data', action)

if not os.path.exists(action_path):
    os.makedirs(action_path)

sequences = 30  # Number of recordings
frames = 10     # Frames per recording

cap = cv2.VideoCapture(0)
frame_count = 0

print(f"\nReady to record {sequences} times for action: {action}")
print("Press 'SPACE' to start, 'q' to quit")

sequence_num = 0
recording = False
recorded_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect_for_video(mp_image, frame_count)
    

    landmarks_data = []
    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            for landmark in hand:
                landmarks_data.extend([landmark.x, landmark.y, landmark.z])
    
    # If no hand detected, fill with zeros
    if len(landmarks_data) == 0:
        landmarks_data = [0] * 63  # 21 points × 3 values
    
    # If only one hand detected, add zeros for second hand
    if len(landmarks_data) == 63:
        landmarks_data.extend([0] * 63)
    
    # Recording
    if recording:
        recorded_frames.append(np.array(landmarks_data))
        
        cv2.putText(frame, f"Recording: {len(recorded_frames)}/{frames}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if len(recorded_frames) == frames:
            # Save recording
            seq_path = os.path.join(action_path, str(sequence_num))
            os.makedirs(seq_path, exist_ok=True)
            
            for frame_idx, frame_data in enumerate(recorded_frames):
                np.save(os.path.join(seq_path, str(frame_idx) + '.npy'), frame_data)
            
            sequence_num += 1
            recorded_frames = []
            recording = False
            
            print(f"Saved recording {sequence_num}/{sequences}")
            
            if sequence_num >= sequences:
                print(f"\nFinished! Collected {sequences} recordings for action '{action}'")
                break
            
            time.sleep(1)  # Short pause
    
    # Display status
    cv2.putText(frame, f"Action: {action}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Collected: {sequence_num}/{sequences}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if not recording:
        cv2.putText(frame, "Press SPACE to start recording", (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw hand landmarks
    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Data Collection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and not recording and sequence_num < sequences:
        recording = True
        recorded_frames = []
        print(f"Recording {sequence_num + 1}...")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nRecording finished!")

