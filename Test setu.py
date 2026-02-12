import os
import numpy as np
import torch
import pickle

print("\n" + "="*60)
print("Model and Data Check")
print("="*60)

# 1. Check data existence
print("\n1. Checking data...")
if os.path.exists('data'):
    actions = os.listdir('data')
    print("data folder exists")
    print(f"Available classes: {actions}")
    
    for action in actions:
        action_path = os.path.join('data', action)
        if os.path.isdir(action_path):
            num_files = len([f for f in os.listdir(action_path) if f.endswith('.npy')])
            print(f"   - {action}: {num_files} samples")
else:
    print("data folder not found! Run collect_data.py first")

# 2. Check actions.pkl file
print("\n2. Checking actions file...")
if os.path.exists('actions.pkl'):
    with open('actions.pkl', 'rb') as f:
        actions = pickle.load(f)
    print("actions.pkl exists")
    print(f"Classes: {actions}")
else:
    print("actions.pkl not found!")

# 3. Check model
print("\n3. Checking model...")
if os.path.exists('sign_language_model.pth'):
    try:
        checkpoint = torch.load('sign_language_model.pth', map_location='cpu')
        print("sign_language_model.pth exists")
        print("Model information:")
        print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   - Accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        print(f"   - Input size: {checkpoint.get('input_size', 'N/A')}")
        print(f"   - Hidden size: {checkpoint.get('hidden_size', 'N/A')}")
        print(f"   - Num classes: {checkpoint.get('num_classes', 'N/A')}")
    except Exception as e:
        print(f"Error reading model: {e}")
else:
    print("sign_language_model.pth not found! Run train_model.py first")

# 4. Check hand_landmarker.task
print("\n4. Checking MediaPipe file...")
if os.path.exists('hand_landmarker.task'):
    size_mb = os.path.getsize('hand_landmarker.task') / (1024 * 1024)
    print(f"hand_landmarker.task exists ({size_mb:.1f} MB)")
else:
    print("hand_landmarker.task not found!")
    print("Download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

# 5. Test loading sample
print("\n5. Testing sample loading...")
try:
    if os.path.exists('data'):
        first_action = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))][0]
        first_file = os.path.join('data', first_action, 'sequence_0.npy')
        
        if os.path.exists(first_file):
            sample = np.load(first_file)
            print(f"Loaded sample from {first_action}")
            print(f"Data shape: {sample.shape}")
            print(f"   - Number of frames: {sample.shape[0]}")
            print(f"   - Number of features: {sample.shape[1]}")
        else:
            print(f"Sample file not found: {first_file}")
except Exception as e:
    print(f"Error: {e}")

# Summary
print("\n" + "="*60)
print("Summary:")
print("="*60)

ready = True
messages = []

if not os.path.exists('data'):
    ready = False
    messages.append("Collect data: Run collect_data.py")
else:
    messages.append("Collect data: OK")

if not os.path.exists('sign_language_model.pth'):
    ready = False
    messages.append("Train model: Run train_model.py")
else:
    messages.append("Train model: OK")

if not os.path.exists('hand_landmarker.task'):
    ready = False
    messages.append("MediaPipe file: Download from the link")
else:
    messages.append("MediaPipe file: OK")

for msg in messages:
    print(msg)

if ready:
    print("\nEverything is ready! Run the application:")
    print("   python sign_language_app.py")
else:
    print("\nSome files are missing! Complete the steps above")

print("="*60 + "\n")
