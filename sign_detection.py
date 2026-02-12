
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import torch
import torch.nn as nn
import pickle
import numpy as np
import os
from collections import deque

# نفس تعريف النموذج اللي في train_model.py
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=64, num_layers=3, num_classes=3):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1200x700")
        self.root.configure(bg='white')

        # المتغيرات
        self.current_sentence = ""
        self.current_char = ""
        self.actions = []
        self.sequence = deque(maxlen=10)  # 10 إطارات
        self.frame_count = 0
        
        # عداد للتأكد من الحرف
        self.last_prediction = ""
        self.prediction_count = 0
        self.min_prediction_count = 10  # لازم يتكرر 10 مرات

        # تحميل النموذج
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load('sign_language_model.pth', map_location=self.device)
            with open('actions.pkl', 'rb') as f:
                self.actions = pickle.load(f)
            
            # إنشاء النموذج وتحميل الأوزان
            self.model = SignLanguageLSTM(
                checkpoint['input_size'], 
                checkpoint['hidden_size'],
                checkpoint['num_layers'], 
                checkpoint['num_classes']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ تم تحميل النموذج!")
            print(f"📝 الحروف: {self.actions}")
        except Exception as e:
            print(f"❌ خطأ في تحميل النموذج: {e}")
            self.model = None
            self.actions = []

        # إعداد MediaPipe - VIDEO mode بدلاً من LIVE_STREAM
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,  # VIDEO mode
            num_hands=2
        )

        self.detector = HandLandmarker.create_from_options(options)

        # Hand connections
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        # الكاميرا
        self.cap = cv2.VideoCapture(0)

        # الواجهة
        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        title = tk.Label(
            self.root, 
            text="Sign Language To Text Conversion",
            font=("Courier", 24, "bold"), 
            bg='white'
        )
        title.pack(pady=20)

        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40)

        # الكاميرا
        camera_frame = tk.Frame(main_frame, bg='black', width=400, height=400)
        camera_frame.grid(row=0, column=0, padx=20, pady=20)
        camera_frame.grid_propagate(False)

        self.camera_label = tk.Label(camera_frame, bg='black')
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # رسم اليد
        canvas_frame = tk.Frame(main_frame, bg='white', highlightbackground='gray', highlightthickness=1)
        canvas_frame.grid(row=0, column=1, padx=20, pady=20)

        self.hand_canvas = tk.Canvas(canvas_frame, width=350, height=350, bg='white')
        self.hand_canvas.pack()

        # النصوص
        text_frame = tk.Frame(main_frame, bg='white')
        text_frame.grid(row=1, column=0, columnspan=2, sticky='w', pady=10)

        char_frame = tk.Frame(text_frame, bg='white')
        char_frame.pack(anchor='w', pady=5)

        tk.Label(
            char_frame, 
            text="Character :", 
            font=("Courier", 16, "bold"), 
            bg='white'
        ).pack(side=tk.LEFT, padx=10)

        self.char_label = tk.Label(
            char_frame, 
            text=self.current_char, 
            font=("Courier", 20, "bold"),
            bg='white', 
            fg='blue'
        )
        self.char_label.pack(side=tk.LEFT)

        sentence_frame = tk.Frame(text_frame, bg='white')
        sentence_frame.pack(anchor='w', pady=5)

        tk.Label(
            sentence_frame, 
            text="Sentence :", 
            font=("Courier", 16, "bold"), 
            bg='white'
        ).pack(side=tk.LEFT, padx=10)

        self.sentence_label = tk.Label(
            sentence_frame, 
            text=self.current_sentence, 
            font=("Courier", 16),
            bg='white', 
            wraplength=800
        )
        self.sentence_label.pack(side=tk.LEFT)

        # الأزرار
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)

        tk.Button(
            button_frame, 
            text="Clear", 
            font=("Courier", 12, "bold"), 
            bg='lightgray',
            padx=20, 
            pady=8, 
            command=self.clear_sentence
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame, 
            text="Add Space", 
            font=("Courier", 12, "bold"), 
            bg='lightgray',
            padx=20, 
            pady=8, 
            command=self.add_space
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame, 
            text="Exit", 
            font=("Courier", 12, "bold"), 
            bg='lightgray',
            padx=20, 
            pady=8, 
            command=self.on_closing
        ).pack(side=tk.LEFT, padx=5)

    def extract_landmarks(self, results):
        """استخراج landmarks من اليد"""
        landmarks_data = []
        
        if results.hand_landmarks:
            for hand in results.hand_landmarks:
                for landmark in hand:
                    landmarks_data.extend([landmark.x, landmark.y, landmark.z])
        
        # إذا ما فيه يد
        if len(landmarks_data) == 0:
            landmarks_data = [0] * 63
        
        # إذا فيه يد وحدة بس
        if len(landmarks_data) == 63:
            landmarks_data.extend([0] * 63)
        
        return np.array(landmarks_data, dtype=np.float32)

    def predict_sign(self):
        """التنبؤ بالحرف"""
        if self.model is None or len(self.sequence) < 10:
            return None, 0
        
        try:
            X = torch.FloatTensor(np.expand_dims(self.sequence, axis=0)).to(self.device)
            
            with torch.no_grad():
                output = self.model(X)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_sign = self.actions[predicted_class.item()]
            conf = confidence.item()
            
            # نسبة ثقة أعلى من 70%
            if conf > 0.7:
                return predicted_sign, conf
            else:
                return None, conf
                
        except Exception as e:
            print(f"خطأ في التنبؤ: {e}")
            return None, 0

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        self.frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # كشف اليد
        results = self.detector.detect_for_video(mp_image, self.frame_count)
        
        # استخراج landmarks
        landmarks = self.extract_landmarks(results)
        self.sequence.append(landmarks)

        # التنبؤ
        if len(self.sequence) == 10:
            predicted_sign, confidence = self.predict_sign()
            
            if predicted_sign:
                # التأكد من الحرف بتكراره عدة مرات
                if predicted_sign == self.last_prediction:
                    self.prediction_count += 1
                else:
                    self.last_prediction = predicted_sign
                    self.prediction_count = 1
                
                # إذا تكرر كثير، نضيفه للجملة
                if self.prediction_count >= self.min_prediction_count:
                    if self.current_char != predicted_sign:
                        self.current_char = predicted_sign
                        self.current_sentence += predicted_sign
                        self.char_label.config(text=self.current_char)
                        self.sentence_label.config(text=self.current_sentence)
                    self.prediction_count = 0
                
                # عرض على الشاشة
                cv2.putText(
                    frame, 
                    f"Sign: {predicted_sign} ({confidence*100:.1f}%)",
                    (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )

        # رسم اليد على الفريم
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                points = [(int(l.x*frame.shape[1]), int(l.y*frame.shape[0])) for l in hand_landmarks]
                
                for c in self.HAND_CONNECTIONS:
                    cv2.line(frame, points[c[0]], points[c[1]], (0, 255, 0), 2)
                
                for p in points:
                    cv2.circle(frame, p, 5, (255, 0, 0), -1)
            
            cv2.putText(
                frame, 
                f"Detected {len(results.hand_landmarks)} hand(s)",
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

# عرض الكاميرا
        frame_resized = cv2.resize(frame, (400, 400))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.camera_label.imgtk = img
        self.camera_label.configure(image=img)

        # رسم اليد على Canvas
        self.draw_hand_skeleton(results)

        self.root.after(10, self.update_frame)

    def draw_hand_skeleton(self, results):
        self.hand_canvas.delete("all")
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                points = [(int(l.x*350), int(l.y*350)) for l in hand_landmarks]
                
                for c in self.HAND_CONNECTIONS:
                    self.hand_canvas.create_line(
                        points[c[0]][0], points[c[0]][1],
                        points[c[1]][0], points[c[1]][1],
                        fill='#00FF00', 
                        width=3
                    )
                
                for p in points:
                    self.hand_canvas.create_oval(
                        p[0]-5, p[1]-5, 
                        p[0]+5, p[1]+5, 
                        fill='blue'
                    )

    def add_space(self):
        self.current_sentence += " "
        self.sentence_label.config(text=self.current_sentence)

    def clear_sentence(self):
        self.current_sentence = ""
        self.current_char = ""
        self.sentence_label.config(text="")
        self.char_label.config(text="")
        self.sequence.clear()
        self.prediction_count = 0
        self.last_prediction = ""

    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()



    

