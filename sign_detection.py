import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1200x700")
        self.root.configure(bg='white')
        
        # Variables
        self.current_sentence = ""
        self.current_char = ""
        self.suggestions = []
        
        # Setup MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )
        
        self.detector = HandLandmarker.create_from_options(options)
        self.frame_count = 0
        
        # Define hand connections
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        # Setup camera
        self.cap = cv2.VideoCapture(0)
        
        # Build UI
        self.setup_ui()
        
        # Start update
        self.update_frame()
    
    def setup_ui(self):
        # Title
        title = tk.Label(
            self.root, 
            text="Sign Language To Text Conversion",
            font=("Courier", 24, "bold"),
            bg='white'
        )
        title.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40)
        
        # Camera frame
        camera_frame = tk.Frame(main_frame, bg='black', width=400, height=400)
        camera_frame.grid(row=0, column=0, padx=20, pady=20, sticky='nw')
        camera_frame.grid_propagate(False)
        
        self.camera_label = tk.Label(camera_frame, bg='black')
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Hand skeleton frame
        canvas_frame = tk.Frame(main_frame, bg='white', highlightbackground='gray', highlightthickness=1)
        canvas_frame.grid(row=0, column=1, padx=20, pady=20, sticky='ne')
        
        self.hand_canvas = tk.Canvas(
            canvas_frame, 
            width=350, 
            height=350, 
            bg='white', 
            highlightthickness=0
        )
        self.hand_canvas.pack()
        
        # Detected text
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
            font=("Courier", 16),
            bg='white'
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
            bg='white'
        )
        self.sentence_label.pack(side=tk.LEFT)
        
        # Suggestions
        self.suggestions_frame = tk.Frame(main_frame, bg='white')
        self.suggestions_frame.grid(row=2, column=0, columnspan=2, sticky='w', pady=20)
        
        tk.Label(
            self.suggestions_frame, 
            text="Suggestions :", 
            font=("Courier", 14, "bold"),
            bg='white',
            fg='red'
        ).pack(side=tk.LEFT, padx=10)
        
        # Buttons placeholder (added dynamically)
        self.suggestion_buttons_frame = tk.Frame(self.suggestions_frame, bg='white')
        self.suggestion_buttons_frame.pack(side=tk.LEFT)
        
        # Control buttons
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.grid(row=2, column=1, sticky='e', pady=20, padx=20)
        
        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            font=("Courier", 12, "bold"),
            bg='lightgray',
            fg='black',
            relief=tk.SOLID,
            borderwidth=2,
            padx=20,
            pady=8,
            command=self.clear_sentence
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        speak_btn = tk.Button(
            button_frame,
            text="Speak",
            font=("Courier", 12, "bold"),
            bg='lightgray',
            fg='black',
            relief=tk.SOLID,
            borderwidth=2,
            padx=20,
            pady=8,
            command=self.speak_sentence
        )
        speak_btn.pack(side=tk.LEFT, padx=5)
        
        exit_btn = tk.Button(
            button_frame,
            text="Exit",
            font=("Courier", 12, "bold"),
            bg='lightgray',
            fg='black',
            relief=tk.SOLID,
            borderwidth=2,
            padx=20,
            pady=8,
            command=self.on_closing
        )
        exit_btn.pack(side=tk.LEFT, padx=5)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Hand detection
            results = self.detector.detect_for_video(mp_image, self.frame_count)
            
            # Draw on frame
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    points = []
                    for landmark in hand_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        points.append((x, y))
                    
                    # Draw connections
                    for connection in self.HAND_CONNECTIONS:
                        start_point = points[connection[0]]
                        end_point = points[connection[1]]
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                    
                    # Draw points
                    for point in points:
                        cv2.circle(frame, point, 5, (255, 0, 0), -1)
                
                # Show number of hands detected
                cv2.putText(
                    frame,
                    f"Detected {len(results.hand_landmarks)} hand(s)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Display camera
            frame_resized = cv2.resize(frame, (400, 400))
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            
            # Draw hand skeleton on canvas
            self.draw_hand_skeleton(results)
        
        self.root.after(10, self.update_frame)
    
    def draw_hand_skeleton(self, results):
        self.hand_canvas.delete("all")
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                points = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * 350)
                    y = int(landmark.y * 350)
                    points.append((x, y))
                
                # Draw connections
                for connection in self.HAND_CONNECTIONS:
                    self.hand_canvas.create_line(
                        points[connection[0]][0], points[connection[0]][1],
                        points[connection[1]][0], points[connection[1]][1],
                        fill='#00FF00', width=3
                    )
                
                # Draw points
                for point in points:
                    self.hand_canvas.create_oval(
                        point[0]-5, point[1]-5,
                        point[0]+5, point[1]+5,
                        fill='blue', outline='blue'
                    )
    
    def update_suggestions(self, suggestions_list):
        """Update suggestions dynamically"""
        
        # Add new buttons
        for suggestion in suggestions_list:
            btn = tk.Button(
                self.suggestion_buttons_frame,
                text=suggestion,
                font=("Courier", 12, "bold"),
                bg='lightgray',
                fg='black',
                relief=tk.SOLID,
                borderwidth=2,
                padx=15,
                pady=8,
                command=lambda s=suggestion: self.use_suggestion(s)
            )
            btn.pack(side=tk.LEFT, padx=5)
    
    def use_suggestion(self, suggestion):
        self.sentence_label.config(text=suggestion)
        self.current_sentence = suggestion
    
    def clear_sentence(self):
        self.sentence_label.config(text="")
        self.current_sentence = ""
        self.char_label.config(text="")
    
    def speak_sentence(self):
        # Placeholder for text-to-speech
        print(f"Speaking: {self.current_sentence}")
    
    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


