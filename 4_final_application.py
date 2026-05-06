import tkinter as tk
from tkinter import font
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import pandas as pd
import pickle
import pyttsx3
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Load AI and Speech Engine ---
try:
    with open('gesture_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Could not find 'gesture_model_final.pkl'.")
    exit()

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_text(text):
    if text.strip():
        engine.say(text)
        engine.runAndWait()

# --- 2. Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

columns = []
for i in range(21):
    columns.extend([f'x{i}', f'y{i}', f'z{i}'])

# --- 3. The Desktop Application Class ---
class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x700")
        self.window.configure(bg="#1E1E1E") # Dark mode background

        # Variables for Word Building & Stabilization
        self.current_word = ""
        self.last_prediction = ""
        
        # --- Stabilization Variables (Anti-Flicker Fix) ---
        self.current_candidate = ""
        self.consecutive_frames = 0
        self.STABILITY_THRESHOLD = 10 # The AI must see the same sign for 10 frames straight
        
        # --- UI Layout ---
        title_font = font.Font(family="Helvetica", size=24, weight="bold")
        ui_font = font.Font(family="Helvetica", size=14)

        # Header
        self.lbl_title = tk.Label(window, text="Task-Aware Sign Language Translator", font=title_font, bg="#1E1E1E", fg="#00FFCC")
        self.lbl_title.pack(pady=15)

        # Video Frame
        self.canvas = tk.Canvas(window, width=640, height=480, bg="#000000", highlightthickness=2, highlightbackground="#00FFCC")
        self.canvas.pack()

        # Word Display
        self.lbl_word = tk.Label(window, text="Current Word: ", font=("Helvetica", 20, "bold"), bg="#1E1E1E", fg="#FFFFFF")
        self.lbl_word.pack(pady=10)

        # Button Panel
        self.btn_frame = tk.Frame(window, bg="#1E1E1E")
        self.btn_frame.pack(pady=10)

        self.btn_speak = tk.Button(self.btn_frame, text="🗣️ Speak Word (Enter)", font=ui_font, bg="#00FFCC", fg="#000000", width=20, command=self.trigger_speech)
        self.btn_speak.grid(row=0, column=0, padx=10)

        self.btn_clear = tk.Button(self.btn_frame, text="❌ Clear (Backspace)", font=ui_font, bg="#FF5555", fg="#FFFFFF", width=20, command=self.clear_word)
        self.btn_clear.grid(row=0, column=1, padx=10)

        # Open Camera
        self.cap = cv2.VideoCapture(0)
        
        # Keyboard Bindings
        self.window.bind('<Return>', lambda event: self.trigger_speech())
        self.window.bind('<BackSpace>', lambda event: self.delete_letter())

        # Start Video Loop
        self.update_video()

    def update_video(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmark_data = []
                    for landmark in hand_landmarks.landmark:
                        landmark_data.extend([landmark.x, landmark.y, landmark.z])
                    
                    df = pd.DataFrame([landmark_data], columns=columns)
                    prediction = model.predict(df)[0]
                    
                    # --- Stabilization Logic ---
                    if prediction == self.current_candidate:
                        self.consecutive_frames += 1
                    else:
                        self.current_candidate = prediction
                        self.consecutive_frames = 1
                    
                    # Only accept the letter if it hits our stability threshold
                    if self.consecutive_frames == self.STABILITY_THRESHOLD:
                        if prediction != self.last_prediction:
                            self.current_word += prediction
                            self.last_prediction = prediction
                            self.lbl_word.config(text=f"Current Word: {self.current_word}")
                            
            else:
                # Reset memory and buffers if the hand drops out of frame
                self.last_prediction = "" 
                self.current_candidate = ""
                self.consecutive_frames = 0

            # Convert OpenCV frame to Tkinter format
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            self.photo = ImageTk.PhotoImage(image=pil_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Loop this function every 15 milliseconds
        self.window.after(15, self.update_video)

    def trigger_speech(self):
        if self.current_word:
            print(f"Speaking: {self.current_word}")
            threading.Thread(target=speak_text, args=(self.current_word,)).start()
            self.current_word = ""
            self.lbl_word.config(text="Current Word: ")

    def clear_word(self):
        self.current_word = ""
        self.lbl_word.config(text="Current Word: ")

    def delete_letter(self):
        self.current_word = self.current_word[:-1]
        self.lbl_word.config(text=f"Current Word: {self.current_word}")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# --- 4. Launch the App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root, "Sign Language Translator")
    root.mainloop()