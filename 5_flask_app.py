from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pandas as pd
import pickle
import pyttsx3
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# --- 1. Global State Variables ---
# We use globals here so the web buttons and the video loop share the same memory
current_word = ""
last_added_letter = ""

# --- 2. Setup Voice Engine (Thread-Safe Fix) ---
def speak_text(text):
    if text.strip():
        # Initialize a fresh engine INSIDE the thread to prevent audio driver crashes
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        # The engine will safely shut down after speaking, ready for the next click

# --- 3. Load the AI Model ---
try:
    with open('gesture_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find 'gesture_model_final.pkl'.")
    model = None

# --- 4. Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

columns = []
for i in range(21):
    columns.extend([f'x{i}', f'y{i}', f'z{i}'])

# --- 5. The Video & Word Builder Generator ---
def generate_frames():
    global current_word, last_added_letter
    cap = cv2.VideoCapture(0)
    
    current_candidate = ""
    consecutive_frames = 0
    STABILITY_THRESHOLD = 10  # Must see same sign for 10 frames

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and model is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmark_data = []
                for landmark in hand_landmarks.landmark:
                    landmark_data.extend([landmark.x, landmark.y, landmark.z])
                
                df = pd.DataFrame([landmark_data], columns=columns)
                prediction = model.predict(df)[0]
                
                # --- Stabilization & String Building Logic ---
                if prediction == current_candidate:
                    consecutive_frames += 1
                else:
                    current_candidate = prediction
                    consecutive_frames = 1
                
                # If stable and it's a NEW letter, add it to the word
                if consecutive_frames >= STABILITY_THRESHOLD:
                    if prediction != last_added_letter:
                        current_word += prediction
                        last_added_letter = prediction
        else:
            # Reset memory if hand leaves the frame
            consecutive_frames = 0
            current_candidate = ""
            last_added_letter = ""

        # Draw the built word dynamically on the screen
        cv2.putText(frame, f"Word: {current_word}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 204), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 6. Web Routes & API Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Button Commands ---
@app.route('/speak', methods=['POST'])
def speak():
    global current_word, last_added_letter
    if current_word:
        print(f"Speaking: {current_word}")
        # Run speech in a separate thread so it doesn't freeze the video
        threading.Thread(target=speak_text, args=(current_word,)).start()
        current_word = "" 
        last_added_letter = ""
    return "OK"

@app.route('/delete', methods=['POST'])
def delete():
    global current_word, last_added_letter
    current_word = current_word[:-1]
    last_added_letter = current_word[-1] if current_word else ""
    return "OK"

@app.route('/clear', methods=['POST'])
def clear():
    global current_word, last_added_letter
    current_word = ""
    last_added_letter = ""
    return "OK"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)