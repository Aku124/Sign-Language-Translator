import cv2
import mediapipe as mp
import csv
import os

# --- 1. Configuration ---
NUM_FRAMES = 300  # How many frames to capture per burst
DATASET_FILE = 'custom_dataset.csv'

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create CSV Headers if the file doesn't exist yet
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

def capture_burst(label):
    cap = cv2.VideoCapture(0)
    print(f"\n--- GET READY FOR LETTER: {label} ---")
    print("Press 'S' to start the burst capture. Press 'Q' to quit.")

    frames_collected = 0
    capturing = False

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw the skeleton so you know it sees you
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # UI Instructions on screen
        if not capturing:
            cv2.putText(frame, f"Sign '{label}' and press 'S' to start", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Capturing: {frames_collected}/{NUM_FRAMES}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Burst Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Start capture on 'S'
        if key == ord('s') and not capturing:
            capturing = True
            print("Capturing...")

        # Quit on 'Q'
        if key == ord('q'):
            break

        # The Burst Logic
        if capturing and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                row = [label]
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                
                with open(DATASET_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                frames_collected += 1

        if frames_collected >= NUM_FRAMES:
            print(f"✅ Successfully captured {NUM_FRAMES} frames for '{label}'.")
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Run the Collector ---
if __name__ == "__main__":
    while True:
        target_letter = input("\nEnter the letter you want to record (or type 'exit' to stop): ").strip().upper()
        if target_letter == 'EXIT':
            break
        if len(target_letter) == 1 and target_letter.isalpha():
            capture_burst(target_letter)
        else:
            print("Please enter a single valid letter.")