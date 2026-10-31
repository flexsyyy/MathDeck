import os
import time
from dotenv import load_dotenv
import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import pyttsx3

# Initialize Streamlit app
st.set_page_config(layout="wide")

# Define columns in Streamlit layout
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
    finger_status = st.empty()

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the rate of speech

def speak(message):
    engine.say(message)
    engine.runAndWait()

# Configure API key
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Set GEMINI_API_KEY in the .env file and restart.")
    st.stop()
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-flash-lite-latest')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set frame width
cap.set(4, 720)   # Set frame height

# Check if camera is opened successfully
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    speak("Camera not detected")
    st.stop()

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        return fingersUp(lmList), lmList
    return None

def fingersUp(lmList):
    # cvzone 2.0's thumb test assumes a particular hand, so it flips with the hand used.
    # Thumb: tip farther from the pinky base than the joint below it, works for either hand.
    def dist(a, b):
        return np.hypot(lmList[a][0] - lmList[b][0], lmList[a][1] - lmList[b][1])
    fingers = [1 if dist(4, 17) > dist(3, 17) else 0]
    for tip in (8, 12, 16, 20):
        fingers.append(1 if lmList[tip][1] < lmList[tip - 2][1] else 0)
    return fingers

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
        prev_pos = current_pos
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
        prev_pos = None  # Reset previous position when the canvas is cleared
    else:
        prev_pos = None  # Any non-drawing gesture lifts the pen, so the next stroke starts fresh

    return prev_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        try:
            response = model.generate_content(
                ["Solve this math problem. Reply with just the final answer and one short line of working.", pil_image],
                request_options={"timeout": 30},
            )
            return response.text
        except Exception as e:
            print(f"Error interacting with AI model: {e}")
            return "Error: Unable to process request."
    return ""

def is_camera_covered(img, threshold=20):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute average brightness
    avg_brightness = np.mean(gray)
    # Check if brightness is below the threshold
    return avg_brightness < threshold

prev_pos = None
canvas = None
output_text = ""  # Initialize the output_text variable
was_covered = False
last_ai_call = 0

while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to capture image from webcam")
        speak("Camera not detected")
        break

    # Check if camera is covered
    if is_camera_covered(img):
        if not was_covered:
            speak("Camera is covered")
        was_covered = True
        output_text = ""  # Ensure output_text is reset when camera is covered
        continue
    was_covered = False  # Skip the rest of the processing if the camera is covered

    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        if fingers == [1, 1, 1, 1, 0] and time.time() - last_ai_call > 5:
            last_ai_call = time.time()
            output_text = sendToAI(model, canvas, fingers) or output_text
    else:
        prev_pos = None  # Hand left the frame: lift the pen

    image_combined = cv2.add(img, canvas)
    FRAME_WINDOW.image(image_combined, channels="BGR")
    finger_status.caption(f"Fingers (thumb..pinky): {info[0] if info else 'no hand'}")

    if output_text:
        output_text_area.text(output_text)

# Release resources
cap.release()
cv2.destroyAllWindows()
