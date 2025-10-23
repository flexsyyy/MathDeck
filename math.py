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
api_key = "AIzaSyBhDf_4rSKxjB2SxiJUOtEyxgdHh3Jat8o"  # Consider using environment variables for sensitive information
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

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
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

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
    elif fingers == [0, 1, 1, 0, 0]:
        prev_pos = None  # Reset previous position to break the line

    return prev_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        try:
            response = model.generate_content(["Solve this math problem", pil_image])
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

while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to capture image from webcam")
        speak("Camera not detected")
        break

    # Check if camera is covered
    if is_camera_covered(img):
        speak("NIGGA")
        output_text = ""  # Ensure output_text is reset when camera is covered
        continue  # Skip the rest of the processing if the camera is covered

    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

# Release resources
cap.release()
cv2.destroyAllWindows()
