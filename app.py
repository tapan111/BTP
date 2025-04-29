import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
import time

# Load your models
model = tf.keras.models.load_model('isl_bilstmonly.h5')
model1 = tf.keras.models.load_model('isl_cnnlstm.h5')
transformer_model = tf.keras.models.load_model('isl_trans.h5')

actions = np.array(['','I','You','Love','Hello','Namaste', 'Bye', 
                    'Thanks', 'Welcome', 'Indian', 'Good Morning','Good Afternoon', 
                    'Good night','Sorry','Please','Car','Food','Water','Today',
                    'Tomorrow','Time','Family','Mother','Father','Tree','House','Beautiful',
                    'Yes','No','Deaf'])

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []  # List of (sign, timestamp)
        self.threshold = 0.3
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.ensemble_pred = np.zeros(len(actions))
        self.final_pred_class = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        # Make detections
        image, results = mediapipe_detection(img, self.holistic)
        draw_styled_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        if len(self.sequence) == 30:
            lstm_pred = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            cnn_lstm_pred = model1.predict(np.expand_dims(self.sequence, axis=0))[0]
            transformer_pred = transformer_model.predict(np.expand_dims(self.sequence, axis=0))[0]
            self.ensemble_pred = (0.2 * lstm_pred + 0.2 * cnn_lstm_pred + 0.6 * transformer_pred)
            self.final_pred_class = np.argmax(self.ensemble_pred)
            self.sequence = []

            # Add sign if above threshold
            if self.ensemble_pred[self.final_pred_class] > self.threshold:
                sign = actions[self.final_pred_class]
                # Only add if not already present or if re-recognized
                if not any(s == sign for s, _ in self.sentence):
                    self.sentence.append((sign, current_time))
                else:
                    # Update timestamp if sign is recognized again
                    self.sentence = [(s, t) if s != sign else (s, current_time) for s, t in self.sentence]

        # Remove signs older than 5 seconds
        self.sentence = [(s, t) for (s, t) in self.sentence if current_time - t < 5]

        # Prepare sentence for display (last 3 signs, most recent last)
        display_sentence = [s for (s, _) in self.sentence][-3:]

        # Draw sentence
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(display_sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image

st.title("Real-time Indian Sign Language Recognition")
st.write("Show your sign in front of the webcam. Each recognized sign will appear above the video for 5 seconds.")

webrtc_streamer(
    key="key",
    video_processor_factory=SignLanguageTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
