import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load models and actions as before
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
        self.sentence = []
        self.threshold = 0.3
        self.start_time = time.time()
        self.duration = 30  # seconds
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.ensemble_pred = np.zeros(len(actions))
        self.final_pred_class = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        elapsed = time.time() - self.start_time
        remaining = max(0, int(self.duration - elapsed))
        if remaining == 0:
            self.start_time = time.time()  # reset timer

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
            self.ensemble_pred = (lstm_pred + cnn_lstm_pred + transformer_pred) / 3
            self.final_pred_class = np.argmax(self.ensemble_pred)
            self.sequence = []

        # Viz logic
        if self.ensemble_pred[self.final_pred_class] > self.threshold: 
            if len(self.sentence) > 0: 
                if actions[self.final_pred_class] != self.sentence[-1]:
                    self.sentence.append(actions[self.final_pred_class])
            else:
                self.sentence.append(actions[self.final_pred_class])

        if len(self.sentence) > 3: 
            self.sentence = self.sentence[-3:]

        # Draw timer and sentence
        cv2.rectangle(img, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, ' '.join(self.sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Timer: {remaining}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

st.title("Real-time Indian Sign Language Recognition")

webrtc_streamer(
    key="key",
    video_processor_factory=SignLanguageTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
