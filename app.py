import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
import os
from matplotlib import pyplot as plt
import time
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Multiply, Permute, Lambda, Activation
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Bidirectional,
                                     MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Flatten)
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MultiHeadAttention

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention

model = tf.keras.models.load_model('isl_bilstmonly.h5')
model1 = tf.keras.models.load_model('isl_cnnlstm.h5')
transformer_model = tf.keras.models.load_model('isl_trans.h5')


actions = np.array(['','I','You','Love','Hello','Namaste', 'Bye', 
                    'Thanks', 'Welcome', 'Indian', 'Good Morning','Good Afternoon', 
                    'Good night','Sorry','Please','Car','Food','Water','Today',
                    'Tomorrow','Time','Family','Mother','Father','Tree','House','Beautiful',
                    'Yes','No','Deaf'])

# MediaPipe setup
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

class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.threshold = 0.3
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Make detection
        image, results = mediapipe_detection(img, self.holistic)

        # Draw styled landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Prediction logic
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)

        if len(self.sequence) == 30:
            input_data = np.expand_dims(self.sequence, axis=0)
            lstm_pred = model.predict(input_data)[0]
            cnn_lstm_pred = model1.predict(input_data)[0]
            transformer_pred = transformer_model.predict(input_data)[0]
            ensemble_pred = (0.2 * lstm_pred + 0.2 * cnn_lstm_pred + 0.6 * transformer_pred)
            final_pred_class = np.argmax(ensemble_pred)

            if ensemble_pred[final_pred_class] > self.threshold:
                if len(self.sentence) == 0 or actions[final_pred_class] != self.sentence[-1]:
                    self.sentence.append(actions[final_pred_class])
            
            if len(self.sentence) > 3:
                self.sentence = self.sentence[-3:]

            self.sequence = []

        # Display the resulting sentence
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(self.sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image

# Streamlit UI
st.title("Real-time Indian Sign Language Recognition")

webrtc_streamer(key="key", video_processor_factory=SignLanguageTransformer)

