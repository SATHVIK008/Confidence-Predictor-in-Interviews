import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from tempfile import NamedTemporaryFile
import pandas as pd

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('final_model_weights.hdf5')

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model()
face_detector = load_face_detector()

# Define the function to predict emotion percentages
def predict_emotion_percentages(video_path, num_frames_to_extract=30):
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    video = cv2.VideoCapture(video_path)
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames_to_extract)

    frames_to_predict = []
    frame_number = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 3)

            for x, y, w, h in faces:
                sub_face_img = gray[y: y + h, x: x + w]
                resized = cv2.resize(sub_face_img, (48, 48))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                frames_to_predict.append(reshaped)

            if len(frames_to_predict) >= num_frames_to_extract:
                break

        frame_number += 1

    predictions = []
    for frame_data in frames_to_predict:
        result = model.predict(frame_data)
        predictions.append(result)

    # Aggregate results and calculate percentages
    avg_predictions = np.mean(predictions, axis=0)[0]  # Mean across all frames for each class
    percentage_dict = {class_names[i]: avg_predictions[i] * 100 for i in range(len(class_names))}

    confidence_percentage = (
        percentage_dict["Happy"] + percentage_dict["Neutral"] + 
        percentage_dict["Surprise"] + percentage_dict["Angry"] 
    )
    stress_percentage = percentage_dict["Disgust"] +  percentage_dict["Fear"] + percentage_dict["Sad"]

    video.release()
    return confidence_percentage, stress_percentage

# Streamlit app layout
st.title("Interview Performance Assessment")
st.write("Upload a video file to analyze emotion percentages for confidence and stress.")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(uploaded_file.read())
        temp_video_path = temp_video_file.name

    # Display video player
    st.video(temp_video_path)

    # Run emotion analysis
    st.write("Analyzing emotions in the video...")
    confidence_percentage, stress_percentage = predict_emotion_percentages(temp_video_path)

    # Display results
    st.subheader("Analysis Results")
    st.write(f"**Confidence Percentage:** {confidence_percentage:.2f}%")
    st.write(f"**Stress Percentage:** {stress_percentage:.2f}%")

    confidence_stress_df = pd.DataFrame({
        "Category": ["Confidence", "Stress"],
        "Percentage": [confidence_percentage, stress_percentage]
    })
    confidence_stress_df = confidence_stress_df.set_index("Category")
    st.bar_chart(confidence_stress_df)