import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import time
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the model and load the weights
face_exp_model = model_from_json(open("facial_expression_model_structure.json","r",encoding="utf-8").read())
face_exp_model.load_weights('facial_expression_model_weights.h5')

# Function to classify review sentiment
def review_rating(review):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(review)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Main Streamlit app
def main():
    st.title("Emotion Detection and Review Analysis")

    # Button for attention detection
    detect_attention = st.button("Detect Attention")

    # Button for review analysis
    analyze_review = st.button("Analyze Review")

    if detect_attention:
        # Start video capture
        webcam_video_stream = cv2.VideoCapture(0)

        # Timer for attention detection
        start_time = time.time()
        elapsed_time = 0

        # Loop through every frame in the video
        while elapsed_time < 10:  # Detect attention for 10 seconds
            # Get the current frame from the video stream as an image
            ret, current_frame = webcam_video_stream.read()
            
            # Resize the current frame to 1/4 size to process faster
            current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
            
            # Detect all faces in the image
            all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='hog')
            
            # Loop through the face locations
            for current_face_location in all_face_locations:
                # Extract the face from the frame
                top_pos, right_pos, bottom_pos, left_pos = [i*4 for i in current_face_location]
                current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

                # Preprocess input, convert it to an image like as the data in dataset
                current_face_image_gray = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
                current_face_image_gray = cv2.resize(current_face_image_gray, (48, 48))
                img_pixels = image.img_to_array(current_face_image_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255 

                # Do prediction using model
                exp_predictions = face_exp_model.predict(img_pixels) 
                max_index = np.argmax(exp_predictions[0])
                emotion_label = emotions_label[max_index]
                
                # Display emotion label and draw green box around face on main frame
                cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 255, 0), 2)
                cv2.putText(current_frame, emotion_label, (left_pos, top_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display the main frame with emotions and green boxes
            st.image(current_frame, channels="BGR", use_column_width=True)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

        # Release the video stream
        webcam_video_stream.release()

    if analyze_review:
        review = st.text_input("Enter the review about the lecture")
        if review:
            analysis_result = review_rating(review)
            st.write("The review is classified as", analysis_result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
