from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
from skimage.color import rgb2gray
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import streamlit as st
import av

face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
model = load_model('model_gender.h5')
gender_dict = {0 : 'Man', 1 : 'Woman'}

st.title("Gender identification App")

class VideoProcessor:
    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")

        #faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)
        face, confidence = cv.detect_face(frame)
        
        # loop through detected faces
        for idx, f in enumerate(face):

        # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

        # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
            face_crop = np.copy(frame[startY:endY,startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

        # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96,96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0) 
        
        # apply gender detection on face
            face_crop  = rgb2gray(face_crop )
            pred = model.predict(face_crop)[0] 
            label = gender_dict[pred[0].round()]
        
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            
        # write label above face rectangle
            cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(frame, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                    )
    )
