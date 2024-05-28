import streamlit as st
import constants
from Home import face_real_rec as face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

if st.session_state['authentication_status']:
    st.title('Real-Time Attendance System')

    # Time
    waitTime = 30
    setTime = time.time()
    realTimePred = face_rec.RealTimePred()

    # Real Time Prediction
    with st.spinner('Retrieving data from redis...'):
        redis_face_db = face_rec.retrieve_data(name=constants.REGISTER_KEY)
        st.success('Data loaded successfully')


    def video_frame_callback(frame):  # flipped = img[::-1, :, :]
        global setTime
        img = frame.to_ndarray(format="bgr24")  # 3 dimension numpy array
        # operation perform on array
        pred_img = realTimePred.face_prediction(img, redis_face_db, 'facial_features', None)

        time_now = time.time()
        diff_time = time_now - setTime
        if diff_time >= waitTime:
            realTimePred.save_logs_redis()
            setTime = time.time()  # reset

            print('Save data to redis')

        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


    webrtc_streamer(key="realtime_prediction", video_frame_callback=video_frame_callback)

else:
    st.warning('You are not authenticated')
