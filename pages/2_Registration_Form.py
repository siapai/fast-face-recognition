import streamlit as st
from Home import face_real_rec as face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import shortuuid

if st.session_state['authentication_status']:

    st.subheader('Registration Form')

    registration_form = face_rec.RegistrationForm()

    # Step1: Collect person name and role
    # form
    person_id = st.text_input(label='ID', placeholder='User ID')
    person_name = st.text_input(label='Name', placeholder='First & Last Name')


    def registration_frame_callback(frame):
        img = frame.to_ndarray(format='bgr24')
        reg_img, embedding = registration_form.get_embedding(img)
        if embedding is not None:
            with open(f'{person_id}.txt', mode='ab') as f:
                np.savetxt(f, embedding)
        return av.VideoFrame.from_ndarray(reg_img, format='bgr24')


    webrtc_streamer(key='registration', video_frame_callback=registration_frame_callback)

    # Step3: Save the data in redis database

    if st.button('Submit'):
        success, message = registration_form.save_data_to_redis(person_name, person_id)
        if success:
            st.success(f'{person_name} registered successfully')
        else:
            if message == 'invalid_name':
                st.error('Person ID or Name is not provided')
            elif message == 'embedding_not_exist':
                st.error('Embedding source file is not exist. Please refresh and execute again.')
else:
    st.warning('You are not authenticated')
