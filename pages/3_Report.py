import streamlit as st
from Home import face_real_rec as face_rec
import constants

if st.session_state['authentication_status']:
    st.subheader('Reporting')

    def load_logs(name, end=-1):
        logs_list = face_rec.r.lrange(name, start=0, end=end)
        return logs_list


    tab1, tab2 = st.tabs(['Registered Data', 'Attendance Logs'])
    col1, col2 = st.columns([1, 1])

    with tab1:
        redis_face_db = face_rec.retrieve_data(name=constants.REGISTER_KEY)
        if st.button('Refresh Data'):
            redis_face_db = face_rec.retrieve_data(name=constants.REGISTER_KEY)
        with st.spinner('Retrieving data from redis...'):
            st.dataframe(redis_face_db[['Name', 'User_Id']])

    with tab2:
        if st.button('Refresh Logs'):
            st.write(load_logs(name=constants.LOG_KEY))
        else:
            st.write(load_logs(name=constants.LOG_KEY))
else:
    st.warning('You are not authenticated')
