import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

st.set_page_config(page_title="Attendance System")
st.header("Attendance System using Face Recognition")


with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

with st.spinner("Loading Models and Connecting to Redis DB"):
    import face_real_rec

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
elif not authentication_status:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')



