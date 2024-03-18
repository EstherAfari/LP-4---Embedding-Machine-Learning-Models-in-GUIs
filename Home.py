import streamlit as st
import pandas as pd
import numpy as np
import yaml

st.set_page_config(
    page_title = 'Home',
    page_icon= ':)',
    layout= 'wide'
    )

st.title('Welcome to the Vodafone Customer Churn Prediction App')
# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def save_config():
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file)

def authenticate(username, password):
    if username in config['credentials']['usernames']:
        stored_password = config['credentials']['usernames'][username]['password']
        if password == stored_password:
            return True
    return False


# Check if the user is logged in
if 'name' not in st.session_state:
    st.sidebar.title("Login/Create Account")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if authenticate(username, password):
            st.session_state["name"] = username
            st.sidebar.success("Login successful.")
        else:
            st.sidebar.error("Invalid username or password. Please try again.")
    if st.sidebar.button("Create Account"):
        st.sidebar.success("Please enter your full name below to create an account.")
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        if new_username in config['credentials']['usernames']:
            st.sidebar.error("Username already exists. Please choose a different username.")
        else:
            config['credentials']['usernames'][new_username] = {'email': '', 'logged_in': False, 'name': new_username, 'password': new_password}
            save_config()
            st.sidebar.success("Account created successfully. You can now log in.")
    
    st.warning("You need to log in or create an account to access the data.")
else:
    # Authenticate user login
    username = st.session_state['name']
    password = st.text_input("Password", type="password")