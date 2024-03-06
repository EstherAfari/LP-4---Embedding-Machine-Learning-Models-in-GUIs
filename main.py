import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title = 'Home',
    page_icon= ':)',
    layout= 'wide'
    )

st.title('Hello World')
st.button('Submit')
st.text_input('Enter your name')

df =pd.read_csv('Churn_Data_Tableau.csv')

st.dataframe(df)

data = pd.DataFrame(
    np.random.randn(20,2),
    columns =['MonthlyCharges','TotalCharges']
)


st.bar_chart(data)
st.line_chart(data)