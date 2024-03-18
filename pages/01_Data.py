import streamlit as st
import pyodbc 
import pandas as pd

st.set_page_config(
    page_title='View Data',
    page_icon= '',
    layout='wide'
)
st.title ('Available Data from Vodafone')

# @st.cache_resource(show_spinner='Connecting to Database...')
# def initialize_connection():
#     connection = pyodbc.connect(
#         "DRIVER={SQL Server};SERVER="
#         + st.secrets["server"]
#         + ";DATABASE="
#         + st.secrets["database"]
#         + ";UID="
#         + st.secrets["uid"]
#         + ";PWD="
#         + st.secrets["pwd"]
#     )

#     return connection

# conn = initialize_connection()

# @st.cache_data()
# def query_database(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         rows = cur.fetchall()

#         df = pd.DataFrame.from_records(data=rows, columns=[column[0]for column in cur.description])
#     return df
# @st.cache_data()

col1, col2 = st.columns(2)
with col1:
        
    st.selectbox('Select the feature type',options=['All features', 'Numeric Features', 'Categorical Features'], key = 'selected_columns')


final_df = pd.read_csv('./dataset.csv')


def select_all_features():
    df = final_df
    df = st.dataframe(df)

    return df

def select_numeric_features():
    df = final_df.select_dtypes(include='number')
    df = st.dataframe(df)
    
    return df

def select_categorical_features():
    df = final_df.select_dtypes(include='object')
    df = st.dataframe(df)
    
    return df


if __name__ =='__main__ ':

    if st.session_state['selected_columns']== 'All features':
        st.dataframe(select_all_features())
        
    elif st.session_state['selected_columns']=='Numeric Features':
        st.dataframe(select_numeric_features())
        
    else :
        st.dataframe(select_categorical_features())
        



    