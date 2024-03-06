import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title='Predict',
    page_icon= '',
    layout='wide'
)

st.cache_resource(show_spinner='Model Loading')
def load_logistic_pipeline():
    pipeline = joblib.load('./Models/completed_model.joblib')
    return pipeline
    

st.cache_resource(show_spinner='Model Loading')
def load_svc_pipeline():
    pipeline = joblib.load('./Models/SVC_completed_model.joblib')
    return pipeline
    

def select_model():
    
    col1, col2 = st.columns(2)

    with col1:
    
      st.selectbox('Select a Model', options=['Logistic Classifier', 'SVC'], key = 'selected_model')

    with col2:
        pass
    
    if st.session_state['selected_model'] == 'Logistic Classifier':
      pipeline =  load_logistic_pipeline()
    else:
      pipeline = load_svc_pipeline()

    encoder  = joblib.load('./Models/encoder.joblib')    

    return pipeline, encoder

def make_prediction(pipeline, encoder):
   gender = st.session_state['gender']
   SeniorCitizen = st.session_state['SeniorCitizen']
   Partner = st.session_state['Partner']
   Dependents = st.session_state['Dependents']
   tenure = st.session_state['tenure']
   PhoneService = st.session_state['Phone_Service']
   MultipleLines = st.session_state['Multiple_Lines']
   InternetService = st.session_state['Internet_Service']
   OnlineSecurity = st.session_state['Online_Security']
   OnlineBackup = st.session_state['Online_Backup']
   DeviceProtection = st.session_state['Device_Protection']
   TechSupport = st.session_state['Tech_Support']
   StreamingTV = st.session_state['Streaming_TV']
   StreamingMovies = st.session_state['Streaming_Movies']
   Contract = st.session_state['Contract']
   PaperlessBilling = st.session_state['Paperless_Billing']
   MonthlyCharges = st.session_state['Monthly_Charges']
   TotalCharges = st.session_state['Total_Charges ']
   PaymentMethod = st.session_state['Payment_Method']
   
   
  

   columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
   'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
   'MonthlyCharges', 'TotalCharges']

   data = [[gender, SeniorCitizen, Partner, Dependents, tenure,
    PhoneService, MultipleLines, InternetService, OnlineSecurity,
   OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
   StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
   MonthlyCharges, TotalCharges ] ]

   df =pd.DataFrame(data, columns=columns)

   prediction = pipeline.predict(df)
   st.session_state['prediction'] = prediction

   return prediction

  

st.title('Predicting Customer Churn')

def exhibit_form(): 

  with st.form('input_feature'):

    pipeline, encoder = select_model()

    col1, col2, col3 = st.columns(3)

    with col1:
      st.write('### Personal Data')
      st.selectbox('Select gender ', options= [ 'Male', 'Female'], key='gender')
      st.selectbox('Do you have a partner', options= ['Yes', 'No'], key='Partner')
      st.selectbox('Do you have dependents', options= ['Yes', 'No' ], key='Dependents')
      st.selectbox('Are you a Senior Citizen', options= ['Yes', 'No'])


    with col2:
      st.write('### Billing Information')
      st.selectbox('Select the contract type', options= ['Month-to-month', 'One year', 'Two year'], key='contract')
      st.selectbox('Paperless Billing', options= ['Yes', 'No'], key= 'Paperless_Billing')
      st.number_input('Enter Monthly Charges ', min_value= 1, max_value= 5689, step=1, key= 'Monthly_Income' )
      st.number_input('Enter Total Charges ', min_value= 1, max_value= 5689, step=1, key='Total_Charges' )
      st.selectbox('Select the Payment Method', options= ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card'], key='Payment_Method')

    with col3:
      st.write('### Services Provided')
      st.number_input('Enter Tenure ', min_value= 1, max_value= 5689, step=1, key='tenure')
      st.selectbox('Select Phone Services', options= ['Yes', 'No'], key='Phone_services')
      st.selectbox('Do you have Multiple Lines', options= ['Yes', 'No'], key='Multiple_Lines')
      st.selectbox('Select Internet Service option', options= ['DSL', 'Fiber optic', 'No'], key='Internet_Service ')
      st.selectbox('Is Online Security available', options= ['Yes', 'No'], key='Online_Security')
      st.selectbox('Is Online backup available', options= ['Yes', 'No'], key='Online_backup')
      st.selectbox('Is Device Protection available', options= ['Yes', 'No'], key='Device_Protection')
      st.selectbox('Is Tech Support available', options= ['Yes', 'No'], key='Tech_Support')
      st.selectbox('Is Streaming TV available', options= ['Yes', 'No'], key= 'Streaming_TV')
      st.selectbox('Is Streaming Movies available', options= ['Yes', 'No'], key='Streaming_Movies')

      st.form_submit_button('Submit', on_click = make_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder))

if __name__ =='__main__':
  st.title('Make a prediction')
  select_model()
  #exhibit_form()

  #final_prediction = st.session_state['prediction']

  #st.markdown(f' ### {final_prediction}')

  st.write(st.session_state)

    
    