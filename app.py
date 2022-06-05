import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Layout
st.set_page_config(page_title= 'AutoML', layout= 'wide', page_icon= '📈')

# Model Building
def build_model(df):
  X= df.iloc[:, :-1] 
  Y= df.iloc[:, -1]
  
  st.markdown('**1.2 Shape**')
  st.write('Independent Variables aka/ X')
  st.info(X.shape)
  st.write('Dependent Variable aka/ Y')
  st.info(Y.shape)
  
  st.markdown('**1.3 Variables**')
  st.write('X (shown upto 20 values)')
  st.info(list(X.columns[:20]))
  st.write('Y')
  st.info(Y.name)
  
  X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= split_size, random_state= seed_number)
  reg= LazyRegressor(verbose= 0, ignore_warnings= False, custom_metric= None)
  models_train, predictions_train= reg.fit(X_train, X_train, Y_train, Y_train)
  models_test, predictions_test= reg.fit(X_train, X_test, Y_train, Y_test)
  
  st.subheader('2. Model Metrics')
  
  st.write('Training Set')
  st.write(predictions_train)
  st.markdown(filedownload(predictions_train, 'Training Metrics'), unsafe_allow_html= True)
  
  st.write('Test Set')
  st.write(predictions_test)
  st.markdown(filedownload(predictions_test, 'Test Metrics'), unsafe_allow_html= True)
  
  st.subheader('3. Plots @ Model Performance')
  
  with st.markdown('**R-squared**'):
    
    # Portrait Plot
    predictions_test['R-Squared']= [0 if i< 0 else i for i in predictions_test['R-Squared']]
    plt.figure(figsize= (3, 9))
    sns.set_theme(style= 'whitegrid')
    ax1= sns.barplot(y= predictions_test.index, x= 'R-Squared', data= predictions_test)
    ax1.set(xlim= (0, 1))
  st.markdown(imagedownload(plt, 'R^2 Plot [Portrait]'), unsafe_allow_html= True)
  
    # Landscape Plot
  plt.figure(figsize= (9, 3))
  sns.set_theme(style= 'whitegrid')
  ax1= sns.barplot(x= predictions_test.index, y= 'R-Squared', data= predictions_test)
  ax1.set(ylim= (0, 1))
  plt.xticks(rotation= 90)
  st.pyplot(plt)
  st.markdown(imagedownload(plt, 'R^2 Plot [Landscape]'), unsafe_allow_html= True) 
    
  with st.markdown('**Root Mean Squared Error**'):
    
    # Portrait Plot
    predictions_test['RMSE']= [50 if i> 50 else i for i in predictions_test['RMSE']]
    plt.figure(figsize= (3, 9))
    sns.set_theme(style= 'whitegrid')
    ax2= sns.barplot(y= predictions_test.index, x= 'RMSE', data= predictions_test)
  st.markdown(imagedownload(plt, 'RMSE Plot [Portrait]'), unsafe_allow_html= True)
  
    # Landscape Plot
  plt.figure(figsize= (9, 3))
  sns.set_theme(style= 'whitegrid')
  ax2= sns.barplot(x= predictions_test.index, y='RMSE', data= predictions_test)
  plt.xticks(rotation= 90)
  st.pyplot(plt)
  st.markdown(imagedownload(plt, 'RMSE Plot [Landscape]'), unsafe_allow_html= True)
  
  with st.markdown('**Processing Time**'):
    
    # Portrait Plot
    predictions_test['Time Taken']= [0 if i< 0 else i for i in predictions_test['Time Taken']]
    plt.figure(figsize= (3, 9))
    sns.set_theme(style= 'whitegrid')
    ax3= sns.barplot(y= predictions_test.index, x= 'Time Taken', data= predictions_test)
  st.markdown(imagedownload(plt, 'Processing-Time Plot [Portrait]'), unsafe_allow_html= True)
  
    # Landscape Plot
  plt.figure(figsize= (9, 3))
  sns.set_theme(style= 'whitegrid')
  ax3= sns.barplot(x= predictions_test.index, y= 'Time Taken', data= predictions_test)  
  plt.xticks(rotation= 90)
  st.pyplot(plt)
  st.markdown(imagedownload(plt, 'Processing-Time Plot [Landscape]'), unsafe_allow_html= True)

  
# Download data as CSV
def filedownload(df, filename):
  csv= df.to_csv(index= False)
  b64= base64.b64encode(csv.encode()).decode() # string <-> byte conversion
  href= f'<a href="data:file/csv;base64,{b64}" download= {filename}>Download {filename} </a>'
  return href


#Download plots as PDF
def imagedownload(plt, filename):
  s= io.BytesIO()
  plt.savefig(s, format= 'pdf', bbox_inches= 'tight')
  plt.close()
  b64= base64.b64encode(s.getvalue()).decode() # string <-> byte conversion
  href= f'<a href="data:image/png;base64,{b64}" download= {filename}>Download {filename} </a>'
  return href


st.write("""
         # Automated Machine Learning~ AutoML
         
         Input a dataset and discover the best Supervised Learning Algorithm for your use case!   
         
         📝: Large datasets require a couple of minutes to build on.  
         """)


with st.sidebar.header('1. Upload Data'):
  uploaded_file= st.sidebar.file_uploader('Upload your input CSV file', type=['csv'])
  st.sidebar.markdown("""
  [Example CSV input file](https://raw.githubusercontent.com/kunal-bhar/autoML-app/main/delaney_solubility_with_descriptors.csv)
                      """)
  
  
with st.sidebar.header('2. Set Parameters'):
  split_size= st.sidebar.slider('Select Training Set Size (%)', 10, 90, 80, 5)
  seed_number= st.sidebar.slider('Set Random Seed Number', 1, 100, 42, 1)
    

st.subheader('1. Dataset')

if uploaded_file is not None:
  df= pd.read_csv(uploaded_file)
  st.markdown('**1.1 Peek**')
  st.write(df)
  build_model(df)
else:
  st.info('Use the sidebar to upload CSV files.')
  if st.button('Example Input File'):
    
    # Boston Housing Data
    boston= load_boston()
    X= pd.DataFrame(boston.data, columns= boston.feature_names).loc[:250]
    Y= pd.Series(boston.target, name= 'response').loc[:250]
    df= pd.concat([X, Y], axis= 1)
    
    st.markdown('The Boston Housing Dataset is used as an example.')
    st.write(df.head(5)) 
    
    build_model(df)


      
    
  
  