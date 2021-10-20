import streamlit as st
import pandas as pd
import requests
import base64
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
#%%
# ----------------------------------------------- #
# --------------- Create Fuctions --------------- #
# ----------------------------------------------- #

# Return a sample dataset from github
def sample_data():
    url = 'https://raw.githubusercontent.com/juanmartinsantos/dataset/main/docs/SampleDataset.csv'
    res = requests.get(url, allow_redirects=True)
    with open('Data1.csv','wb') as file:
        file.write(res.content)
        data = pd.read_csv('Data1.csv', sep=",")
    return data

# Function to change the type of training
def type_training(choosen):
    if choosen == 'Data-split':
        return st.sidebar.slider('Data split ratio (% for Training Set):', 10, 90, 80, 5)
    else:
        return st.sidebar.slider('Number of k-fold:', 3, 10, 5, 1)

# Create a download link
def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def reg_metrics(real, predictions, error):
    if error == "MSE":
        mse_error= sklearn.metrics.mean_squared_error(real, predictions, squared=True)
        return mse_error
    if error == "RMSE":
        rmse_error= sklearn.metrics.mean_squared_error(real, predictions, squared=False)
        return rmse_error
    if error == "r2":
        rsq_error= sklearn.metrics.r2_score(real, predictions)
        return rsq_error
    if error == "MAE":
        mae_error= sklearn.metrics.mean_absolute_error(real, predictions)
        return mae_error

#%%
# ----------------------------------------------- #
# ---------------- Create Models ---------------- #
# ----------------------------------------------- #

def build_model_NN(df, **kwargs):
    # seed(123)
    # Default setup
    setup_rnn = {'n_neighbors':5,'algorithm':"brute"}
    setup_rnn.update(kwargs)
    # Kmeans setup 
    k = setup_rnn['n_neighbors']
    algorithm_ = setup_rnn['algorithm']
    
    # Move the out variable to the end
    out = df[name_output]
    df = df.drop(name_output, axis=1)
    df[name_output] = out
    
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y
    
    if train_criterion == "Data-split":
        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100,  random_state= int(parameter_random_state))
        # Set Method
        st.markdown('**1.2. Dataset splits**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)

        st.markdown('**1.3. Variable details**:')
        st.write('Training features')
        st.info(list(X.columns))
        st.write('Output feature')
        st.info(Y.name)
        
        st.subheader('2. Model Performance')
        # st.markdown('**2.1. Training set**')
        # Model
        Y_pred_train = KNeighborsRegressor(n_neighbors=k, algorithm=algorithm_)
        Y_pred_train.fit(X_train, Y_train)
        # Predictions
        Y_pred_test= Y_pred_train.predict(X_test)
    
    else:
        Y_test=Y
        
        # Cross-validation are created
        kf = KFold(n_splits=int(split_size), random_state=int(parameter_random_state), shuffle=True)
        indx = list(kf.split(X))
        
        st.markdown('**1.2. k-fold cross-validation**')
        st.write('Training fold')
        st.info(len(X.iloc[indx[1][0]]))
        st.write('Test fold')
        st.info(len(X.iloc[indx[1][1]]))

        st.markdown('**1.3. Variable details**:')
        st.write('Training features')
        st.info(list(X.columns))
        st.write('Output feature')
        st.info(Y.name)
        
        
        # Predictions set are created
        Y_pred_test = pd.DataFrame(index=range(len(X)), columns=['predictions'])
        # Set model
        Y_pred_train = KNeighborsRegressor(n_neighbors=k, algorithm=algorithm_)
        
        for folds in range(len(indx)):
            # folds =0
            # Seleccion los idx de las particiones
            train_fold=X.iloc[indx[folds][0]]
            output_train_fold= Y.iloc[indx[folds][0]]
            
            test_fold=X.iloc[indx[folds][1]]      
        
            # Predictions
            Y_pred_train.fit(train_fold, output_train_fold)
            Y_pred_test.iloc[indx[folds][1]]= Y_pred_train.predict(test_fold).reshape(-1,1)
    
    st.write(parameter_criterion + ' value:')
    st.info(round(reg_metrics(real=Y_test, predictions=Y_pred_test, error=parameter_criterion), 2))
    
    st.write('Setup:')
    st.info(setup_rnn)
    
def build_model_RL(df, **kwargs):
    # Move the out variable to the end
    out = df[name_output]
    df = df.drop(name_output, axis=1)
    df[name_output] = out
    
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    if train_criterion == "Data-split":
        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100,  random_state= int(parameter_random_state))
        # Set Method
        st.markdown('**1.2. Dataset splits**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)
    
        st.markdown('**1.3. Variable details**:')
        st.write('Training features')
        st.info(list(X.columns))
        st.write('Output feature')
        st.info(Y.name)
            
        st.subheader('2. Model Performance')
        # st.markdown('**2.1. Training set**')
        # Model
        Y_pred_train = LinearRegression()
        Y_pred_train.fit(X_train, Y_train)
        # Predictions
        Y_pred_test= Y_pred_train.predict(X_test)

    else:
        Y_test=Y
        
        # Cross-validation are created
        kf = KFold(n_splits=int(split_size), random_state=int(parameter_random_state), shuffle=True)
        indx = list(kf.split(X))
        
        st.markdown('**1.2. k-fold cross-validation**')
        st.write('Training fold')
        st.info(len(X.iloc[indx[1][0]]))
        st.write('Test fold')
        st.info(len(X.iloc[indx[1][1]]))

        st.markdown('**1.3. Variable details**:')
        st.write('Training features')
        st.info(list(X.columns))
        st.write('Output feature')
        st.info(Y.name)
        
        
        # Predictions set are created
        Y_pred_test = pd.DataFrame(index=range(len(X)), columns=['predictions'])
        # Set model
        Y_pred_train = LinearRegression()
        
        for folds in range(len(indx)):
            # folds =0
            # Seleccion los idx de las particiones
            train_fold=X.iloc[indx[folds][0]]
            output_train_fold= Y.iloc[indx[folds][0]]
            
            test_fold=X.iloc[indx[folds][1]]      
        
            # Predictions
            Y_pred_train.fit(train_fold, output_train_fold)
            Y_pred_test.iloc[indx[folds][1]]= Y_pred_train.predict(test_fold).reshape(-1,1)
    
    st.write(parameter_criterion + ' value:')
    st.info(round(reg_metrics(real=Y_test, predictions=Y_pred_test, error=parameter_criterion), 2))

def build_model_SVR(df, **kwargs):
    
    setup_rsvr = {'kernel':'linear', 'degree':3, 'gamma':'scale', 'coef0':0.0, 'tol':1e-3}
    setup_rsvr.update(kwargs)
    #parameters
    kernel_ = setup_rsvr['kernel']
    degree_ = setup_rsvr['degree']
    gamma_ = setup_rsvr['gamma']
    coef0_ = setup_rsvr['coef0']
    tol_ = setup_rsvr['tol']
    
    # Move the out variable to the end
    out = df[name_output]
    df = df.drop(name_output, axis=1)
    df[name_output] = out
    
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y


    if train_criterion == "Data-split":
        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100,  random_state= int(parameter_random_state))
        # Set Method
        st.markdown('**1.2. Dataset splits**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)
    
        st.markdown('**1.3. Variable details**:')
        st.write('Training features')
        st.info(list(X.columns))
        st.write('Output feature')
        st.info(Y.name)
            
        st.subheader('2. Model Performance')
        # st.markdown('**2.1. Training set**')
        # Model
        Y_pred_train = SVR(kernel= kernel_, degree=degree_, gamma=gamma_, coef0=coef0_, tol=tol_, verbose=False)
        Y_pred_train.fit(X_train, Y_train)
        # Predictions
        Y_pred_test= Y_pred_train.predict(X_test)

    else:
        Y_test=Y
        
        # Cross-validation are created
        kf = KFold(n_splits=int(split_size), random_state=int(parameter_random_state), shuffle=True)
        indx = list(kf.split(X))
        
        st.markdown('**1.2. k-fold cross-validation**')
        st.write('Training fold')
        st.info(len(X.iloc[indx[1][0]]))
        st.write('Test fold')
        st.info(len(X.iloc[indx[1][1]]))

        st.markdown('**1.3. Variable details**:')
        st.write('Training features')
        st.info(list(X.columns))
        st.write('Output feature')
        st.info(Y.name)
        
        
        # Predictions set are created
        Y_pred_test = pd.DataFrame(index=range(len(X)), columns=['predictions'])
        # Set model
        Y_pred_train = SVR(kernel= kernel_, degree=degree_, gamma=gamma_, coef0=coef0_, tol=tol_, verbose=False)
        
        for folds in range(len(indx)):
            # folds =0
            # Seleccion los idx de las particiones
            train_fold=X.iloc[indx[folds][0]]
            output_train_fold= Y.iloc[indx[folds][0]]
            
            test_fold=X.iloc[indx[folds][1]]      
        
            # Predictions
            Y_pred_train.fit(train_fold, output_train_fold)
            Y_pred_test.iloc[indx[folds][1]]= Y_pred_train.predict(test_fold).reshape(-1,1)
    
    st.write(parameter_criterion + ' value:')
    st.info(round(reg_metrics(real=Y_test, predictions=Y_pred_test, error=parameter_criterion), 2))

def build_model_RF(df, **kwargs):
    
    setup_rrf = {'n_estimators':100, 'random_state':0}
    setup_rrf.update(kwargs)
    #parameters
    n_estimators_ = setup_rrf['n_estimators']
    random_state_ = setup_rrf['random_state']
    
    # Move the out variable to the end
    out = df[name_output]
    df = df.drop(name_output, axis=1)
    df[name_output] = out
    
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y


    if train_criterion == "Data-split":
        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100,  random_state= int(parameter_random_state))
        # Set Method
        st.markdown('**1.2. Dataset splits**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)
    
        st.markdown('**1.3. Variable details**:')
        st.write('Training features')
        st.info(list(X.columns))
        st.write('Output feature')
        st.info(Y.name)
            
        st.subheader('2. Model Performance')
        # st.markdown('**2.1. Training set**')
        # Model
        Y_pred_train = RandomForestRegressor(n_estimators = n_estimators_, random_state = random_state_)
        Y_pred_train.fit(X_train, Y_train)
        # Predictions
        Y_pred_test= Y_pred_train.predict(X_test)

    else:
        Y_test=Y
        
        # Cross-validation are created
        kf = KFold(n_splits=int(split_size), random_state=int(parameter_random_state), shuffle=True)
        indx = list(kf.split(X))
        
        st.markdown('**1.2. k-fold cross-validation**')
        st.write('Training fold')
        st.info(len(X.iloc[indx[1][0]]))
        st.write('Test fold')
        st.info(len(X.iloc[indx[1][1]]))

        st.markdown('**1.3. Variable details**:')
        st.write('Training features')
        st.info(list(X.columns))
        st.write('Output feature')
        st.info(Y.name)
        
        
        # Predictions set are created
        Y_pred_test = pd.DataFrame(index=range(len(X)), columns=['predictions'])
        # Set model
        Y_pred_train = RandomForestRegressor(n_estimators = n_estimators_, random_state = random_state_)
        
        for folds in range(len(indx)):
            # folds =0
            # Seleccion los idx de las particiones
            train_fold=X.iloc[indx[folds][0]]
            output_train_fold= Y.iloc[indx[folds][0]]
            
            test_fold=X.iloc[indx[folds][1]]      
        
            # Predictions
            Y_pred_train.fit(train_fold, output_train_fold)
            Y_pred_test.iloc[indx[folds][1]]= Y_pred_train.predict(test_fold).reshape(-1,1)
    
    st.write(parameter_criterion + ' value:')
    st.info(round(reg_metrics(real=Y_test, predictions=Y_pred_test, error=parameter_criterion), 2))


#%%
# ----------------------------------------------- #
# --------------- Create Interfaz --------------- #
# ----------------------------------------------- #

st.write("""
# Regression models app
This is a first version
""")

# ------ Insert the logo 
st.sidebar.image('https://raw.githubusercontent.com/juanmartinsantos/dataset/main/docs/logo.png', use_column_width=True)

# ------ Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your training dataset'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    # Create link to download CSV file
    tmp_download_link = download_link(sample_data(), 'SampleDataset.csv', 'Example CSV input file')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
    # Command of data criterion
    data_criterion = st.sidebar.radio('Would you like a sample dataset?', ('No','Yes'))
    name_output = st.sidebar.text_input('Enter the name of the output variable', ('output'))

# For dataset unseen
# with st.sidebar.subheader('Do you predict on a dataset unseen?'):
#     make_criterion = st.sidebar.radio('Method:',('No', 'Yes'))
# if make_criterion == 'Yes':
#     with st.sidebar.header('-- Upload your test dataset'):
#         uploaded_file_test = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], key=(123))

# ------ Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Training Parameters'):
    parameter_random_state = st.sidebar.text_input('Seed number (random_state):', value= '1234')
    train_criterion = st.sidebar.radio('Choose a type of training:', ('k-fold', 'Data-split'))
    split_size = type_training(train_criterion)

with st.sidebar.subheader('3. Choose a Regression Algorithm'):
    model_criterion = st.sidebar.radio('Methods:',('k-NN', 'Regression Linear', 'SVR', 'Random Forest'))

# ------ Models
with st.sidebar.subheader('4. Set Model Parameters'):
    if model_criterion == 'k-NN':
        parameter_n_neighbors = st.sidebar.slider('Number of neighbors:', 0, 12, 3, 1)
        parameter_type_algorithm = st.sidebar.radio('Choose a type of algorithm (dafault: auto):',('auto', 'ball_tree', 'kd_tree', 'brute'))
    
    elif model_criterion == 'Regression Linear':
        st.sidebar.text("None")
    
    elif model_criterion == 'SVR':
        parameter_kernel = st.sidebar.radio('Kernel:',('rbf', 'poly', 'sigmoid'))
        parameter_gamma = st.sidebar.radio('Gamma:',('scale', 'auto'))
        
    elif model_criterion == 'Random Forest':
        parameter_n_estimators = st.sidebar.slider('Number of estimators:', 0, 500, 100, 20)

# ------ Metrics
with st.sidebar.subheader('5. Set Performance Metrics'):
    parameter_criterion = st.sidebar.radio('Choose a performance metric:',('RMSE', 'MSE', 'MAE', 'r2'))


# Main panel
# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    # build_model(df)
else:
    # LOAD A DATASET 
    if data_criterion == 'Yes':
        df = sample_data()
        st.markdown('The sample dataset is used as the example.')
        st.write(df.head(5))
        if model_criterion == "k-NN":
            build_model_NN(df, n_neighbors=parameter_n_neighbors, algorithm=parameter_type_algorithm)
            
        elif model_criterion == "Regression Linear":
            build_model_RL(df)
        
        elif model_criterion == "SVR":
            build_model_SVR(df, kernel= parameter_kernel, gamma=parameter_gamma)
        
        elif model_criterion == "Random Forest":
            build_model_RF(df, n_estimators=parameter_n_estimators)






























