# Liraries
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
import seaborn as sns
from sklearn import linear_model
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
import numpy as np

#%%
# ----------------------------------------------- #
# --------------- Create Fuctions --------------- #
# ----------------------------------------------- #
# Return a sample dataset from github
def sample_data():
    url = 'https://raw.githubusercontent.com/juanmartinsantos/RegressionModelsApp/main/docs/SampleDataset.csv'
    res = requests.get(url, allow_redirects=True)
    with open('Data1.csv','wb') as file:
        file.write(res.content)
        data = pd.read_csv('Data1.csv', sep=",", decimal=',')
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

def normalization_data(df):
    normalized_data = preprocessing.normalize(df)
    normalized_data= pd.DataFrame(normalized_data, columns=df.columns)
    return normalized_data

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

def get_show_parameters(X_train, X_test, X, Y):
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

def get_show_parameters_FOLD(X, Y, indx):
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

def show_data(df, name_output):
    parameter_normalize = st.radio("Do you want to normalize?", ("Absolute", "Normalized"))
    st.markdown('**1.1. Glimpse of dataset**')
    st.markdown('The sample dataset is used as the example.')
    if parameter_normalize == "Absolute":
        st.write(df.head(5))
    else: st.write(normalization_data(df).head(5))
    st.write('Dimensionality of the dataset')
    st.info(df.shape)
    
    if not name_output in df.columns:
        st.warning("Invalid output name, stopping execution here")
        st.stop()
    return parameter_normalize

# Plot
def ploting(df, feature):
    plt = sns.displot(df, x=df.columns[feature])
    return st.pyplot(fig=plt)

#%%
# Plots
def ploting_heatmap(df):
    fig, ax = plt.subplots()
    sns.heatmap(round(df.corr(),1), ax=ax, linewidths=.5, cbar_kws={"orientation": "horizontal"}, annot=True, cmap="YlGnBu")
    st.write("Correlation Matrix:")
    st.write(fig)

def ploting_hist(df, feature):
    fig, ax = plt.subplots()
    sns.distplot(df[feature], ax=ax, hist_kws=dict(color='lightsteelblue', edgecolor="black", linewidth=1))
    st.write(fig)

def plot_descriptions(df):
    descp = df.describe()
    st.write(descp)

def general_plot(df, plot_criterion):
        
        if plot_criterion == "-":
            None
    
        elif plot_criterion == "Descriptions":
            plot_descriptions(df)
            
        elif plot_criterion == "Correlation Matrix":
            ploting_heatmap(df)
            
        elif plot_criterion == "Histogram":
            feature_criterion = st.selectbox('Choose a column:', (df.columns.insert(0, "-")))
            if feature_criterion != "-":
                ploting_hist(df, feature= feature_criterion)

#%%
# ----------------------------------------------- #
# ----------------- Add Models ------------------ #
# ----------------------------------------------- #
def add_parameters(model_criterion):
    params = dict()
    
    if model_criterion == '-':
        st.sidebar.text("None")
    
    elif model_criterion == 'k-NN':
        parameter_n_neighbors = st.sidebar.slider('Number of neighbors:', 0, 12, 5, 1)
        parameter_type_algorithm = st.sidebar.radio('Choose a type of algorithm:',('auto', 'ball_tree', 'kd_tree', 'brute'))
        params["parameter_n_neighbors"]=parameter_n_neighbors
        params["parameter_type_algorithm"]=parameter_type_algorithm
        
    elif model_criterion == 'SVR':
        parameter_kernel = st.sidebar.radio('Kernel:',('rbf', 'poly', 'sigmoid'))
        params["parameter_kernel"] = parameter_kernel
        # parameter_gamma = st.sidebar.slider('C:', 0, 10, 1, 1)
        # params["parameter_gamma"] = parameter_gamma
        
    elif model_criterion == 'Linear Regression':
        st.sidebar.text("None")
        
    elif model_criterion == 'Lasso Regression':
        # parameter_max_iter = st.sidebar.slider('Number of Iterations:', 1000, 20000, 1000, 1000)
        parameter_alpha = st.sidebar.slider('Alpha:', 0, 10, 1, 1)
        parameter_selection = st.sidebar.radio('Looping Over Features:', ('cyclic', 'random'))
        parameter_normalize = st.sidebar.radio('Normalize:', ('Yes', 'No'))
        # params["parameter_max_iter"] = parameter_max_iter
        params["parameter_alpha"] = parameter_alpha
        params["parameter_normalize"] = parameter_normalize
        params["parameter_selection"] = parameter_selection
        
    elif model_criterion == 'Random Forest':
        parameter_n_estimators = st.sidebar.slider('Number of estimators:', 0, 500, 100, 20)
        params["parameter_n_estimators"] = parameter_n_estimators
        
    elif model_criterion == 'XGBoost':
        parameter_booster = st.sidebar.radio('booster:', ('tree', 'linear'))
        parameter_objective = st.sidebar.radio('objective:', ('squared', 'squaredlog', 'pseudohuber'))
        parameter_num_boost_round = st.sidebar.slider('num_boost_round:',1,100,10,1)
        parameter_max_depth = st.sidebar.slider('max_depth:',1,20,6,1)
        parameter_num_parallel_tree = st.sidebar.slider('num_parallel_tree:',1,10,1,1)
        params["parameter_booster"] = 'gb' + parameter_booster
        params["parameter_objective"] = 'reg:' + parameter_objective + 'error'
        params["parameter_num_boost_round"] = parameter_num_boost_round
        params["parameter_max_depth"] = parameter_max_depth
        params["parameter_num_parallel_tree"] = parameter_num_parallel_tree
        
    elif model_criterion == 'ELM':
        parameter_n_hidden = st.sidebar.slider('Hidden layer:', 3, 100, 20, 1)
        parameter_func_act = st.sidebar.selectbox('Activation function:',('tanh', 'sine', 'tribas', 'sigmoid', 'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric'))
        params["parameter_n_hidden"]=parameter_n_hidden
        params["parameter_func_act"]=parameter_func_act
        
    elif model_criterion == 'MLPR':
        parameter_hidden_layer_sizes = st.sidebar.slider('Hidden Layer Sizes:', 10, 200, 100, 10)
        parameter_max_iter_MLP = st.sidebar.slider('Max. of Iterations:', 100, 20000, 1000, 100)
        parameter_activation = st.sidebar.radio('Activation function:', ('relu', 'identity', 'logistic', 'tanh'))
        params["parameter_hidden_layer_sizes"] = parameter_hidden_layer_sizes
        params["parameter_max_iter_MLP"] = parameter_max_iter_MLP
        params["parameter_activation"] = parameter_activation
        
    return params

def get_regressor(model_criterion, parameters, train_xgb=None, out_xgb=None):
    
    if model_criterion == 'k-NN':
        rgs = KNeighborsRegressor(n_neighbors=parameters['parameter_n_neighbors'], algorithm=parameters['parameter_type_algorithm'])
        
    elif model_criterion == 'SVR':
        # rgs = SVR(kernel= parameters['parameter_kernel'], gamma=parameters['parameter_gamma'], verbose=False)
        rgs = SVR(kernel= parameters['parameter_kernel'], verbose=False)
        
    elif model_criterion == 'Linear Regression':
        rgs = LinearRegression()
    
    elif model_criterion == 'Lasso Regression':
        rgs = linear_model.Lasso(alpha= parameters['parameter_alpha'],normalize=parameters['parameter_normalize'],
                                 selection=parameters['parameter_selection'],random_state= int(parameter_random_state))
    
    elif model_criterion == 'Random Forest':
        rgs = RandomForestRegressor(n_estimators = parameters['parameter_n_estimators'], random_state = int(parameter_random_state))
    
    elif model_criterion == 'XGBoost':
        rgs = {"booster":parameters["parameter_booster"], "objective": parameters["parameter_objective"], 
                              "max_depth": int(parameters["parameter_max_depth"]), 
                              "num_parallel_tree":int(parameters["parameter_num_parallel_tree"])}
        
    elif model_criterion == 'ELM':
        rgs = ELMRegressor(n_hidden= parameters['parameter_n_hidden'], activation_func= parameters['parameter_func_act'], random_state = int(parameter_random_state))
        
    elif model_criterion == 'MLPR':
        rgs = MLPRegressor(hidden_layer_sizes=parameters['parameter_hidden_layer_sizes'], activation=parameters['parameter_activation'],
                           max_iter=parameters['parameter_max_iter_MLP'], random_state=int(parameter_random_state))
        
    return rgs

#%%
# ----------------------------------------------- #
# ---------------- Create Models ---------------- #
# ----------------------------------------------- #
def build_model(df, parameters):
    # Move the output variable to the end
    out = df[name_output]
    df = df.drop(name_output, axis=1)
    df[name_output] = out
    
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y ("output")

    # ----- Data splitting ----- #
    if train_criterion == "Data-split":
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100, random_state= int(parameter_random_state))
        
        # Show Method Parameters
        get_show_parameters(X_train, X_test, X, Y)
            
        # Set model    
        Y_pred_train = get_regressor(model_criterion, parameters)
        
        # Model
        if model_criterion == 'XGBoost': 
            train_matrix = xgb.DMatrix(X_train, label= Y_train)
            Y_pred_train = xgb.train(params=Y_pred_train, dtrain= train_matrix, num_boost_round= int(parameters["parameter_num_boost_round"]))
            X_test = xgb.DMatrix(X_test)
        
        elif model_criterion == 'ELM':
            Y_pred_train.fit(np.array(X_train), np.array(Y_train))
            X_test = np.array(X_test)
        
        else:
            Y_pred_train.fit(X_train, Y_train)
        
        # Predictions
        Y_pred_test= Y_pred_train.predict(X_test)
        
    # ----- K-fold ----- #
    else:
        Y_test=Y
        # Cross-validation are created
        kf = KFold(n_splits=int(split_size), random_state=int(parameter_random_state), shuffle=True)
        indx = list(kf.split(X))
        
        # Show Method Parameters
        get_show_parameters_FOLD(X, Y, indx)
        
        # Predictions set are created
        Y_pred_test = pd.DataFrame(index=range(len(X)), columns=['predictions'])
        
        for folds in range(len(indx)):
            # Select partition indexes
            train_fold = X.iloc[indx[folds][0]]
            output_train_fold = Y.iloc[indx[folds][0]]
            test_fold = X.iloc[indx[folds][1]]      
            
            # Set model
            Y_pred_train = get_regressor(model_criterion, parameters)
            
            # Model
            if model_criterion == 'XGBoost':
                train_matrix = xgb.DMatrix(train_fold, label= output_train_fold)
                Y_pred_train = xgb.train(params=Y_pred_train, dtrain=train_matrix, num_boost_round= int(parameters["parameter_num_boost_round"]))
                test_fold= xgb.DMatrix(test_fold)
                
            elif model_criterion == 'ELM':
                Y_pred_train.fit(np.array(train_fold), np.array(output_train_fold))
                test_fold = np.array(test_fold)
                
            else: 
                Y_pred_train.fit(train_fold, output_train_fold)
            
            # Predictions
            Y_pred_test.iloc[indx[folds][1]]= Y_pred_train.predict(test_fold).reshape(-1,1)
    
    st.subheader('2. Model Performance')
    st.write(parameter_criterion + ' value:')
    st.info(round(reg_metrics(real=Y_test, predictions=Y_pred_test, error=parameter_criterion), 2))

#%% 
# Predictions on unseen data

def get_predict_unseen(df, df_unseen, parameters):
    # Move the out variable to the end
    out = df[name_output]
    df = df.drop(name_output, axis=1)
    df[name_output] = out
    
    X_train = df.iloc[:,:-1] # Using all column except for the last column as X
    Y_train = df.iloc[:,-1] # Selecting the last column as Y
    
    # Set model  
    Y_pred_train = get_regressor(model_criterion, parameters)
    
    # Model
    if model_criterion == 'XGBoost':
        train_matrix = xgb.DMatrix(X_train, label= Y_train)
        Y_pred_train = xgb.train(params=Y_pred_train, dtrain=train_matrix, num_boost_round= int(parameters["parameter_num_boost_round"]))
        df_unseen= xgb.DMatrix(df_unseen)
        
    elif model_criterion == 'ELM':
        Y_pred_train.fit(np.array(X_train), np.array(Y_train))
        df_unseen = np.array(df_unseen)
        
    else: 
        Y_pred_train.fit(X_train, Y_train)
        
    # Predictions
    Y_pred_test= Y_pred_train.predict(df_unseen)
    
    return Y_pred_test

#%%
# ----------------------------------------------- #
# --------------- Create Interfaz --------------- #
# ----------------------------------------------- #
st.write("""
# Regression Models App
Create your model
""")

# ------ Insert the logo 
st.sidebar.image('https://raw.githubusercontent.com/juanmartinsantos/RegressionModelsApp/main/docs/logo.png', use_column_width=True)


# ------ Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your training dataset'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    # Create link to download CSV file
    tmp_download_link = download_link(sample_data(), 'SampleDataset.csv', 'Example CSV input file')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
    # Command of data criterion
    data_criterion = st.sidebar.radio('Would you like a sample dataset?', ('No','Yes'))
    name_output = st.sidebar.text_input('Enter the name of the output variable', ('output'))

# ------ Control on image
if uploaded_file is None and data_criterion == 'No': st.image('https://raw.githubusercontent.com/juanmartinsantos/RegressionModelsApp/main/docs/loading.png', use_column_width=False)

# ------ Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Training Parameters'):
    parameter_random_state = st.sidebar.text_input('Seed number (random_state):', value= '1234')
    train_criterion = st.sidebar.radio('Choose a type of training:', ('k-fold', 'Data-split'))
    split_size = type_training(train_criterion)

with st.sidebar.subheader('3. Choose a Regression Algorithm'):
    model_criterion = st.sidebar.selectbox('Models:', ('-','k-NN', 'SVR', 'ELM', 'Linear Regression', 'Lasso Regression', 'Random Forest', 'XGBoost', 'MLPR'))

# ------ Models
with st.sidebar.subheader('4. Set Model Parameters'):
    parameters = add_parameters(model_criterion)

# ------ Metrics
with st.sidebar.subheader('5. Set Performance Metrics'):
    parameter_criterion = st.sidebar.selectbox('Choose a performance metric:',('RMSE', 'MSE', 'MAE', 'r2'))

# ------ For dataset unseen
with st.sidebar.subheader('6. Do you want to predict on an unseen dataset?'):
    make_criterion = st.sidebar.radio('Method:',('No', 'Yes'))
if make_criterion == 'Yes':
    with st.sidebar.header('-- Upload your test dataset'):
        uploaded_file_test = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], key=(123))

st.sidebar.write("Developed by [Juan Martín](https://linktr.ee/juanmartinwebs)")

# ----------------------------------------------- #
# ------------------ Main panel ----------------- #
# ----------------------------------------------- #
# Subtitle
if uploaded_file is not None or data_criterion == 'Yes': st.subheader('1. Dataset')

# Displays the dataset
if uploaded_file is not None:    
    # LOAD A DATASET 
    df = pd.read_csv(uploaded_file, sep=";")
    norm= show_data(df, name_output)
        
    if norm == "Normalized":
        df=normalization_data(df)
    
    # Ploting
    if model_criterion == "-":
        plot_criterion = st.selectbox('Explore data:', ("-", "Descriptions", "Correlation Matrix", "Histogram"))
        general_plot(df, plot_criterion)
    
    if model_criterion != "-":
        build_model(df, parameters)
    
else:
    # LOAD A DATASET 
    if data_criterion == 'Yes':
        df = sample_data()
        norm= show_data(df, name_output)
        
        if norm == "Normalized":
            df=normalization_data(df)
        
        # Ploting
        if model_criterion == "-":
            plot_criterion = st.selectbox('Explore data:', ("-", "Descriptions", "Correlation Matrix", "Histogram"))
            general_plot(df, plot_criterion)
        
        if model_criterion != "-":
            build_model(df, parameters)


# --- Prediction on unseen dataset
if make_criterion == 'Yes' and uploaded_file_test is not None:
    st.markdown('**2. Predictions**:')
    df_unseen = pd.read_csv(uploaded_file_test, sep = ';')
    st.markdown('**2.1. Glimpse of predictions dataset**')
    st.write(df_unseen.head(5))
    
    if st.button('Run'):
        pred = pd.DataFrame(get_predict_unseen(df, df_unseen, parameters))
        st.download_button(label= 'Download', data= pred.to_csv(sep=';', index = False, header=False), file_name= model_criterion + '_predictions.csv')
        
        # st.balloons()
