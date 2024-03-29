# Liraries
import base64
import requests
# import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.svm import SVR

#%%
# ----------------------------------------------- #
# --------------- Create Fuctions --------------- #
# ----------------------------------------------- #
# Return a sample dataset from github
def sample_data():
    url = 'https://raw.githubusercontent.com/juanmartinsantos/RegressionModelsApp/main/docs/SampleDataset.csv'
    res = requests.get(url, allow_redirects=True)
    with open('SampleDataset.csv','wb') as file:
        file.write(res.content)
        data = pd.read_csv('SampleDataset.csv', sep=",", decimal=',')
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
    scaler = preprocessing.MinMaxScaler()
    names = df.columns
    d = scaler.fit_transform(df)
    normalized_data = pd.DataFrame(d, columns=names)
    return normalized_data

def reg_metrics(real, predictions, error):
    if error == "MSE":
        mse_error = mean_squared_error(real, predictions, squared=True)
        return mse_error
    if error == "RMSE":
        rmse_error = mean_squared_error(real, predictions, squared=False)
        return rmse_error
    if error == "r2":
        rsq_error= r2_score(real, predictions)
        return rsq_error
    if error == "MAE":
        mae_error = mean_absolute_error(real, predictions)
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
    if name_output == 'output':
        st.markdown('The sample dataset is used as the example.')
    
    if df.isnull().values.any():
        st.markdown('*This dataset contained $\b missing values$, which has been $\b removed$*')
        df = df.dropna()
    
    if parameter_normalize == "Absolute":
        # st.dataframe(df)
        st.write(df.head(5))
    else:
        # st.dataframe(normalization_data(df))
        st.write(normalization_data(df).head(5))
    
    st.write('Dimensionality of the dataset')
    st.info(df.shape)
    
    if not name_output in df.columns:
        st.warning("Invalid output name, stopping execution here")
        st.stop()
        
    return parameter_normalize

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

def plot_descriptions(data):
    descp = data.describe()
    st.write(descp)
    
def ploting_boxplot(df, feature):
    fig, ax = plt.subplots()
    sns.boxplot(y=df[feature], ax=ax)
    st.write(f"Box plot for {feature}")
    st.pyplot(fig)

def ploting_scatter(df, x_feature, y_feature):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_feature, y=y_feature, ax=ax)
    st.write(f"Scatter Plot for {x_feature} vs {y_feature}")
    st.pyplot(fig)

def general_plot(df, plot_criterion):
        
    if plot_criterion == "-":
        None
    
    elif plot_criterion == "Descriptions":
        plot_descriptions(df)
            
    elif plot_criterion == "Correlation Matrix":
        ploting_heatmap(df)
            
    elif plot_criterion == "Histogram":
        feature_criterion = st.selectbox('Choose a column for Histogram:', (df.columns.insert(0, "-")))
        if feature_criterion != "-":
            ploting_hist(df, feature=feature_criterion)
                
    elif plot_criterion == "Box Plots":
        feature_criterion = st.selectbox('Choose a column for Box Plot:', (df.columns.insert(0, "-")))
        if feature_criterion != "-":
            ploting_boxplot(df, feature=feature_criterion)
            
    elif plot_criterion == "Scatter Plots":
        st.subheader("Scatter Plots")
        x_feature = st.selectbox('Choose a column for X axis:', df.columns.tolist())
        y_feature = st.selectbox('Choose a column for Y axis:', df.columns.tolist())
        if x_feature and y_feature:
            ploting_scatter(df, x_feature, y_feature)



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
        # params["parameter_max_iter"] = parameter_max_iter
        params["parameter_alpha"] = parameter_alpha
        params["parameter_selection"] = parameter_selection
        
    elif model_criterion == 'Random Forest':
        parameter_n_estimators = st.sidebar.slider('Number of estimators:', 0, 500, 100, 20)
        params["parameter_n_estimators"] = parameter_n_estimators
        
    elif model_criterion == 'XGBoost':
        parameter_num_estimators = st.sidebar.slider('n_estimators:',100,10000,1000,100)
        parameter_max_depth = st.sidebar.slider('max_depth:',1,20,7,1)
        parameter_booster = st.sidebar.radio('booster:', ('tree', 'linear'))
        parameter_objective = st.sidebar.radio('objective:', ('squared', 'squaredlog', 'pseudohuber'))
        params["parameter_num_estimators"] = parameter_num_estimators
        params["parameter_max_depth"] = parameter_max_depth
        params["parameter_booster"] = 'gb' + parameter_booster
        params["parameter_objective"] = 'reg:' + parameter_objective + 'error'
                
    elif model_criterion == 'MLPR':
        parameter_hidden_layer_sizes = st.sidebar.slider('Hidden Layer Sizes:', 10, 200, 100, 10)
        parameter_max_iter_MLP = st.sidebar.slider('Max. of Iterations:', 100, 20000, 1000, 100)
        parameter_activation = st.sidebar.radio('Activation function:', ('relu', 'identity', 'logistic', 'tanh'))
        params["parameter_hidden_layer_sizes"] = parameter_hidden_layer_sizes
        params["parameter_max_iter_MLP"] = parameter_max_iter_MLP
        params["parameter_activation"] = parameter_activation
        
    return params

def get_regressor(model_criterion, parameters):
    
    if model_criterion == 'k-NN':
        rgs = KNeighborsRegressor(n_neighbors=parameters['parameter_n_neighbors'], algorithm=parameters['parameter_type_algorithm'])
        
    elif model_criterion == 'SVR':
        # rgs = SVR(kernel= parameters['parameter_kernel'], gamma=parameters['parameter_gamma'], verbose=False)
        rgs = SVR(kernel= parameters['parameter_kernel'], verbose=False)
        
    elif model_criterion == 'Linear Regression':
        rgs = LinearRegression()
    
    elif model_criterion == 'Lasso Regression':
        rgs = Lasso(alpha= parameters['parameter_alpha'],selection=parameters['parameter_selection'],random_state= int(parameter_random_state))
    
    elif model_criterion == 'Random Forest':
        rgs = RandomForestRegressor(n_estimators = parameters['parameter_n_estimators'], random_state = int(parameter_random_state))
    
    elif model_criterion == 'XGBoost':
        rgs = xgb.XGBRegressor(n_estimators=parameters['parameter_num_estimators'], max_depth=parameters['parameter_max_depth'], 
                               booster=parameters["parameter_booster"], objective=parameters["parameter_objective"], random_state=int(parameter_random_state))
        
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
    df=df.astype(float)
    
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
    df=df.astype(float)
    
    X_train = df.iloc[:,:-1] # Using all column except for the last column as X
    Y_train = df.iloc[:,-1] # Selecting the last column as Y
    
    # Set model  
    Y_pred_train = get_regressor(model_criterion, parameters)
    
    # Model
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
    tpsep = st.sidebar.selectbox('Type of sep:', (';', ','))    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    # Create link to download CSV file
    tmp_download_link = download_link(sample_data(), 'SampleDataset.csv', 'Example CSV input file')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
    # Command of data criterion
    if uploaded_file is None: data_criterion = st.sidebar.radio('Would you like a sample dataset?', ('No','Yes'))
    # Show output variable
    if uploaded_file is not None:         
        uploaded_file.seek(0)
        df_names = pd.read_csv(uploaded_file, sep=tpsep,low_memory=False)
        name_output = st.sidebar.selectbox('Select the goal variable:', (df_names.columns.insert(0, "-")))
    else:
        name_output = 'output'

# ------ Control on image
if uploaded_file is None and data_criterion == 'No': st.image('https://raw.githubusercontent.com/juanmartinsantos/RegressionModelsApp/main/docs/loading.png', use_column_width=False)

# ------ Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Training Parameters'):
    parameter_random_state = st.sidebar.text_input('Seed number (random_state):', value= '1234')
    train_criterion = st.sidebar.radio('Choose a type of training:', ('k-fold', 'Data-split'))
    split_size = type_training(train_criterion)

with st.sidebar.subheader('3. Choose a Regression Algorithm'):
    model_criterion = st.sidebar.selectbox('Models:', ('-','k-NN', 'SVR', 'Linear Regression', 'Lasso Regression', 'Random Forest', 'XGBoost', 'MLPR'))

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

st.sidebar.write("Developed by [Juan Martín](https://snipfeed.co/juanmartin)")

# ----------------------------------------------- #
# ------------------ Main panel ----------------- #
# ----------------------------------------------- #
# Subtitle
if uploaded_file is not None or data_criterion == 'Yes': st.subheader('1. Dataset')

# Displays the dataset
if uploaded_file is not None:    
    # LOAD A DATASET 
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=tpsep, decimal=',')
    norm = show_data(df, name_output)
    df = df.dropna()
        
    if norm == "Normalized":
        df=normalization_data(df)
    
    # Ploting
    if model_criterion == "-":
        plot_criterion = st.selectbox('Explore data:', ("-", "Descriptions", "Correlation Matrix", "Histogram", "Box Plots", "Scatter Plots"))
        general_plot(df, plot_criterion)
    
    if model_criterion != "-":
        build_model(df, parameters)
    
else:
    # LOAD A DATASET 
    if data_criterion == 'Yes':
        df = sample_data()
        norm = show_data(df, name_output)
        
        if norm == "Normalized":
            df= normalization_data(df)
        
        # Ploting
        if model_criterion == "-":
            plot_criterion = st.selectbox('Explore data:', ("-", "Descriptions", "Correlation Matrix", "Histogram", "Box Plots", "Scatter Plots"))
            general_plot(df, plot_criterion)
        
        if model_criterion != "-":
            build_model(df, parameters)


# --- Prediction on unseen dataset
if make_criterion == 'Yes' and uploaded_file_test is not None:
    st.markdown('**2. Predictions**:')
    df_unseen = pd.read_csv(uploaded_file_test, sep = ';', decimal=',')
    st.markdown('**2.1. Glimpse of predictions dataset**')
    
    if norm == "Normalized":
            df_unseen= normalization_data(df_unseen)
            st.write(df_unseen.head(5))
    else: 
        st.write(df_unseen.head(5))
    
    if st.button('Run'):
        pred = pd.DataFrame(get_predict_unseen(df, df_unseen, parameters))
        st.download_button(label= 'Download', data= pred.to_csv(sep=';', index = False, header=False), file_name= model_criterion + '_predictions.csv')
        
        # st.balloons()


########################################################################
# --------------------------- Custom style --------------------------- #
########################################################################
# Remove “Made with Streamlit”
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Add footer
footer="""<style>
a:link , a:visited{
color: Sky Blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: purple;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: gray;
text-align: center;
}
</style>
<div class="footer">
<p>This app has been shared in the <a style='display: block; text-align: center;' href="https://discuss.streamlit.io/t/weekly-roundup-snowflake-animations-podcast-summaries-optimization-apps-and-more/20490" target="_blank">Streamlit's Weekly Roundup</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
