import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime
# To work with text data 
import re
import string
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn import linear_model
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

scaler = MinMaxScaler()


def text_cleaning(text):
    text  = "".join([char for char in text if char not in string.punctuation])    
    return text

def convert_to_category(column_name, data):
    # Creating a instance of label Encoder.
    le = LabelEncoder()
    label = le.fit_transform(data[column_name])

    data.drop(column_name, axis=1, inplace=True)
    
    # Appending the array to our dataFrame
    # with column name 'Purchased'
    data[column_name] = label

    return data

def convert_to_time_range(dataTemp):
    time_ranges = []
    for times in dataTemp['OrderTime']:
        times = times.split(":")[0]
        
        if float(times) >= 0 and float(times) <= 3:
            time_ranges.append(0)

        elif float(times) > 3 and float(times) <= 6:
            time_ranges.append(1)

        elif float(times) > 6 and float(times) <= 9:
            time_ranges.append(2)

        elif float(times) > 9 and float(times) <= 12:
            time_ranges.append(3)

        elif float(times) >= 12 and float(times) <= 15:
            time_ranges.append(4)

        elif float(times) > 15 and float(times) <= 18:
            time_ranges.append(5)

        elif float(times) > 18 and float(times) <= 21:
            time_ranges.append(6)

        elif float(times) > 21 and float(times) <= 24:
            time_ranges.append(7)


    return time_ranges

def add_weather(dataTemp):
    weather = []

    for temperature in dataTemp['Temperature']:
        if temperature <= 17.0:
            weather.append("Winter")

        elif temperature > 17.0 and temperature <= 22.0:
            weather.append("Rainy")

        elif temperature > 22.0 and temperature <= 25.0:
            weather.append("Autumn")

        elif temperature > 25.0:
            weather.append("Summer")

    return weather

def add_weather_single(temp):
    
    if temperature <= 17.0:
        weather = "Winter"

    elif temperature > 17.0 and temperature <= 22.0:
        weather = "Rainy" 

    elif temperature > 22.0 and temperature <= 25.0:
        weather = "Autumn" 

    elif temperature > 25.0:
        weather = "Summer" 

    return weather

def add_price_range(dataTemp):
    prices = []
    for price in dataTemp['Price']:
        if price <= 50:
            prices.append(0)

        elif price > 50 and price <= 100:
            prices.append(1)

        elif price > 100 and price <= 150:
            prices.append(2)

        elif price > 150 and price <= 200:
            prices.append(3)

        elif price > 200 and price <= 250:
            prices.append(4)

        elif price > 250 and price <= 300:
            prices.append(5)

        if price > 300:
            prices.append(6)

    return prices

def plot_predictions(y_pred_data, y_actual_data):
    # Plotting CustomerId vs Predicted ItemId
    plt.figure(figsize=(10, 6))
    plt.scatter(range(100), y_pred_data, color='red', label='Incorrect prediction')
    plt.scatter(range(100), y_actual_data, color='green', label='Correct prediction')
    plt.xlabel('Customer Id')
    plt.ylabel('ItemId')
    plt.title('Predicted vs Actual ItemId')
    plt.legend()
    plt.show()

def visualize_data(clf, x_test, y_test):
    y_pred_data = []
    y_actual_data = []

    y_pred = clf.predict(x_test[:100])
    actual_pred = y_test[:100]

    for i in y_pred:
        y_pred_data.append(i)

    for j in actual_pred:
        y_actual_data.append(j)

    print()
    print("Predictions: ", y_pred_data)
    print()
    print("Actual Values: ", y_actual_data)

    return y_pred_data, y_actual_data

def process_classification(x_train, y_train, x_test, y_test):
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators = 100)
    
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(x_train, y_train)
    
    # performing predictions on the test dataset
    y_pred = clf.predict(x_test)
    
    # metrics are used to find accuracy or error
    print()
    
    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

    return clf

def visualize_Preprocessing(data):
    print("Before Preprocessing")
    print(data.head(10))
    print("***************************************************************")

    # Let's clean the text 
    data['OrderType'] = data['OrderType'].apply(text_cleaning)

    data['Order Time'] = convert_to_time_range(data)
    data['Price Range'] = add_price_range(data)

    data = data.drop(['OrderTime'], axis=1)

    data = convert_to_category('OrderType', data)
    data = convert_to_category('Weather', data)
    data = convert_to_category('Category', data)
    data = convert_to_category('Tag', data)
    # data = convert_to_category('ItemName', data)

    print()
    print("After Preprocessing")
    print(data.head(10))
    print("***************************************************************")
    print()

    return data

def allocate_train_test_values(data):

    X = data.drop(['ItemId'], axis=1)
    y = data['ItemId']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    cols = x_train.columns

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train, columns=[cols])
    x_test = pd.DataFrame(x_test, columns=[cols])

    return x_train, x_test, y_train, y_test

def prepare_data():
    tempdata = pd.read_csv("OrderData.V4.csv")
    tempdata['Weather'] = add_weather(tempdata)
    tempdata.to_csv('Trainingdata.csv', sep=',')

def save_train_model():

    # # Step 1: Prepare final data from Raw data.
    prepare_data()
    
    # # Step 2: Load data from Excel.
    fields = ['CustomerID', 'OrderType', 'OrderTime', 'Temperature', 'Category', 'Tag','Price', 'Weather', 'ItemId']
    data = pd.read_csv("Trainingdata.csv", skipinitialspace=True, usecols=fields, nrows=1200000)
    
    # # Step 3: Display Preprocessing samples.
    data = visualize_Preprocessing(data)

    # # Step 4: Allocate train data and test data to coresponding variables.
    x_train, x_test, y_train, y_test = allocate_train_test_values(data)
    
    # # Step 5: Classify data using Random Forest classifier.
    clf = process_classification(x_train, y_train, x_test, y_test)

    # # Step 6: Show Actual and Predicted Item Ids.
    y_pred_data, y_actual_data = visualize_data(clf, x_test, y_test)    

    # # Step 7: Plot the predictions.
    plot_predictions(y_pred_data, y_actual_data)
 

save_train_model()
