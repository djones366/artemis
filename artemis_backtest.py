#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import date_range
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column
import datetime
import json
import requests
from pandas import json_normalize
import smtplib
import os
from tickers import *
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os.path
import datetime
import mysql.connector


# In[2]:


sma_period = 14
ema_span = 14
rsi_period = 14
atr_period = 14
macd_fast_span = 12
macd_slow_span = 26


# In[3]:


tf.compat.v1.logging.set_verbosity('ERROR')


# In[4]:


cnx = mysql.connector.connect(user='root', password='Vatical777!',host='localhost',database='eod_data')
cursor = cnx.cursor()


# In[5]:


# tickers = ['AAPL', 'NVDA', 'ROKU', 'BYND']


# In[6]:


class Stock():
    api_key = '3RK1ZFFCYSLPOR3GVVDPCVAPB3RPQPMM'
    learning_rate = 0.0055
    epochs = 75
    batch_size = 3000
    test_split = 0.3
    validation_split = 0.3
    alpha = 0.005
    label_name = 'Short_GNG'
    look_ahead_periods = 11
    moving_average_period_14 = 14
    RSI_period_14 = 14
    atr_period = 14
    target_fraction = 0.1

  
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.df = pd.DataFrame()
        self.SPY_df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.model_filepath = 'model_data/{}/'.format(self.ticker_symbol)
        self.normalization_data_filepath = 'model_data/{}/{}.json'.format(self.ticker_symbol, self.ticker_symbol)
        self.normalization_data = {}
        self.model = None
        self.model_data = {}
        self.prediction_data = {}
        
    def retrieve_data_db(self,startDate,endDate):
        cursor.execute('SELECT * FROM {} WHERE Date BETWEEN "{}" AND "{}"'.format(ticker,startDate,endDate))
        table_rows = cursor.fetchall()
        self.df = pd.DataFrame(table_rows, columns=['Date','Datetime','Open','High','Low','Close','Volume'])
        self.df.drop(columns=['Datetime'],axis=1,inplace=True)
        self.df.set_index('Date', inplace=True)
        
        cursor.execute('SELECT * FROM SPY WHERE Date BETWEEN "{}" AND "{}"'.format(startDate,endDate))
        table_rows = cursor.fetchall()
        self.SPY_df = pd.DataFrame(table_rows, columns=['Date','Datetime','Open_SPY','High_SPY','Low_SPY','Close_SPY','Volume_SPY'])
        self.SPY_df.drop(columns=['Datetime'],axis=1,inplace=True)
        self.SPY_df.set_index('Date', inplace=True)
        
        self.df['High_SPY'] = self.SPY_df['High_SPY']
        self.df['Low_SPY'] = self.SPY_df['Low_SPY']
        self.df['Open_SPY'] = self.SPY_df['Open_SPY']
        self.df['Close_SPY'] = self.SPY_df['Close_SPY']
        self.df['Volume_SPY'] = self.SPY_df['Volume_SPY']
        
        
    def prepare_model_data(self):
        
        # Fill in any missing data
        self.df['High'].fillna(method='ffill')
        self.df['Low'].fillna(method='ffill')
        self.df['Open'].fillna(method='ffill')
        self.df['Close'].fillna(method='ffill')
        self.df['Volume'].fillna(method='ffill')
    
        # Calcualate and assign delta columns
        self.df['delta_High'] = self.df.High.diff()
        self.df['delta_Low'] = self.df.Low.diff()
        self.df['delta_Open'] = self.df.Open.diff()
        self.df['delta_Close'] = self.df.Close.diff()
        self.df['delta_Volume'] = self.df.Volume.diff()
    
        # Calculate simple moving average
        self.df['sma'] = self.df.Close.rolling(window=sma_period).mean()
        
        #Calculate exponential moving average
        self.df['ema'] = self.df.Close.ewm(span=ema_span,adjust=False).mean()
    
        #Calculate RSI
        self.df['rsi_down'] = self.df.Close.diff()
        self.df.loc[(self.df.rsi_down > 0), 'rsi_down'] = 0
        self.df.rsi_down = abs(self.df.rsi_down)
        self.df['rsi_up'] = self.df.Close.diff()
        self.df.loc[(self.df.rsi_up < 0), 'rsi_up'] = 0
        self.df['rsi_down_avg'] = self.df.rsi_down.ewm(com=rsi_period - 1, adjust=False).mean()
        self.df['rsi_up_avg'] = self.df.rsi_up.ewm(com=rsi_period - 1, adjust=False).mean()
        self.df['rs'] = self.df.rsi_up_avg/self.df.rsi_down_avg
        self.df['rsi'] = 100 -(100 / (1 + self.df.rs))
    
        #Calculate MACD
        self.df['macd_fast'] = self.df.Close.ewm(span=macd_fast_span,adjust=False).mean()
        self.df['macd_slow'] = self.df.Close.ewm(span=macd_slow_span,adjust=False).mean()
        self.df['macd'] = self.df.macd_fast - self.df.macd_slow
        self.df['macdsignal'] = self.df.macd.ewm(span=9,adjust=False).mean()
        self.df['macdhist'] = self.df.macd - self.df.macdsignal
    
        #Calculate ATR
        self.df['previous_close'] = self.df.Close.shift(1)
        self.df['tr'] = np.maximum((self.df.High - self.df.Low), np.maximum((abs(self.df.High - self.df.previous_close)), abs(self.df.previous_close - self.df.Low)))
        self.df['atr'] = self.df.tr.ewm(com=atr_period - 1).mean()

        # Calculate and assign gradient columns
        self.df['dHigh'] = np.gradient(self.df['High'])
        self.df['dLow'] = np.gradient(self.df['Low'])
        self.df['dOpen'] = np.gradient(self.df['Open'])
        self.df['dClose'] = np.gradient(self.df['Close'])
        self.df['dVolume'] = np.gradient(self.df['Volume'])
        self.df['dsma'] = np.gradient(self.df['sma'])
        self.df['dema'] = np.gradient(self.df['ema'])
        self.df['drsi'] = np.gradient(self.df['rsi'])
        self.df['dmacd'] = np.gradient(self.df['macd'])
        self.df['dmacdsignal'] = np.gradient(self.df['macdsignal'])
        self.df['dmacdhist'] = np.gradient(self.df['macdhist'])
        self.df['datr'] = np.gradient(self.df['atr'])
        
        # Iterate through rows to calculate delta values
        self.df['Short_GNG'] = 0.0
        for i in range(self.df.shape[0]):
            # Search the next look_ahead_periods to find the highest closing value
            LA_Min = self.df['Low'][i:i + self.look_ahead_periods].min()
        
            # Find close of current row for loop calcs
            row_close = self.df['Close'][i]
        
            # Fraction change of the min closing value in the next look_ahead_periods
            fraction = (row_close - LA_Min)/row_close
        
            # Determine positive go/no-go for the row based on comparison to target_fraction
            if fraction >= self.target_fraction:
                Short_GNG = 1.0
            else:
                Short_GNG = 0.0
            
            #Find index of current row for loop assignments
            index = self.df.index[i]
        
            # Assign value calculated in the loop to the current row
            self.df.loc[index, 'Short_GNG'] = Short_GNG
            
        #Drop not needed colums
        self.df.drop(columns=['rsi_down', 'rsi_up', 'rsi_down_avg', 'rsi_up_avg', 'rs', 'macd_fast', 'macd_slow', 'previous_close', 'tr'], inplace=True)
        
        # Normalize data with z score
        # Create a list column names and remove Long_GNG and Short_GNG  
        columns = list(self.df.columns)
        columns.remove('Short_GNG')
    
        # Calculate and record mean and std in stock dictionary
        for column_name in columns:
            mean = self.df[column_name].mean()
            std = self.df[column_name].std(ddof=0)
            self.normalization_data[column_name + '_mean'] = mean
            self.normalization_data[column_name + '_std'] = std
            self.df[column_name] = ((self.df[column_name]-mean)/std)
        
        # Drop rows with NAs or NaNs
        self.df.dropna(inplace=True)
        
    def split_data(self):
        
        # Shuffle rows
        self.df = self.df.reindex(np.random.permutation(self.df.index))

        # Calculate splits
        test_index_value = int(len(self.df)*(1 - self.test_split))
    
        # Determine indicies of splits
        start_index = self.df.index[0]
        test_index = self.df.index[test_index_value]
    
        # Split data into train-validate and test groups
        self.train_df = self.df.loc[start_index:test_index]
        self.test_df = self.df.loc[test_index:]


    def build_model(self):
        
        # Create list of df columns and remove Long_GNG and Short_GNG
        columns = list(self.df.columns)
        columns.remove('Short_GNG')
    
        # Create numeric feature columns from list of feature columns
        feature_columns = []
        for column in columns:
            feature_columns.append(tf.feature_column.numeric_column(column))
            
        # Convert the list of feature columns into a layer that will later be fed into the model.
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    
        # Most simple tf.keras models are sequential.
        self.model = tf.keras.models.Sequential()
    
        # Add the layer containing the feature columns to the model.
        self.model.add(feature_layer)
    
        # Add multiple additional layers
        self.model.add(tf.keras.layers.Dense(units=40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.alpha)))
        self.model.add(tf.keras.layers.Dense(units=24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.alpha)))
        self.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
        # Construct the layers into a model that TensorFlow can execute.
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])


    def train_model(self):
        
        # Print tracking statement
#         print('\n' + self.ticker_symbol)
    
        # Split the dataset into features and label.
        features = {name:np.array(value) for name, value in self.train_df.items()}
        label = np.array(features.pop(self.label_name))
        history = self.model.fit(x=features, y=label, batch_size=self.batch_size, epochs=self.epochs, verbose=0, shuffle=True)
    
        # To track the progression of training, gather a snapshot of the model's mean squared error at each epoch. 
        results = pd.DataFrame(history.history)
        self.model_data ['ticker_symbol'] = self.ticker_symbol
        self.model_data['train_loss'] = results.iloc[-1]['loss']
        self.model_data['train_accuracy'] = results.iloc[-1]['accuracy']

    def evaluate_model(self):
        
        # Split the dataset into features and label
        features = {name:np.array(value) for name, value in self.test_df.items()}
        label = np.array(features.pop(self.label_name))
        results = self.model.evaluate(x=features, y=label, batch_size=self.batch_size)
        self.model_data['test_loss'] = results[0]
        self.model_data['test_accuracy'] = results[1]


#     def save_model(self):
        
#         # Save model to drive for later restoration and use
#         self.model.save(self.model_filepath)
    
#         # Save the normalization_data dictionary for use when later loading the model
#         f = open(self.normalization_data_filepath, 'w')
#         json.dump(self.normalization_data, f)
#         f.close()

    def prepare_prediction_data(self):
        
        # Fill in any missing data
        self.df['High'].fillna(method='ffill')
        self.df['Low'].fillna(method='ffill')
        self.df['Open'].fillna(method='ffill')
        self.df['Close'].fillna(method='ffill')
        self.df['Volume'].fillna(method='ffill')
    
        # Calcualate and assign delta columns
        self.df['delta_High'] = self.df.High.diff()
        self.df['delta_Low'] = self.df.Low.diff()
        self.df['delta_Open'] = self.df.Open.diff()
        self.df['delta_Close'] = self.df.Close.diff()
        self.df['delta_Volume'] = self.df.Volume.diff()
    
        # Calculate simple moving average
        self.df['sma'] = self.df.Close.rolling(window=sma_period).mean()
        
        #Calculate exponential moving average
        self.df['ema'] = self.df.Close.ewm(span=ema_span,adjust=False).mean()
    
        #Calculate RSI
        self.df['rsi_down'] = self.df.Close.diff()
        self.df.loc[(self.df.rsi_down > 0), 'rsi_down'] = 0
        self.df.rsi_down = abs(self.df.rsi_down)
        self.df['rsi_up'] = self.df.Close.diff()
        self.df.loc[(self.df.rsi_up < 0), 'rsi_up'] = 0
        self.df['rsi_down_avg'] = self.df.rsi_down.ewm(com=rsi_period - 1, adjust=False).mean()
        self.df['rsi_up_avg'] = self.df.rsi_up.ewm(com=rsi_period - 1, adjust=False).mean()
        self.df['rs'] = self.df.rsi_up_avg/self.df.rsi_down_avg
        self.df['rsi'] = 100 -(100 / (1 + self.df.rs))
    
        #Calculate MACD
        self.df['macd_fast'] = self.df.Close.ewm(span=macd_fast_span,adjust=False).mean()
        self.df['macd_slow'] = self.df.Close.ewm(span=macd_slow_span,adjust=False).mean()
        self.df['macd'] = self.df.macd_fast - self.df.macd_slow
        self.df['macdsignal'] = self.df.macd.ewm(span=9,adjust=False).mean()
        self.df['macdhist'] = self.df.macd - self.df.macdsignal
    
        #Calculate ATR
        self.df['previous_close'] = self.df.Close.shift(1)
        self.df['tr'] = np.maximum((self.df.High - self.df.Low), np.maximum((abs(self.df.High - self.df.previous_close)), abs(self.df.previous_close - self.df.Low)))
        self.df['atr'] = self.df.tr.ewm(com=atr_period - 1).mean()

        # Calculate and assign gradient columns
        self.df['dHigh'] = np.gradient(self.df['High'])
        self.df['dLow'] = np.gradient(self.df['Low'])
        self.df['dOpen'] = np.gradient(self.df['Open'])
        self.df['dClose'] = np.gradient(self.df['Close'])
        self.df['dVolume'] = np.gradient(self.df['Volume'])
        self.df['dsma'] = np.gradient(self.df['sma'])
        self.df['dema'] = np.gradient(self.df['ema'])
        self.df['drsi'] = np.gradient(self.df['rsi'])
        self.df['dmacd'] = np.gradient(self.df['macd'])
        self.df['dmacdsignal'] = np.gradient(self.df['macdsignal'])
        self.df['dmacdhist'] = np.gradient(self.df['macdhist'])
        self.df['datr'] = np.gradient(self.df['atr'])
            
        #Drop not needed colums
        self.df.drop(columns=['rsi_down', 'rsi_up', 'rsi_down_avg', 'rsi_up_avg', 'rs', 'macd_fast', 'macd_slow', 'previous_close', 'tr'],inplace=True)
        
        # Normalize data with z score using previously calculated std and mean
        # Create a list column names
        columns = list(self.df.columns)
        
        # Calculate and record mean and std in stock dictionary
        for column_name in columns:
            mean = self.normalization_data[column_name + '_mean']
            std = self.normalization_data[column_name + '_std']
            self.df[column_name] = ((self.df[column_name]-mean)/std)
            
        
        # Drop rows with NAs or NaNs
        self.df.dropna(inplace=True)
        
        
    def generate_prediction(self):
        # Split the dataset into features and label
        features = {name:np.array(value) for name, value in self.df.items()}
        result = self.model.predict(x=features)
        prediction = result[-1].item()
            
        # Print result for tracking status
#         print('{} {}'.format(self.ticker_symbol, prediction))
            
        # Add prediction result to prediction data dictionary
        self.prediction_data['ticker_symbol'] = self.ticker_symbol
        self.prediction_data['target_fraction'] = self.target_fraction
        self.prediction_data['prediction'] = prediction


# In[7]:


def failed_string(failed_tickers):
    
    """Return a formatted string with the items provided in the failed list. The
        returned string is intended to be used as part of a notification to the user
        after attempting to retrieve data from several stocks in the tickers list."""
    
    # If failed_list is empty, return 'No failed items'
    if not failed_tickers:
        
        return_string = 'No failed items.'

    # If failed list contains items, return a formatted string with the items
    else:
        
        # Create a comma separated list of the items in the failed_list
        failed_tickers_string = ', '.join(failed_tickers)

        # Develop a formatted string using the failed_list_string
        return_string = 'The following items failed to produce a result: {}.'.format(failed_tickers_string)
        
    return return_string


# In[8]:


def send_email_with_attachment(email_list, subject, message, attachment_file_path='/home/markdonaho/stock_picker/results_data.csv', from_email='vaticalgroup@gmail.com', password='vhrjkybrvrkavfqr', email_server='smtp.gmail.com', email_server_port='587'):
    for to_email in email_list:
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))


        filename = os.path.basename(attachment_file_path)
        attachment = open(attachment_file_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)


        msg.attach(part)

        server = smtplib.SMTP(email_server, email_server_port)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()


# In[9]:


def send_text_message(phone_number_list, message, from_email='vaticalgroup@gmail.com', password='vhrjkybrvrkavfqr', email_server='smtp.gmail.com', email_server_port=587):
    for phone_number in phone_number_list:
        server = smtplib.SMTP( email_server, email_server_port)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, phone_number, message)
        server.quit()


# In[10]:


def find_exit_data(eod_df, index, target):
    req_exit_date = eod_df.iloc[index + 10]['Date']
    for i in range(index,index+11):
        daily_low = eod_df.iloc[i]['Low']
        
        if daily_low <= target:
            exit_date = eod_df.iloc[i]['Date']
            exit_price = target
            hit_target = True
            return exit_date, exit_price, hit_target, req_exit_date
        
        else:
            exit_date = req_exit_date
            exit_price = eod_df.iloc[index + 10]['Open']
            hit_target = False
            
    return exit_date, exit_price, hit_target, req_exit_date 


# In[11]:


phone_number_list = ['7572866310@mms.att.net', '9186912606@mms.att.net']
email_list = ['danieljones366@gmail.com', 'markdonaho@gmail.com']


# In[12]:


send_text_message(phone_number_list, 'Artemis backtest started.')


# In[13]:


backtest_data = pd.DataFrame(columns=['prediction_date','ticker_symbol','target_fraction','prediction'])


# In[14]:


build_start_date = '2016-4-1'
build_end_date = '2021-2-1'
build_start_date = datetime.date(int(build_start_date.split('-')[0]), int(build_start_date.split('-')[1]), int(build_start_date.split('-')[2]))
build_end_date = datetime.date(int(build_end_date.split('-')[0]), int(build_end_date.split('-')[1]), int(build_end_date.split('-')[2]))
build_date_list = []
active_date = build_start_date
num_days = int((build_end_date - build_start_date).total_seconds()/60/60/24)
for i in range(num_days):
    if active_date.weekday() == 6:
        build_date_list.append(str(active_date))
        
    active_date = active_date + datetime.timedelta(1)
    
print(build_date_list)


# In[15]:


today_date = str(datetime.date.today())
backtest_filename = 'artemis-backtest_' + today_date + '.csv'
f = open(backtest_filename, 'w')
f.write('{},{},{},{}{}'.format('prediction_date','ticker_symbol','target_fraction','prediction','\n'))
f.close()


# In[16]:


counter = 0
for build_date in build_date_list:    
    for ticker in tickers:
        try:
            #model build Sunday
            end_date = build_date
            start_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))-datetime.timedelta(3650))
            print(ticker, 'Model Build Date -', end_date)
            stock = Stock(ticker)
            stock.retrieve_data_db(start_date, end_date)
            stock.prepare_model_data()
            stock.split_data()
            stock.build_model()
            stock.train_model()
            stock.evaluate_model()
            #Monday's predictions
            end_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2])))
            start_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))-datetime.timedelta(300))
            stock.retrieve_data_db(start_date, end_date)
            stock.prepare_prediction_data()
            stock.generate_prediction()
            prediction_date = end_date
            ticker_symbol = stock.prediction_data['ticker_symbol']
            target_fraction = stock.prediction_data['target_fraction']
            prediction = stock.prediction_data['prediction']
            backtest_data.loc[counter] = [prediction_date,ticker_symbol,target_fraction,prediction]
            f = open(backtest_filename, 'a')
            f.write('{},{},{},{}{}'.format(prediction_date,ticker_symbol,target_fraction,prediction,'\n'))
            f.close()
            tf.keras.backend.clear_session()
            counter += 1
            #Tuesday's predictions
            end_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))+datetime.timedelta(1))
            start_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))-datetime.timedelta(300))
            stock.retrieve_data_db(start_date, end_date)
            stock.prepare_prediction_data()
            stock.generate_prediction()
            prediction_date = end_date
            ticker_symbol = stock.prediction_data['ticker_symbol']
            target_fraction = stock.prediction_data['target_fraction']
            prediction = stock.prediction_data['prediction']
            backtest_data.loc[counter] = [prediction_date,ticker_symbol,target_fraction,prediction]
            f = open(backtest_filename, 'a')
            f.write('{},{},{},{}{}'.format(prediction_date,ticker_symbol,target_fraction,prediction,'\n'))
            f.close()
            tf.keras.backend.clear_session()
            counter += 1
            #Wednesday's predictions
            end_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))+datetime.timedelta(1))
            start_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))-datetime.timedelta(300))
            stock.retrieve_data_db(start_date, end_date)
            stock.prepare_prediction_data()
            stock.generate_prediction()
            prediction_date = end_date
            ticker_symbol = stock.prediction_data['ticker_symbol']
            target_fraction = stock.prediction_data['target_fraction']
            prediction = stock.prediction_data['prediction']
            backtest_data.loc[counter] = [prediction_date,ticker_symbol,target_fraction,prediction]
            f = open(backtest_filename, 'a')
            f.write('{},{},{},{}{}'.format(prediction_date,ticker_symbol,target_fraction,prediction,'\n'))
            f.close()
            tf.keras.backend.clear_session()
            counter += 1
            #Thursday's predictions
            end_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))+datetime.timedelta(1))
            start_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))-datetime.timedelta(300))
            stock.retrieve_data_db(start_date, end_date)
            stock.prepare_prediction_data()
            stock.generate_prediction()
            prediction_date = end_date
            ticker_symbol = stock.prediction_data['ticker_symbol']
            target_fraction = stock.prediction_data['target_fraction']
            prediction = stock.prediction_data['prediction']
            backtest_data.loc[counter] = [prediction_date,ticker_symbol,target_fraction,prediction]
            f = open(backtest_filename, 'a')
            f.write('{},{},{},{}{}'.format(prediction_date,ticker_symbol,target_fraction,prediction,'\n'))
            f.close()
            tf.keras.backend.clear_session()
            counter += 1
            #Friday's predictions
            end_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))+datetime.timedelta(1))
            start_date = str(datetime.date(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))-datetime.timedelta(300))
            stock.retrieve_data_db(start_date, end_date)
            stock.prepare_prediction_data()
            stock.generate_prediction()
            prediction_date = end_date
            ticker_symbol = stock.prediction_data['ticker_symbol']
            target_fraction = stock.prediction_data['target_fraction']
            prediction = stock.prediction_data['prediction']
            backtest_data.loc[counter] = [prediction_date,ticker_symbol,target_fraction,prediction]
            f = open(backtest_filename, 'a')
            f.write('{},{},{},{}{}'.format(prediction_date,ticker_symbol,target_fraction,prediction,'\n'))
            f.close()
            tf.keras.backend.clear_session()
            counter += 1
        except:
            print('Exception:',ticker)
            pass


# In[17]:


today_date = str(datetime.date.today())
artemis_accuracy_filename = 'artemis-accuracy_' + today_date + '.csv'
f = open(artemis_accuracy_filename, 'w')
f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}{}'.format('ticker_symbol','target_fraction','prediction','target','prediction_date','entry_date','exit_date','req_exit_date','entry_price','exit_price','highest_high','lowest_low','hit_target','\n'))
f.close()


# In[18]:


prediction_threshold = 0.9
api_key = '3RK1ZFFCYSLPOR3GVVDPCVAPB3RPQPMM'
periodType = 'year'
frequencyType = 'daily'
frequency = 1


counter = 0
except_counter = 0
for i in range(len(backtest_data)):
    try:
        if backtest_data.iloc[i]['prediction'] >= prediction_threshold:
            ticker_symbol = backtest_data.iloc[i]['ticker_symbol']
            target_fraction = backtest_data.iloc[i]['target_fraction']
            prediction = backtest_data.iloc[i]['prediction']
            print(ticker_symbol, target_fraction, prediction)
            prediction_date = backtest_data.iloc[i]['prediction_date']

            
            db_prediction_date = str((datetime.date(int(prediction_date.split('-')[0]), int(prediction_date.split('-')[1]), int(prediction_date.split('-')[2])))-datetime.timedelta(10))
            db_exit_date = str((datetime.date(int(prediction_date.split('-')[0]), int(prediction_date.split('-')[1]), int(prediction_date.split('-')[2])))+datetime.timedelta(60))


            
            cursor.execute('SELECT * FROM {} WHERE Date BETWEEN "{}" AND "{}"'.format(ticker_symbol,db_prediction_date,db_exit_date))
            table_rows = cursor.fetchall()
            eod_data = pd.DataFrame(table_rows, columns=['Date','Datetime','Open','High','Low','Close','Volume'])
            eod_data.drop(columns=['Datetime'],axis=1,inplace=True)
            
            
            entry_index = list(eod_data.loc[eod_data['Date']==(datetime.date(int(prediction_date.split('-')[0]), int(prediction_date.split('-')[1]), int(prediction_date.split('-')[2])))].index)[0] + 1
            entry_date = eod_data.iloc[entry_index]['Date']
            target = eod_data.iloc[entry_index - 1]['Close'] * 0.9
            entry_price = eod_data.iloc[entry_index]['Open']
            highest_high = eod_data['High'][entry_index:(entry_index+11)].max()
            lowest_low = eod_data['Low'][entry_index:(entry_index+11)].min()
            exit_date, exit_price, hit_target, req_exit_date = find_exit_data(eod_data, entry_index, target)
            f = open(artemis_accuracy_filename, 'a')
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}{}'.format(ticker_symbol,target_fraction,prediction,target,prediction_date,entry_date,exit_date,req_exit_date,entry_price,exit_price,highest_high,lowest_low,hit_target,'\n'))
            f.close()
            counter += 1
    except:
        except_counter +=1
        pass
    
print('Exception Counter:', except_counter)


# In[19]:


cursor.close()
cnx.close()


# In[20]:


send_text_message(phone_number_list, 'Artemis backtest finished.')

