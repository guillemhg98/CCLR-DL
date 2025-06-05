# utils_DL.py
# ------------------------------------------------------
# Utility functions for deep learning-based time series prediction
# Author: Guillem Hernández Guillamet
# Version: 1.0
# Date: 04/06/2025
# Description:
#   This module defines utility functions to create GRU and LSTM models,
#   perform fitting, make predictions, and support basic inverse transforms
#   and automated grid search optimization for model parameters.
# ------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np

# --- Model Definitions ---
# model 1: GRU unit --------------------------------------------------------------
def create_model_gru(X_train, optimizer='adam'):
    
    model = Sequential()
    model.add(GRU(units = 100, input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(RepeatVector(FORECAST_RANGE))
    model.add(Dropout(0.2))
    model.add(GRU(units = 100, return_sequences = True,))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_features)))
    
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

# model 2: LSTM unit --------------------------------------------------------------
def create_model_lstm(X_train, optimizer='adam'):
    
    model = Sequential()
    model.add(LSTM(units = 100, input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(RepeatVector(FORECAST_RANGE))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 100, return_sequences = True,))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_features)))
    
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

# model 3: BILSTM unit --------------------------------------------------------------
def create_model_bilstm(X_train, optimizer='adam'):
    
    model = Sequential()
    model.add(Bidirectional(LSTM(units = 100), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(RepeatVector(FORECAST_RANGE))
    model.add(Bidirectional(LSTM(units = 100, return_sequences=True)))
    model.add(TimeDistributed(Dense(n_features)))
    
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

# model 4: ENCODER-DECODER LSTM unit ------------------------------------------------
def create_model_enc_dec(X_train, optimizer='adam'):
    
    model_enc_dec = Sequential()
    model_enc_dec.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_enc_dec.add(RepeatVector(FORECAST_RANGE))
    model_enc_dec.add(LSTM(100, activation='relu', return_sequences=True))
    model_enc_dec.add(TimeDistributed(Dense(n_features)))
    
    model_enc_dec.compile(optimizer=optimizer, loss='mse')
    
    return model_enc_dec

def create_model_enc_dec_cnn(X_train, optimizer='adam',kern_size = 3):
    
    model_enc_dec_cnn = Sequential()
    model_enc_dec_cnn.add(Conv1D(filters=64, kernel_size=kern_size, activation='relu', 
                                 input_shape=(X_train.shape[1], X_train.shape[2])))
    model_enc_dec_cnn.add(Conv1D(filters=64, kernel_size=kern_size, activation='relu'))
    model_enc_dec_cnn.add(MaxPooling1D(pool_size=2))
    model_enc_dec_cnn.add(Flatten())
    model_enc_dec_cnn.add(RepeatVector(FORECAST_RANGE))
    model_enc_dec_cnn.add(LSTM(200, activation='relu', return_sequences=True))
    model_enc_dec_cnn.add(TimeDistributed(Dense(100, activation='relu')))
    model_enc_dec_cnn.add(TimeDistributed(Dense(n_features)))
    
    model_enc_dec_cnn.compile(loss='mse', optimizer=optimizer)
    
    return model_enc_dec_cnn


# model 5: VECTOR OUTPUT -----------------------------------------------------------
def create_model_vector_output(X_train, optimizer='adam',kern_size = 3):
    
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2])) 
    conv = Conv1D(filters=4, kernel_size=kern_size, activation='relu')(input_layer)
    conv = Conv1D(filters=6, kernel_size=kern_size, activation='relu')(conv)

    lstm = LSTM(100, return_sequences=True, activation='relu')(conv)
    dropout = Dropout(0.2)(lstm)
    lstm = LSTM(100, activation='relu')(dropout)
    dense = Dense(FORECAST_RANGE*n_features, activation='relu')(lstm)
    output_layer = Reshape((FORECAST_RANGE,n_features))(dense)
    model_vector_output = Model([input_layer], [output_layer])
    
    model_vector_output.compile(optimizer=optimizer, loss='mse')
    
    return model_vector_output

# model 6: Multi head LSTM cnn -----------------------------------------------------------
def create_model_multi_head_cnn_lstm(X_train, optimizer='adam',kern_size = 3):
    
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))  #Look back, n_features
    head_list = []
    for i in range(0, n_features):
        conv_layer_head = Conv1D(filters=4, kernel_size=kern_size, activation='relu')(input_layer)
        conv_layer_head_2 = Conv1D(filters=6, kernel_size=kern_size, activation='relu')(conv_layer_head)
        conv_layer_flatten = Flatten()(conv_layer_head_2)
        head_list.append(conv_layer_flatten)

    concat_cnn = Concatenate(axis=1)(head_list)
    reshape = Reshape((head_list[0].shape[1], n_features))(concat_cnn)
    lstm = LSTM(100, activation='relu')(reshape)
    repeat = RepeatVector(FORECAST_RANGE)(lstm)
    lstm_2 = LSTM(100, activation='relu', return_sequences=True)(repeat)
    dropout = Dropout(0.2)(lstm_2)
    dense = Dense(n_features, activation='linear')(dropout)
    multi_head_cnn_lstm_model = Model(inputs=input_layer, outputs=dense)
    
    multi_head_cnn_lstm_model.compile(optimizer=optimizer, loss='mse')
    
    return multi_head_cnn_lstm_model


# --- Model Training ---
def fit_model(model, X_train, y_train, epochs, batch_size, validation_data=None, patience=10, verbose=1):
    checkpoint_callback = ModelCheckpoint(
        filepath='best_model.keras',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.005,
        patience=patience,
        mode='min'
    )

    rlrop_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.001,
        mode='min'
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping_callback, rlrop_callback],
        verbose=verbose
    )
    return history

# --- Prediction and Inverse Transformation ---
def prediction(model, X_test):
    return model.predict(X_test)

def inverse_transform(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(np.reshape(y_true, (-1, 1)))
    y_pred_inv = scaler.inverse_transform(np.reshape(y_pred, (-1, 1)))
    return y_true_inv, y_pred_inv

# --- Grid Search ---
def auto_grid_search(model_fn, X_train, y_train, X_test, y_test, scaler, validation_data, patience=10):
    epochs_grid = [10, 20, 50, 100]
    batch_sizes = [16, 32, 64]
    optimizers = ['SGD', 'Adam', 'RMSprop']

    best_mape = float('inf')
    best_config = {}

    for epochs in epochs_grid:
        for batch_size in batch_sizes:
            for opt in optimizers:
                model = model_fn(X_train, optimizer=opt)
                fit_model(model, X_train, y_train, epochs, batch_size, validation_data, patience, verbose=0)
                yhat = prediction(model, X_test)
                y_true_inv, yhat_inv = inverse_transform(y_test, yhat, scaler)
                mape = tf.keras.losses.MeanAbsolutePercentageError()(y_true_inv, yhat_inv).numpy()
                print(f"MAPE: {mape:.4f}  | Optimizer: {opt}, Epochs: {epochs}, Batch Size: {batch_size}")

                if mape < best_mape:
                    best_mape = mape
                    best_config = {'optimizer': opt, 'epochs': epochs, 'batch_size': batch_size}

    print("\nBest Config:", best_config, "with MAPE:", best_mape)
    return best_config