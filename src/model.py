# src/model.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_model():
    # fMRI Model
    input_fMRI = Input(shape=(246, 25, 1), name='fMRI_input')
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(input_fMRI)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.4)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x1)
    x1 = Dropout(0.5)(x1)

    # PPG Model
    input_ppg = Input(shape=(10000, 1), name='ppg_input')
    x2 = LSTM(16, return_sequences=True, dropout=0.4, recurrent_dropout=0.4, kernel_regularizer=l2(0.01))(input_ppg)
    x2 = LSTM(16, dropout=0.4, recurrent_dropout=0.4, kernel_regularizer=l2(0.01))(x2)
    x2 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x2)
    x2 = Dropout(0.5)(x2)

    # Resp Model
    input_resp = Input(shape=(10000, 1), name='resp_input')
    x3 = LSTM(16, return_sequences=True, dropout=0.4, recurrent_dropout=0.4, kernel_regularizer=l2(0.01))(input_resp)
    x3 = LSTM(16, dropout=0.4, recurrent_dropout=0.4, kernel_regularizer=l2(0.01))(x3)
    x3 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x3)
    x3 = Dropout(0.5)(x3)

    # Combine Models
    combined = concatenate([x1, x2, x3])
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(combined)
    x = Dropout(0.5)(x)
    output_class = Dense(3, activation='softmax', name='class_output')(x)
    output_level = Dense(1, activation='linear', name='level_output')(x)

    model = Model(inputs=[input_fMRI, input_ppg, input_resp], outputs=[output_class, output_level])

    opt = Adam(learning_rate=0.00005)
    model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy', 'mse'])

    return model

def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    return [early_stopping, reduce_lr]
