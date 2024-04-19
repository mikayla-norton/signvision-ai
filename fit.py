import keras
from keras.layers import Input, Dense, Flatten, Dropout
from keras.callbacks import Callback, EarlyStopping
from keras.models import Model
import numpy as np
from tensorflow.keras.regularizers import l2

class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

def pretrainedNetwork(xtrain,ytrain,pct,pretrained,pretrainedweights,classweight,numclasses,numepochs,optimizer):
    base_model = pretrained # Topless
    # Add top layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(numclasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # Train top layer
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()
    # Fit model
    history = model.fit(xtrain,ytrain, epochs=numepochs, class_weight=classweight, validation_split=pct, verbose=1,callbacks = [MetricsCheckpoint('logs')])
    return(history, model)

def scaled1dNetwork(xtrain, ytrain, pct, classweight, numclasses, numepochs, optimizer):
    input_shape = (xtrain.shape[1],)  # xtrain after PCA reduction
    inputs = Input(shape=input_shape)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = Dropout(0.5)(x)  # Add dropout of 50%
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)  # Add dropout of 50%
    predictions = Dense(numclasses, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    callbacks_list = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]  # Monitor val_loss instead
    
    model.summary()
    
    history = model.fit(xtrain, ytrain, epochs=numepochs, class_weight=classweight,
                        validation_split=pct, verbose=1, callbacks=callbacks_list)
    
    return history, model