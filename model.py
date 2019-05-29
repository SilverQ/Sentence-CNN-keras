# from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
# from keras.layers import Reshape, Flatten, Dropout, Concatenate
# from keras.callbacks import ModelCheckpoint
# from keras.optimizers import Adam
# from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data
import tensorflow as tf


x, y, vocabulary, vocabulary_inv = load_data()
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=True)
print('Loading data completed')
'''
print(x.shape)
print(y.shape)
print(len(vocabulary))
print(len(vocabulary_inv))

# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)
'''

sequence_length = x.shape[1]            # 56
vocabulary_size = len(vocabulary_inv)   # 18765
embedding_dim = 256
filter_sizes = [2, 3, 4, 5]
num_filters = [128, 256, 512]
drop = 0.4

epochs = 10
batch_size = 150

# this returns a tensor
print("Creating Model...")
# inputs = Input(shape=(sequence_length,), dtype='int32')
inputs = tf.keras.layers.Input(shape=(sequence_length,), dtype='int32')
embedding = tf.keras.layers.Embedding(input_dim=vocabulary_size,
                                      output_dim=embedding_dim,
                                      input_length=sequence_length)(inputs)
reshape = tf.keras.layers.Reshape((sequence_length, embedding_dim, 1))(embedding)

'''
# Original Code
conv_0 = tf.keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = tf.keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = tf.keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = tf.keras.layers.MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = tf.keras.layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            (None, 56)           0
# __________________________________________________________________________________________________
# embedding (Embedding)           (None, 56, 256)      4803840     input_1[0][0]
# __________________________________________________________________________________________________
# reshape (Reshape)               (None, 56, 256, 1)   0           embedding[0][0]
# __________________________________________________________________________________________________
# conv2d (Conv2D)                 (None, 54, 1, 512)   393728      reshape[0][0]
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 53, 1, 512)   524800      reshape[0][0]
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 52, 1, 512)   655872      reshape[0][0]
# __________________________________________________________________________________________________
# max_pooling2d (MaxPooling2D)    (None, 1, 1, 512)    0           conv2d[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 1, 1, 512)    0           conv2d_1[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_2 (MaxPooling2D)  (None, 1, 1, 512)    0           conv2d_2[0][0]
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 3, 1, 512)    0           max_pooling2d[0][0]
#                                                                  max_pooling2d_1[0][0]
#                                                                  max_pooling2d_2[0][0]
# __________________________________________________________________________________________________
# flatten (Flatten)               (None, 1536)         0           concatenate[0][0]
# __________________________________________________________________________________________________
# dropout (Dropout)               (None, 1536)         0           flatten[0][0]
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 2)            3074        dropout[0][0]
# ==================================================================================================
# Total params: 6,381,314
# Trainable params: 6,381,314
# Non-trainable params: 0
# __________________________________________________________________________________________________
#
# Process finished with exit code 0

# Modified code
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            (None, 56)           0                                            
# __________________________________________________________________________________________________
# embedding (Embedding)           (None, 56, 256)      4803840     input_1[0][0]                    
# __________________________________________________________________________________________________
# reshape (Reshape)               (None, 56, 256, 1)   0           embedding[0][0]                  
# __________________________________________________________________________________________________
# conv2d (Conv2D)                 (None, 54, 1, 128)   98432       reshape[0][0]                    
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 53, 1, 128)   131200      reshape[0][0]                    
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 52, 1, 128)   163968      reshape[0][0]                    
# __________________________________________________________________________________________________
# max_pooling2d (MaxPooling2D)    (None, 27, 1, 128)   0           conv2d[0][0]                     
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 27, 1, 128)   0           conv2d_1[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_2 (MaxPooling2D)  (None, 26, 1, 128)   0           conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 27, 1, 256)   98560       max_pooling2d[0][0]              
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 27, 1, 256)   131328      max_pooling2d_1[0][0]            
# __________________________________________________________________________________________________
# conv2d_5 (Conv2D)               (None, 26, 1, 256)   164096      max_pooling2d_2[0][0]            
# __________________________________________________________________________________________________
# max_pooling2d_3 (MaxPooling2D)  (None, 13, 1, 256)   0           conv2d_3[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_4 (MaxPooling2D)  (None, 13, 1, 256)   0           conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_5 (MaxPooling2D)  (None, 13, 1, 256)   0           conv2d_5[0][0]                   
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 13, 1, 512)   393728      max_pooling2d_3[0][0]            
# __________________________________________________________________________________________________
# conv2d_7 (Conv2D)               (None, 13, 1, 512)   524800      max_pooling2d_4[0][0]            
# __________________________________________________________________________________________________
# conv2d_8 (Conv2D)               (None, 13, 1, 512)   655872      max_pooling2d_5[0][0]            
# __________________________________________________________________________________________________
# max_pooling2d_6 (MaxPooling2D)  (None, 6, 1, 512)    0           conv2d_6[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_7 (MaxPooling2D)  (None, 6, 1, 512)    0           conv2d_7[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_8 (MaxPooling2D)  (None, 6, 1, 512)    0           conv2d_8[0][0]                   
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 18, 1, 512)   0           max_pooling2d_6[0][0]            
#                                                                  max_pooling2d_7[0][0]            
#                                                                  max_pooling2d_8[0][0]            
# __________________________________________________________________________________________________
# flatten (Flatten)               (None, 9216)         0           concatenate[0][0]                
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 1024)         9438208     flatten[0][0]                    
# __________________________________________________________________________________________________
# dropout (Dropout)               (None, 1024)         0           dense[0][0]                      
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 128)          131200      dropout[0][0]                    
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 128)          0           dense_1[0][0]                    
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 2)            258         dropout_1[0][0]                  
# ==================================================================================================
# Total params: 16,735,490
# Trainable params: 16,735,490
# Non-trainable params: 0
# __________________________________________________________________________________________________
# Traning Model...
# Train on 7463 samples, validate on 3199 samples
'''

# Modified
conv_00 = tf.keras.layers.Conv2D(num_filters[0], kernel_size=(filter_sizes[0], embedding_dim),
                                 padding='valid', activation='relu', )(reshape)
conv_01 = tf.keras.layers.Conv2D(num_filters[0], kernel_size=(filter_sizes[1], embedding_dim),
                                 padding='valid', activation='relu')(reshape)
conv_02 = tf.keras.layers.Conv2D(num_filters[0], kernel_size=(filter_sizes[2], embedding_dim),
                                 padding='valid', activation='relu')(reshape)

maxpool_00 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(conv_00)
maxpool_01 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(conv_01)
maxpool_02 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(conv_02)

conv_10 = tf.keras.layers.Conv2D(num_filters[1], kernel_size=(filter_sizes[0], 1),
                                 padding='same', activation='relu', )(maxpool_00)
conv_11 = tf.keras.layers.Conv2D(num_filters[1], kernel_size=(filter_sizes[1], 1),
                                 padding='same', activation='relu')(maxpool_01)
conv_12 = tf.keras.layers.Conv2D(num_filters[1], kernel_size=(filter_sizes[2], 1),
                                 padding='same', activation='relu')(maxpool_02)

maxpool_10 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv_10)
maxpool_11 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv_11)
maxpool_12 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv_12)

conv_20 = tf.keras.layers.Conv2D(num_filters[2], kernel_size=(filter_sizes[0], 1),
                                 padding='same', activation='relu', )(maxpool_10)
conv_21 = tf.keras.layers.Conv2D(num_filters[2], kernel_size=(filter_sizes[1], 1),
                                 padding='same', activation='relu')(maxpool_11)
conv_22 = tf.keras.layers.Conv2D(num_filters[2], kernel_size=(filter_sizes[2], 1),
                                 padding='same', activation='relu')(maxpool_12)

maxpool_20 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv_20)
maxpool_21 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv_21)
maxpool_22 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv_22)


concatenated_tensor = tf.keras.layers.Concatenate(axis=1)([maxpool_20, maxpool_21, maxpool_22])

flatten = tf.keras.layers.Flatten()(concatenated_tensor)
dense01 = tf.keras.layers.Dense(units=1024, activation='relu')(flatten)
dropout01 = tf.keras.layers.Dropout(drop)(dense01)
dense02 = tf.keras.layers.Dense(units=128, activation='relu')(dropout01)
dropout02 = tf.keras.layers.Dropout(drop)(dense02)
output = tf.keras.layers.Dense(units=2, activation='softmax')(dropout02)

# this creates a model that includes
model = tf.keras.Model(inputs=inputs, outputs=output)

checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoints/weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                                monitor='val_acc', verbose=1,
                                                save_best_only=True, mode='auto')
adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# adam = tf.keras.optimizers.Adam(lr=0.0001)

model.summary()
# tf.keras.utils.plot_model(model, to_file='model.png')

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
#           callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint], validation_data=(X_test, y_test),
          validation_split=0.3, shuffle=True)  # starts training

# Document
# fit(x=None, y=None, batch_size=None, epochs=1, verbose=1,
#     callbacks=None, validation_split=0.0, validation_data=None,
#     shuffle=True, class_weight=None, sample_weight=None,
#     initial_epoch=0, steps_per_epoch=None,
#     validation_steps=None, validation_freq=1)

'''
# Original Code
# 8529/8529 [==============================] - 6s 692us/sample
# - loss: 0.0126 - acc: 0.9987 - val_loss: 0.8327 - val_acc: 0.7417     -> overfitted so much!!

# Modified
# 8529/8529 [==============================] - 4s 418us/sample
# - loss: 0.0013 - acc: 0.9999 - val_loss: 1.7162 - val_acc: 0.7304

# test_size=0.2 -> test_size=0.3
# 7463/7463 [==============================] - 3s
# 433us/sample - loss: 0.0015 - acc: 1.0000 - val_loss: 1.6439 - val_acc: 0.7409

# batch_size = 30 -> batch_size = 64
# 7463/7463 [==============================] - 2s
# 256us/sample - loss: 0.0049 - acc: 0.9992 - val_loss: 1.4545 - val_acc: 0.7337

# epochs = 10 -> epochs = 4
# 7232/7463 [============================>.] - ETA: 0s - loss: 0.2754 - acc: 0.8920
# Epoch 00004: val_acc improved from 0.73242 to 0.74867, saving model to weights.004-0.7487.hdf5

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)

'''
