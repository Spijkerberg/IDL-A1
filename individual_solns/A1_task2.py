# %% [markdown]
# # Imports

# %%
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import Input, Model, backend as K

print("Tensorflow version: ", tf.__version__)
from IPython.display import display

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# Enable the TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
# # This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)

print("All devices: ", *tf.config.list_logical_devices("TPU"), sep="\n\t")

# %% [markdown]
# # Load in the data

# %%
# How to get path from Google Drive: Get path by linking first, then selecting the file w/ right mouse button; 'get path'
labels = np.load('/content/drive/MyDrive/IDL/A1/labels.npy')
data = np.load('/content/drive/MyDrive/IDL/A1/images.npy')

# %% [markdown]
# #### Example of a clock

# %%
plt.imshow(data[0,:,:]/255.0, cmap='gray')
plt.show()

# %% [markdown]
# # Common sense time:

# %%
def common_sense_time(y_true, y_pred):
  predicted_min = y_pred[0]*60+y_pred[1]
  actual_min = y_true[0]*60+y_true[1]
  # Returns the difference in units of minutes
  return (predicted_min-actual_min)%720

# %%
# Sample examples
print(common_sense_time([11, 55], [0, 5]))
print(common_sense_time([0, 5], [11, 55]))

# %% [markdown]
# # 2A
# 
# Make a classification model

# %% [markdown]
# First make the labels applicable

# %%
def to_classification(labels, min_per_bin : int = 30):
  h, m = labels.T
  return 60//min_per_bin*h +m//min_per_bin

labels_1min = to_classification(labels, 1) # 720 classes

# %%
plt.hist(labels_1min, color='blue')
plt.xlabel("Class", fontsize=14)
plt.ylabel("# Instances", fontsize=14)
plt.show()

# %% [markdown]
# #### Make the classification model

# %%
def classification_model(input_shape):
  """
  Builds a classification model for time.

  This model is composed of three Conv -> BN -> Pool -> Dropout blocks,
  followed by the Dense output layer.

  # Output (we have in total 720 classes of time, 60*12, each with a given probability per item (returned by Softmax)
  """
  inputs = Input(shape=input_shape)
  x = Conv2D(16, (3, 3), padding="same")(inputs)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=-1)(x)
  x = MaxPooling2D(pool_size=(3, 3))(x)
  x = Dropout(0.25)(x)

  x = Conv2D(32, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=-1)(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Dropout(0.25)(x)

  x = Conv2D(32, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=-1)(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Dropout(0.25)(x)
  # MLP part
  x = Flatten()(x)
  x = Dense(144)(x)
  x = Activation("relu")(x)
  x = BatchNormalization()(x)

  x = Dropout(0.25)(x)
  x = Dense(60)(x)
  x = Activation("relu")(x)
  x = BatchNormalization()(x)
  x = Dropout(0.25)(x)

  x = Dense(720)(x)
  x = Activation("softmax")(x)

  model = Model(inputs=inputs,
                outputs = [x],
            )

  return model


# %%
class_mod = classification_model(input_shape=(150, 150, 1))
class_mod.summary()

# %% [markdown]
# Compile

# %%
class_mod.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'],
                  steps_per_execution=100,
                  )

# %% [markdown]
# #### Create training and test sets

# %%
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(data, labels_1min, test_size=0.2, random_state=42)  # random state is to set a seed

# %% [markdown]
# #### Train the model

# %%
history_model = class_mod.fit(X_train_classification/255.0, y_train_classification, # rescale the data to a (0, 1) range
                              batch_size=32, validation_data=(X_test_classification/255.0, y_test_classification),
                              epochs=100, shuffle=True)

class_mod.save_weights('/content/drive/MyDrive/IDL/A1/weights_model_with_shuffle_2a.keras')

# %%
train_acc_2a = history_model.history['accuracy']
test_acc_2a = history_model.history['val_accuracy']

my_array = np.array(train_acc_2a)
filename = 'train_acc_2a.npy'
np.save('/content/drive/MyDrive/IDL/A1'+filename, my_array)

my_array = np.array(test_acc_2a)
filename = 'test_acc_2a.npy'
np.save('/content/drive/MyDrive/IDL/A1'+filename, my_array)

# %% [markdown]
# #### Visualize our results

# %%
train_acc_2a = np.load('/content/drive/MyDrive/IDL/A1train_acc_2a.npy')
test_acc_2a = np.load('/content/drive/MyDrive/IDL/A1test_acc_2a.npy')

plt.plot(np.linspace(1, len(train_acc_2a), len(train_acc_2a)), train_acc_2a, color='blue', label='Training')
plt.plot(np.linspace(1, len(test_acc_2a), len(test_acc_2a)), test_acc_2a, color='red', label='Testing')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend()
plt.show()

# %%
# Predict
class_mod.load_weights('/content/drive/MyDrive/IDL/A1/weights_model_with_shuffle_2a.keras')
y_pred_2a = class_mod.predict(data/255.0)

# %%
# Convert the softmax probability to a 'one-hot' solution
# we need to have labels of a highest probability class.
# Therefore, we use the argmax argument as the probability
# per class, 0 to 720, is between 0 and 1. The highest value
# is the prediction of minute classification.
predictions_classification = np.argmax(y_pred_2a, axis=1)
predictions_classification.shape

# %%
residuals_classification = labels_1min - predictions_classification

# %%
plt.hist(residuals_classification, bins=60, edgecolor='black', color='blue')
plt.xlabel(r"$y_{true}-y_{pred}$")
plt.ylabel("# instances", fontsize=14)
plt.show()

# %% [markdown]
# #### How many instances are correctly classified?

# %%
len(residuals_classification[residuals_classification == 0])

# %% [markdown]
# What is the common sense error?

# %%
# First convert our predicted time into hours and minutes
def convert_to_hours_and_minutes(total_minutes):
    hours = total_minutes // 60  # Get the whole hours
    minutes = total_minutes % 60  # Get the remaining minutes
    return hours, minutes

pred_hours, pred_min = convert_to_hours_and_minutes(predictions_classification)
common_sense_err = np.array([])
for index, hour in enumerate(pred_hours):
  common_sense_err = np.append(common_sense_err, common_sense_time(labels[index], y_pred=[hour, pred_min[index]]))

plt.hist(common_sense_err, edgecolor='black', color='blue')
plt.xlabel("Common sense error", fontsize=14)
plt.ylabel("# Instances", fontsize=14)
plt.show()
print(f"The average common sense error is {round(np.mean(common_sense_err))} minutes")

# %% [markdown]
# # 2B
# 
# Make a regression model

# %% [markdown]
# #### First make the labels applicable

# %%
# Round off each minute as 1/60 = 0.0166666..., use round 3 to make regression speed up (there is no point using all decimals)
labels_regression = labels[:, 0] + np.round(labels[:, 1]/60, 3)

# %% [markdown]
# Number of instances between 0 and 1:

# %%
len(labels_regression[(labels_regression >0)&(labels_regression < 1)])

# %% [markdown]
# #### Histogram of labels:

# %%
plt.hist(labels_regression,bins=12, color='blue', edgecolor='black')
plt.ylabel("# Instances", fontsize=14)
plt.xlabel("Time [hour]", fontsize=14)
plt.show()

# %% [markdown]
# ### Create model

# %%
def regression_model(input_shape, default: bool = False):
        """
        Builds a regression model for floats of time.

        This model is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.

        We have two options, default or no default. This was used for experimental uses.

        The last layer is a regression layer, output is only 1 value, activation function
        is linear.
        """
        inputs = Input(shape=input_shape)
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)

        if default:
          x = Dense(128)(x)
          x = Activation("relu")(x)
          x = BatchNormalization()(x)
          x = Dropout(0.5)(x)
          x = Dense(1)(x)
          x = Activation("linear", name="regression_output")(x)

        else:
          x = Dense(256)(x)
          x = Activation("relu")(x)
          x = BatchNormalization()(x)
          x = Dropout(0.5)(x)
          x = Dense(144)(x)
          x = Activation("relu")(x)
          x = BatchNormalization()(x)
          x = Dropout(0.5)(x)
          x = Dense(1)(x)
          x = Activation("linear", name="regression_output")(x)

        model = Model(inputs=inputs,
                     outputs = [x],
                  )

        return model


model_regression_default = regression_model(input_shape=(150,150, 1), default = True)

# %% [markdown]
# Display the model

# %%
model_regression_default.summary()

# %% [markdown]
# Compile it: use a loss function of mean squared error (also try common sense error)

# %%
model_regression_default.compile(loss='mse', # we have equal distribution of data, so no outliers (no need for MAE as loss function)
              optimizer=keras.optimizers.Adam(learning_rate=0.0005),
              metrics=['mae', 'acc'],
              steps_per_execution=50,
              )

# %% [markdown]
# Split the train and test data - use 80%/20%

# %%
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(data, labels_regression, test_size=0.2, random_state=42)

# %% [markdown]
# Fit the model

# %%
model_regression_default_with_shuffle = model_regression_default.fit(X_train_regression/255.0, y_train_regression,
                                                                      batch_size=32, validation_data=(X_test_regression/255.0, y_test_regression),
                                                                      epochs=50, shuffle=True)

# %% [markdown]
# Save the weights

# %%
model_regression_default.save_weights('/content/drive/MyDrive/IDL/A1/weights_model_regression_with_shuffle_2b.keras')

# %% [markdown]
# Visualize the performance

# %%
train_mae = model_regression_default_with_shuffle.history['mae']
test_mae = model_regression_default_with_shuffle.history['val_mae']

# %%
my_array = np.array(train_mae)
filename = 'train_mae_2b.npy'
np.save('/content/drive/MyDrive/IDL/A1'+filename, my_array)

my_array = np.array(test_mae)
filename = 'test_mae_2b.npy'
np.save('/content/drive/MyDrive/IDL/A1'+filename, my_array)

# %%
train_mae = np.load('/content/drive/MyDrive/IDL/A1train_mae_2b.npy')
test_mae = np.load('/content/drive/MyDrive/IDL/A1test_mae_2b.npy')

plt.plot(np.linspace(1, len(train_mae), len(train_mae)), train_mae, color='blue', label='Training')
plt.plot(np.linspace(1, len(train_mae), len(train_mae)), test_mae, color='red', label='Testing')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("MAE", fontsize=14)
plt.legend()
plt.yscale('log')
plt.show()

# %% [markdown]
# Evaluate the model. Check if we are returned with a float between 0 and 11.9999....

# %%
model_regression_default.load_weights('/content/drive/MyDrive/IDL/A1/weights_model_regression_with_shuffle_2b.keras')
y_pred = model_regression_default.predict(data/255.0)

# %%
residuals = abs(labels_regression - y_pred.reshape(18000))
np.mean(residuals)

# %%
pd.DataFrame(residuals).describe()

# %%
plt.hist(residuals, bins=100, edgecolor='black', color='blue')
plt.xlabel(r"$y_{pred} - y_{true}$")
plt.ylabel("Number of instances")
plt.show()


y_pred_training = model_regression_default.predict(X_train_regression/255.0)
residuals_training = abs(y_train_regression - y_pred_training.reshape(int(18000*0.8)))

y_pred_testing = model_regression_default.predict(X_test_regression/255.0)
residuals_testing = abs(y_test_regression - y_pred_testing.reshape(int(18000*0.2)))

# Visualize
plt.hist(residuals_training, bins=100, edgecolor='black', color='blue', label='Training')
plt.hist(residuals_testing, bins=26, edgecolor='black', color='red', label='Testing')
plt.xlabel(r"$\mathrm{|y_{pred} - y_{true}|}$ [hour]", fontsize=14)
plt.ylabel("# Instances", fontsize=14)
plt.xlim(0, 12)
plt.legend()
plt.show()

# %% [markdown]
# #### Common sense error:

# %%
def convert_hours_array_to_hours_and_minutes(hours_array):
    result = []
    for hours_input in hours_array:
        whole_hours = int(hours_input)  # Extract the whole hours
        remaining_decimal = hours_input - whole_hours  # Calculate the decimal part
        minutes = int(remaining_decimal * 60)  # Convert decimal to minutes
        result.append((whole_hours, minutes))

    return result

hour_min_regr = convert_hours_array_to_hours_and_minutes(y_pred.reshape(18000))

common_sense_err_regr = np.array([])
for index, prediction in enumerate(hour_min_regr):
   common_sense_err_regr = np.append(common_sense_err_regr, common_sense_time(labels[index], y_pred=prediction))

plt.hist(common_sense_err_regr, edgecolor='black', color='blue')
plt.xlabel("Common sense error", fontsize=14)
plt.ylabel("# Instances", fontsize=14)
plt.show()
print(f"The average common sense error is {round(np.mean(common_sense_err_regr))} minutes")

# %% [markdown]
# # 2C
# 
# Idea: make 2 branches after the input, one branch for hours, one for minutes.

# %%
class MultiHeadModel():
    """
    This CNN contains two branches, one for hours, and the other for
    minutes. Each branch contains a sequence blocks of convolution,
    maxpooling, batch normalization and dropout.
    """
    def make_hidden_layers(self, inputs):
        """
        This function creates a sequence of convolution,
        batch normalization, maxpooling and dropout for 3
        blocks. We can use it as a default amount of hidden
        layers for any network.
        """
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        return x

    def build_minutes_branch(self, inputs):
        """
        Builds the minutes branch. For this, we use the structure of
        our model from task 2b (regression) as inspiration for this branch.

        This branch is composed of convolution -> maxpooling -> BN -> Dropout blocks,
        followed by the dense output layers.

        Last layer is a regression layer, the output is only 1 neuron, the activation
        function is linear.
        """
        x = Conv2D(64, (10, 10), padding="same")(inputs)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization(axis=-1)(x)

        x = Conv2D(64, (5, 5), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization(axis=-1)(x)

        x = Conv2D(64, (5, 5), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization(axis=-1)(x)

        x = Conv2D(64, (5, 5), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization(axis=-1)(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(64)(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("linear", name="minute_output")(x)
        return x

    def build_hour_branch(self, inputs, num_hours=12):
        """
        Builds the hour branch. This branch uses the make_hidden_layers
        function defined at the start of the class. After this, we flatten
        our model to go to the dense layers.

        The last layer is softmax, as we want to classify each clock's hour
        using a hour class.

        We define num_hours as the number of classification classes, default is 12,
        but the function can be used for any kind of number of classification classes.
        """
        x = self.make_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(144)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(60)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(num_hours)(x)
        x = Activation("softmax", name="hour_output")(x)
        return x

    def assemble_full_model(self, input_shape, num_hours):
        inputs = Input(shape=input_shape)
        hour_branch = self.build_hour_branch(inputs, num_hours)
        minute_branch = self.build_minutes_branch(inputs)
        model = Model(inputs=inputs, outputs = [hour_branch, minute_branch])
        return model

model = MultiHeadModel().assemble_full_model(input_shape=(150, 150, 1), num_hours=12)

# %% [markdown]
# #### Summary of model:

# %%
model.summary()

# %% [markdown]
# #### Image of model:

# %%
display(keras.utils.plot_model(model,  rankdir="TR",show_layer_activations=True, show_shapes=True))

# %% [markdown]
# #### Compile the model
# 
# Use sparse categorical cross entropy for the hour class, since we have a classification problem. Furthermore, for the regression problem, use the mean squared error. As optimizer use Adam, it is a fine working optimizer, and has a high accuracy for task 1, which also included images. In addition, use the accuracy and mean absolute error. Utilize our TPU by setting the steps per execution to 50.

# %%
model.compile(loss={'hour_output': 'sparse_categorical_crossentropy',
                    'minute_output': 'mse'},
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics={'hour_output': 'accuracy',
                    'minute_output': 'mae'},
              steps_per_execution=50,
              )

# %% [markdown]
# #### It is 'time' to make 2 labelled classes, we have an hours class, this is the first column of our labeled set, labels[:, 0]. Our second labels are found for minutes. Furthermore, make 80/20% train and test sets.

# %%
labels_h = labels[:,0]
labels_m = labels[:,1]

# Split the data
X_train, X_test, y_train_m, y_test_m, y_train_h, y_test_h = train_test_split(data, labels_m, labels_h, test_size=0.2, random_state=42)

# %% [markdown]
# ####Fit the model using two kinds of labels

# %%
history_2c = model.fit(X_train/255.0, [y_train_h, y_train_m], batch_size=32, validation_data=(X_test/255.0, [y_test_h, y_test_m]), epochs=50, shuffle=True)

# model.save_weights('/content/drive/MyDrive/IDL/A1/weights_model_with_shuffle_2c_V2.keras')



# %% [markdown]
# Save the model weights

# %%
# model.save_weights('/content/drive/MyDrive/IDL/A1/weights_model_with_shuffle_2c_V2.keras')

# %%
# Save the values of mae for training and validation
train_mae_2c_minutes = history_2c.history['minute_output_mae']
test_mae_2c_minutes = history_2c.history['val_minute_output_mae']

train_acc_2c_hour = history_2c.history['hour_output_accuracy']
test_acc_2c_hour = history_2c.history['val_hour_output_accuracy']

my_array = np.array(train_mae_2c_minutes)
filename = 'train_mae_2c.npy'
np.save('/content/drive/MyDrive/IDL/A1'+filename, my_array)

my_array = np.array(train_acc_2c_hour)
filename = 'train_accuracy_2c.npy'
np.save('/content/drive/MyDrive/IDL/A1'+filename, my_array)

my_array = np.array(test_mae_2c_minutes)
filename = 'test_mae_2c.npy'
np.save('/content/drive/MyDrive/IDL/A1'+filename, my_array)

my_array = np.array(test_acc_2c_hour)
filename = 'test_acc_2c.npy'
np.save('/content/drive/MyDrive/IDL/A1'+filename, my_array)

# %% [markdown]
# #### Load in the weights from the saved file

# %%
model.load_weights('/content/drive/MyDrive/IDL/A1/weights_model_with_shuffle_2c_V2 (2).keras')

# %%
# Predict
y_pred_2c_all = model.predict(data/255.0)
y_pred_2c_testing = model.predict(X_test/255.0)
y_pred_2c_training = model.predict(X_train/255.0)

# %%
y_pred_2c_all[1].shape

# %%
# Convert the softmax probability to a 'one-hot' solution
# we need to have labels of a highest probability class.
# Therefore, we use the argmax argument as the probability
# per class, 0 to 11, is between 0 and 1. The highest value
# is the prediction of hours.

predictions_h_all = np.argmax(y_pred_2c_all[0], axis=1)

# %%
## Plots including test and training set separation

predictions_h_test = np.argmax(y_pred_2c_testing[0], axis=1)
residuals_2c_h_test = predictions_h_test - y_test_h
residuals_2c_m_test = abs(y_pred_2c_testing[1].reshape(3600) - y_test_m)

predictions_h_train= np.argmax(y_pred_2c_training[0], axis=1)
residuals_2c_h_train = predictions_h_train - y_train_h
residuals_2c_m_train = abs(y_pred_2c_training[1].reshape(int(18000*0.8)) - y_train_m)

# # Visualize
plt.hist(residuals_2c_m_train, bins=30, edgecolor='black', color='blue', label='Training')
plt.hist(residuals_2c_m_test, bins=30, edgecolor='black', color='red', label='Testing')
plt.xlabel(r"$\mathrm{|y_{pred} - y_{true}|}$ [minute]", fontsize=14)
plt.ylabel("# Instances", fontsize=14)
# plt.xlim(0, 12)
plt.legend()
plt.show()


plt.hist(residuals_2c_h_train, bins=12, edgecolor='black', color='blue', label='Training')
plt.hist(residuals_2c_h_test, bins=12, edgecolor='black', color='red', label='Testing')
plt.xlabel(r"$\mathrm{y_{pred} - y_{true}}$ [hour]", fontsize=14)
plt.ylabel("# Instances", fontsize=14)
# plt.xlim(0, 12)
plt.legend()
plt.show()




# %% [markdown]
# #### How many instances are correctly classified for hours?
# 

# %%
residuals_2c_h = predictions_h_all - labels[:,0]
residuals_2c_m = abs(y_pred_2c_all[1].reshape(18000) - labels[:,1])

print("Total number of correctly classified:", len(residuals_2c_h[residuals_2c_h == 0]),
      "\nOf which in the test set:", len(residuals_2c_h_test[residuals_2c_h_test == 0]),
      "\nOf which in the training set:", len(residuals_2c_h_train[residuals_2c_h_train == 0])
      )

# %% [markdown]
# #### What is the absolute mean of the minute residuals? In addition, how many instances are between 0 and 10?

# %%
np.mean(residuals_2c_m)

# %%
len([x for x in residuals_2c_m if 0 < x < 10])

# %% [markdown]
# #### MAE and Accuracy of minutes and hours classes growth over epoch:

# %%
## plot 2

train_mae_2c = np.load('/content/drive/MyDrive/IDL/A1train_mae_2c.npy')
test_mae_2c = np.load('/content/drive/MyDrive/IDL/A1test_mae_2c.npy')

train_accuracy_2c = np.load('/content/drive/MyDrive/IDL/A1train_accuracy_2c.npy')
test_accuracy_2c = np.load('/content/drive/MyDrive/IDL/A1test_acc_2c.npy')

plt.plot(np.linspace(1, len(train_mae_2c), len(train_mae_2c)), train_accuracy_2c, color='blue', label='Training')
plt.plot(np.linspace(1, len(train_mae_2c), len(train_mae_2c)), test_accuracy_2c, color='red', label='Testing')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend()
plt.yscale('log')
plt.show()

plt.plot(np.linspace(1, len(train_mae_2c), len(train_mae_2c)), train_mae_2c, color='blue', label='Training')
plt.plot(np.linspace(1, len(train_mae_2c), len(train_mae_2c)), test_mae_2c, color='red', label='Testing')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("MAE", fontsize=14)
plt.legend()
plt.yscale('log')
plt.show()

# %%
pd.DataFrame(residuals_2c_h).describe()

# %%
pd.DataFrame(residuals_2c_m).describe()

# %% [markdown]
# #### Common sense error report:

# %%
def common_sense_time(y_true, y_pred):
  predicted_min = y_pred[0]*60+y_pred[1]
  actual_min = y_true[0]*60+y_true[1]
  # Returns the difference in units of minutes
  return (predicted_min - actual_min)%720

y_pred_2c_together = np.array(list(zip(predictions_h_all, y_pred_2c_all[1].reshape(18000).astype(int))))
y_pred_2c_together


common_sense_err_multi = np.array([])
for index, prediction in enumerate(y_pred_2c_together):
  common_sense_err_multi = np.append(common_sense_err_multi, common_sense_time(labels[index], prediction))

plt.hist(common_sense_err_multi, edgecolor='black', color='blue')
plt.xlabel("Common sense error", fontsize=14)
plt.ylabel("# Instances", fontsize=14)
plt.show()
print(f"The average common sense error is {round(np.mean(common_sense_err_multi))} minutes")


# %%



