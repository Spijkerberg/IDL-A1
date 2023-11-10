# %% [markdown]
# ## Imports
# %%
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import Input, aModel, backend as K
import numpy as np
import matplotlib.pyplot as plt
print("Tensorflow version: ", tf.__version__)
from IPython.display import display

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

import time

# Needed for saving on drive or locally
import pandas as pd
from google.colab import drive

# Enable the TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)

print("All devices: ", *tf.config.list_logical_devices("TPU"), sep="\n\t")

# %% [markdown]
# ## MNIST Examples from keras

# %%
batch_size = 128
num_classes = 10 # 0,1,...., 9. The digits/outputs
epochs = 20 # Number of runs

# %% [markdown]
# Batch size defines the number of samples that will be propagated through the network. If you have 1050 training samples
# then our algorithm will take in the training samples in batches, where each batch has the same size. However, it must
# be the case that the number of samples is divisible by the batch_size, else we have batches with fractional instances,
# these do not exist.
# 
# In addition, there are some advantages of using a batch size that is smaller than the total number of training samples:
# * We require less memory to be used. By training with fewer sampling, the overall training prodedure takes less memory.
# * Typically, networks train faster with mini-batches as opposed to the entire batch of all instances. This is a consequence of how weights are adjusted after each propagation.
# 
# Disadvantages:
# * The smaller the batches, the less accurate the estimate of the gradient will be.

# %%
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# %%
# Model initialization and fitting
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,))) # input layer
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu')) # layer 1
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax') ) # final layer, output

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'], steps_per_execution=128)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %%


# %%

# Change epoch (default)
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %%


# %% [markdown]
# ## MNIST Fashion - Exercise 1.2

# %%
# Load in data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# %%
y_train_full

# %%
# y_train_full.shape
X_train_full.shape

# %%
# X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
# y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# %%
# For the fashion list we have different labels than the MNIST digits:
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# %% [markdown]
# #### Example of an instance

# %%
# Example of instance 4
print(class_names[y_train_full[3]])
plt.imshow(X_train_full[3,:,:], cmap='gray')
plt.show()



# %% [markdown]
# # Back to the exercise:
# ## Build the MLP neural network!
# 

# %%
# Build the network layer by layer, in a SEQUENCE -> sequential
model = keras.models.Sequential()

""" Add the first layer, the input layer.

It converts each input image into a 1D array: if it receives input data X, it
computes X.reshape(-1, 1). This layer does not have any parameters; it is just
there to do some simple preprocessing. Since it is the first layer in the model, you
should specify the input_shape, which doesn’t include the batch size, only the
shape of the instances. """

model.add(keras.layers.Flatten(input_shape=[28, 28], name="InputLayer"))

# Add layer with 300 neurons. Each Dense layer manages its own weight matrix, containing all the
# connection weights between the neurons and their inputs.

model.add(keras.layers.Dense(300, activation="relu", name="HiddenLayer1"))

# And another layer, with 100 neurons
model.add(keras.layers.Dense(100, activation="relu", name="HiddenLayer2"))

# The final layer, the output layer, we have 10 classes, so we need 10 neurons, 1 per class.
# We use softmax as all classes are exclusive.

""" The network is configured to output N values, one for each class in the classification
task, and the softmax function is used to normalize the outputs, converting them from
weighted sum values into probabilities that sum to one. Each value in the output of the
softmax function is interpreted as the probability of membership for each class."""

model.add(keras.layers.Dense(10, activation="softmax", name="OutputLayer"))

# %% [markdown]
# #### Model in one sequence:

# %%
# it is also possible to do it in one go:
"""
model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
        ])
"""

# %% [markdown]
# #### Visualize our model:

# %%
# Image of our model - useful if we have named layers

display(keras.utils.plot_model(model, show_layer_activations=True))

# Or summary
model.summary()



# %% [markdown]
# The output shape can have None. This implies that the batch size can be anything.

# %% [markdown]
# ### Run the NN:

# %%
""" We have sparse labels (i.e., for each instance, there is just a target
class index, from 0 to 9 in this case), and the classes are exclusive.

Regarding the optimizer, "sgd" means that we will train the model using
simple Stochastic Gradient Descent.

As we are using a TPU, we can adjust the steps_per_execution. This
makes the running and calculating of the weights from our NN sufficiently
faster.
"""

# Choose an optimizer from keras.optimizers.X
opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=[["accuracy"], ["mse"]], steps_per_execution=100)

# Training and evaluation - epochs is the number of runs
history = model.fit(X_train_full / 255.0, y_train_full, epochs=30,
                    validation_split=0.1, batch_size=64, verbose=0) # verbose=0 is stfu

# %% [markdown]
# #### Plot the curves of loss, accuracy, validation loss and validation accuracy:

# %%
import pandas as pd

# %%
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

"""the validation curves are close to the training curves, which means
that there is not too much overfitting """

# %% [markdown]
# ## Evaluate the model:

# %%
y_test.shape

# %%
model.evaluate(X_test, y_test, batch_size=64, verbose=1)

# %% [markdown]
# # Make functions that can make arbitrary Neural Networks. In addition, make a function that can fit the model, as well as another function that can evaluate the model.
# 
# This makes creating an architecture and subsequently training it with various hyperparameters easier and more insightful.

# %%
def make_model(arch, optimizer,
               metrics: list = ["accuracy"],
               steps_per_execution: int = 1,
               summary_model: bool = False):

  # Create a dictionary of possible layers. In this case we can create any arbitary network
  # without having to constantly copy and paste lines.
  layer_dict = {
        "dense": Dense,             # units, activation=None
        "dropout": Dropout,         # rate, noise_shape=None, seed=None
        "conv": Conv2D,             # filters, kernel_size, strides=(1, 1), padding="valid", activation=None
        "maxpooling": MaxPooling2D, # pool_size=(2, 2), strides=None, padding="valid"
        "flatten": Flatten,
        "BN": BatchNormalization,
  }

  ##########################################

  # Initiate model
  model = keras.models.Sequential()

  ## Add input layer
  model.add(Input(shape=arch['shape'][0]))

  ## Add layers
  for layer, kwargs in arch['layers']:
    # Use our dictionary to obtain the callable layer and put in the keywords arguments
    model.add(layer_dict[layer](**kwargs))

  # Output layer
  model.add(layer_dict[arch['output'][0]](**arch['output'][1]))

  ##########################################

  # Print summary if wanted
  if summary_model:
    model.summary()

  model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=metrics,
                steps_per_execution=steps_per_execution)

  # Return model before fitting
  return model


def fit_model(model, X_train, y_train, scaling,
              epochs: int = 30, validation_split: float = 0.1,
              batch: int = 64, verbose: int = 0):

  history = model.fit(X_train/scaling, y_train, epochs=epochs,
                    validation_split = validation_split, batch_size=batch, verbose=verbose) # verbose=0 is no output

  return history

#superfluous function, but for clarity and consistency it is kept (artefact of learning the API)
def eval_model(model, X_test, y_test, batch, verbose=1):
  return model.evaluate(X_test, y_test, batch_size=batch, verbose=verbose)



# %% [markdown]
# # MLP case:

# %% [markdown]
# ### Define the multi layer perceptron architecture:

# %%
mlp_arch = {
    'shape': (28*28, 10), # input, output
    'output': ('dense', {'units': 10, 'activation': 'softmax'}), # output layer
    'layers': [
         ('flatten', {}),
        ('dense', {'units':300, 'activation': 'relu'}),
        ('dense', {'units':100, 'activation': 'relu'}),
        ]
}

# %% [markdown]
# ### Test run:

# %%
mod = make_model(mlp_arch, keras.optimizers.Adam(learning_rate=0.001),
                 steps_per_execution=100, summary_model=True)

# display(keras.utils.plot_model(mod, show_layer_activations=True))

# %%
trainedmodel1 = fit_model(mod, X_train_full.reshape(60000, 784), y_train_full, 255.0,
                          verbose=1)

# X_train_full.reshape(60000, 784).shape


# %%
X_train_full.reshape(60000, 784).shape

# %% [markdown]
# Evaluate our NN:

# %%
eval_model(mod, X_test.reshape(10000, 784), y_test, batch=64, verbose=1)

# %% [markdown]
# ## Time to scale up, we want to now analyse which optimizer, learning rate, batch size and epoch number give us the highest accuracy for the MNIST fashion dataset, given our MLP.

# %%
optimizers = [
     # Keyword arguments (kwargs) come in dictionaries ('{}')
      (keras.optimizers.Adam, {'learning_rate': 0.001}),

      (keras.optimizers.Adagrad, {}),
      (keras.optimizers.Adamax, {'learning_rate': 0.001}),
      (keras.optimizers.Nadam,{}),
      (keras.optimizers.RMSprop, {}),
      ]

accuracies = np.empty((0, 4))
for optimizer, kwarg in optimizers:
  # print(optimizer.__name__, kwarg)
  model1 = make_model(mlp_arch, optimizer=optimizer(**kwarg),
           steps_per_execution=100, summary_model=False)

  fit_model(model1, X_train_full.reshape(60000, 784), y_train_full, 255.0,
                          verbose=0)

  accuracies = np.append(accuracies, np.array([f"Name: {optimizer.__name__}",
                                               f"Kwargs: {kwarg}", "Accuracy:",
                                               eval_model(model1,
                                               X_test.reshape(10000, 784), y_test,
                                               batch=64, verbose=0)[1]]).reshape(1, -1), axis=0)






# %%
print(accuracies[:,-1])
best = np.argmax(np.array([float(i) for i in accuracies[:,-1]]))
print("Best model:", accuracies[:,0][best], accuracies[:,1][best], accuracies[:,2][best], accuracies[:,-1][best])


# if we have different models:
#display(keras.utils.plot_model(model, show_layer_activations=True))

# %% [markdown]
# ## Test architectures, optimizers, batch sizes and epochs for any given choice:

# %%
def test_networks(models, optimizers, batch_sizes, epochs):
  """
  Four nested four loops that fit and test all possible combinations of
  hyperparameters. Running this on NNs, especially CNNs, takes a lot of time
  so please be wary.
  """
  results = np.empty((0,5))

  for model in models:
    for optimizer, kwarg in optimizers:
      model1 = make_model(model, optimizer=optimizer(**kwarg),
           steps_per_execution=100, summary_model=False)

      for batch in batch_sizes:
        for epoch in epochs:
          fit_model(model1, X_train_full.reshape(60000, 784), y_train_full, 255.0, epochs=epoch,
                            verbose=0, batch=batch, validation_split=0.1)

          accuracy = eval_model(model1, X_test.reshape(10000, 784), y_test, batch=batch, verbose=0)[1]

          results = np.append(results, np.array([f"{optimizer.__name__ = }", f"{kwarg = }",
                                                      f"{batch = }",f"{epoch = }", f"{accuracy = }"]).reshape(1, -1), axis=0)


    return results

# %% [markdown]
# ### Results of regular MLP architecture:

# %%
results1 = test_networks(
      [mlp_arch],
       [
     # Keyword arguments (kwargs) come in dictionaries ('{}')
      (keras.optimizers.Adam, {'learning_rate': 0.0005}),
      (keras.optimizers.Adadelta, {'learning_rate': 0.001}),
      (keras.optimizers.Adagrad, {'learning_rate': 0.001}),
      (keras.optimizers.Adamax, {'learning_rate': 0.0005}),
      (keras.optimizers.Nadam,{'learning_rate': 0.0005}),
      (keras.optimizers.RMSprop, {'learning_rate': 0.001}),
      (keras.optimizers.SGD, {'learning_rate': 0.001})
      ],

      batch_sizes=[64], epochs = [10]
)

# %%
# Find the best model

accuracies = [float(i.split('=')[1].strip()) for i in results1[:,-1]]
best = np.argmax(np.array([accuracies]))

print("Best model:", "\n")
for i in np.arange(results1.shape[-1]):
  print(results1[:,i][best])


# %%
results = [np.argsort(accuracies)[:][::-1]][0] # [::-1] is for reverse order, such that the highest accuracy is value 1 in the top5

for i in results:
  print("\nModel:")
  for j in np.arange(results1.shape[-1]):
    print(results1[:,j][i])


# %% [markdown]
# #### Full output:

# %% [markdown]
# In addition, the full ouput of results1 was:
# 
#       ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 32',
#         'epoch = 30', 'accuracy = 0.8683000206947327']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 64',
#         'epoch = 5', 'accuracy = 0.8709999918937683']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 64',
#         'epoch = 10', 'accuracy = 0.8736000061035156']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 64',
#         'epoch = 20', 'accuracy = 0.8697999715805054']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 64',
#         'epoch = 30', 'accuracy = 0.873199999332428']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 128',
#         'epoch = 5', 'accuracy = 0.8690999746322632']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 128',
#         'epoch = 10', 'accuracy = 0.8729000091552734']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 128',
#         'epoch = 20', 'accuracy = 0.871999979019165']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 128',
#         'epoch = 30', 'accuracy = 0.8683000206947327']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 256',
#         'epoch = 5', 'accuracy = 0.8734999895095825']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 256',
#         'epoch = 10', 'accuracy = 0.8725000023841858']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 256',
#         'epoch = 20', 'accuracy = 0.866599977016449']
# 
#        ["optimizer.__name__ = 'RMSprop'", 'kwarg = {}', 'batch = 256',
#         'epoch = 30', 'accuracy = 0.8697999715805054']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 32',
#         'epoch = 5', 'accuracy = 0.8222000002861023']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 32',
#         'epoch = 10', 'accuracy = 0.8388000130653381']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 32',
#         'epoch = 20', 'accuracy = 0.8525999784469604']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 32',
#         'epoch = 30', 'accuracy = 0.8587999939918518']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 64',
#         'epoch = 5', 'accuracy = 0.843500018119812']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 64',
#         'epoch = 10', 'accuracy = 0.8406000137329102']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 64',
#         'epoch = 20', 'accuracy = 0.8314999938011169']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 64',
#         'epoch = 30', 'accuracy = 0.8364999890327454']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 128',
#         'epoch = 5', 'accuracy = 0.848800003528595']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 128',
#         'epoch = 10', 'accuracy = 0.8391000032424927']
# 
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 128',
#         'epoch = 20', 'accuracy = 0.8314999938011169']
#         
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 128',
#         'epoch = 30', 'accuracy = 0.8317999839782715']
#         
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 256',
#         'epoch = 5', 'accuracy = 0.8378000259399414']
#         
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 256',
#         'epoch = 10', 'accuracy = 0.8388000130653381']
#         
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 256',
#         'epoch = 20', 'accuracy = 0.8361999988555908']
#         
#        ["optimizer.__name__ = 'SGD'", 'kwarg = {}', 'batch = 256',
#         'epoch = 30', 'accuracy = 0.8353000283241272']
#        

# %% [markdown]
# #### Let's go again, this time using different epochs and batch sizes:

# %%
def Top5(results):
  accuracies = [float(i.split('=')[1].strip()) for i in results[:,-1]]
  top5 = [np.argsort(accuracies)[-5:][::-1]][0] # [::-1] is for reverse order, such that the highest accuracy is value 1 in the top5

  for i in top5:
    print("\nModel:")
    for j in np.arange(results.shape[-1]):
      print(results[:,j][i])

# %%
results2 = test_networks(
      [mlp_arch],
    [
     ## Keyword arguments (kwargs) come in dictionaries ('{}')
      (keras.optimizers.Adam, {'learning_rate': 0.0005}), (keras.optimizers.Adam, {'learning_rate': 0.001}), (keras.optimizers.Adam, {'learning_rate':0.002}),
      (keras.optimizers.Adagrad, {'learning_rate': 0.001}),
      (keras.optimizers.Adamax, {'learning_rate': 0.0005}), (keras.optimizers.Adamax,  {'learning_rate': 0.001}), (keras.optimizers.Adamax,  {'learning_rate': 0.002}),
      (keras.optimizers.Nadam,{'learning_rate': 0.0005}), (keras.optimizers.Nadam,{'learning_rate': 0.001}), (keras.optimizers.Nadam,{'learning_rate': 0.002}),
      (keras.optimizers.RMSprop, {'learning_rate': 0.001})],
      batch_sizes=[32, 64, 128, 256],
      epochs = [5, 10, 20, 30]
)

# %%
Top5(results2)

# %% [markdown]
# ### Two scenario's:
# * remove the 100 nodes layer, rerun the calculation.
# * also try adding a layers of dropout to the original network. Add dropout between the last layer and the output layer, as well as between the second to last and the last (regular) layer

# %%
#only the 300 nodes layer
mlp_arch_rmove = {
    'shape': (28*28, 10), # input, output
    'output': ('dense', {'units': 10, 'activation': 'softmax'}), # output layer
    'layers': [
         ('flatten', {}),
        ('dense', {'units':300, 'activation': 'relu'}),
        ]
}

# dropout
mlp_arch_drop = {
    'shape': (28*28, 10), # input, output
    'output': ('dense', {'units': 10, 'activation': 'softmax'}), # output layer
    'layers': [
        ('flatten', {}),
        ('dense', {'units':300, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),
        ('dense', {'units':100, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),
        ]
}

#dropout with only the 300 nodes dense layer
mlp_arch_drop_rmove = {
    'shape': (28*28, 10), # input, output
    'output': ('dense', {'units': 10, 'activation': 'softmax'}), # output layer
    'layers': [
        ('flatten', {}),
        ('dense', {'units':300, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5})
        ]
}

# Show models:

# display(keras.utils.plot_model(make_model(mlp_arch_rmove, optimizer=keras.optimizers.Adam(),
#            steps_per_execution=100, summary_model=True), show_layer_activations=True))

# display(keras.utils.plot_model(make_model(mlp_arch_drop, optimizer=keras.optimizers.Adam(),
#            steps_per_execution=100, summary_model=True), show_layer_activations=True))

# %%
results_mlp_rmoveL = test_networks(
      [mlp_arch_rmove],

    [
      (keras.optimizers.Adam, {'learning_rate': 0.0005}), (keras.optimizers.Adam, {'learning_rate': 0.001}),
      (keras.optimizers.Nadam,{'learning_rate': 0.0005}), (keras.optimizers.Nadam,{'learning_rate': 0.001})],
      batch_sizes=[32, 64, 128, 256],
      epochs = [10, 20, 30]
)

# %%
Top5(results_mlp_rmoveL)

# %% [markdown]
# Remark: we see that removing a layer lowers our final accuracy. In this case, our highest scoring model is about equivalent to the model of position 5 for the original model. It shows that we lose performance by removing de 100 nodes layer.

# %%
## Using the dropout model:
results_mlp_dropout = test_networks([mlp_arch_drop],
      [(keras.optimizers.Adam, {'learning_rate': 0.0005}), (keras.optimizers.Adam, {'learning_rate': 0.001}),
      (keras.optimizers.Nadam,{'learning_rate': 0.0005}), (keras.optimizers.Nadam,{'learning_rate': 0.001})],
              batch_sizes=[32, 64, 128, 256],
                       epochs = [10, 20, 30],

                       )


# %%
Top5(results_mlp_dropout)

# %%
results_mlp_dropout1rmove = test_networks([mlp_arch_drop_rmove],
                       [
      (keras.optimizers.Adam, {'learning_rate': 0.0005}),
      (keras.optimizers.Nadam,{'learning_rate': 0.0005}),
                       ],
              batch_sizes=[64, 128, 256],
              epochs = [10, 20, 30],
                       )

Top5(results_mlp_dropout1rmove)

# %% [markdown]
# Remark: we see that the performance has dropped a little bit, but let's go and try again, but this time using one layer of dropout 0.5. In addition, let's just use the best hyperparameters set of the previous result to speed up the process.

# %%
mlp_arch_dropout_1 = {
    'shape': (28*28, 10), # input, output
    'output': ('dense', {'units': 10, 'activation': 'softmax'}), # output layer
    'layers': [
        ('flatten', {}),
        ('dense', {'units':300, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),
        ('dense', {'units':100, 'activation': 'relu'}),
        ]
}

results_mlp_dropout_1 = test_networks([mlp_arch_dropout_1],
                       [
      (keras.optimizers.Adam, {'learning_rate': 0.0005}),
      (keras.optimizers.Nadam,{'learning_rate': 0.0005}),
                       ],
              batch_sizes=[64, 128, 256],
              epochs = [10, 20, 30])

Top5(results_mlp_dropout_1)

# %% [markdown]
# ### Another case, use the dropout model together with regularization.

# %% [markdown]
# About L1 and L2 regularization: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c .
# 
# We use L1 regularization for the layers with the largest number of nodes. L1 regularization helps in order to penalize for a large number of features. We apply this to our best architecture yet to see if it has any significant improvement.

# %%
# , kernel_regularizer='l1'
mlp_arch_l1_drop = {
    'shape': (28*28, 10), # input, output
    'output': ('dense', {'units': 10, 'activation': 'softmax'}), # output layer
    'layers': [
         ('flatten', {}),
        ('dense', {'units':300, 'activation': 'relu', 'kernel_regularizer': 'l1'}),
         ('dropout', {'rate': 0.5}),
        ('dense', {'units':100, 'activation': 'relu'}),
        ]
}

# display(keras.utils.plot_model(make_model(mlp_arch_l1_drop, optimizer=keras.optimizers.Adam(),
#            steps_per_execution=100, summary_model=True), show_layer_activations=True))

# %%
results_mlp_L1_dropout = test_networks([mlp_arch_l1_drop],
                       [
      (keras.optimizers.Adam, {'learning_rate': 0.0005}),
      (keras.optimizers.Nadam,{'learning_rate': 0.0005}),
                       ],
              batch_sizes=[64, 128, 256],
              epochs = [10, 20, 30],
                       )

# %%
Top5(results_mlp_L1_dropout)

# %% [markdown]
# Remark: L1 regularization was a BAD idea!

# %% [markdown]
# # Convolutional network:

# %% [markdown]
# #### The architecture:

# %%
convolution_model ={
    'shape':((28, 28, 1), 10), # input, output (-> we have 10 classes of clothes, each with a given probability per item (returned by Softmax))
    'output': ('dense', {'units': 10, 'activation': 'softmax'}),
    'layers':[
        ('conv', {'filters':64, 'kernel_size': 7, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        # by reducing the spacial dimensionality by two we can afford to double the filters (and thus feature maps)
        ('conv', {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('conv', {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        ('conv', {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('conv', {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        ('flatten', {}),
        ('dense', {'units': 128, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),
        ('dense', {'units': 64, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),
    ]
}

# display(keras.utils.plot_model(make_model(convolution_model, optimizer=keras.optimizers.Adam(),
#            steps_per_execution=100, summary_model=True), show_layer_activations=True))

# %% [markdown]
# ### Let's test!

# %%
# Similar function to before. Takes very long to run.
def test_networks_V2(models, optimizers, batch_sizes, epochs,
                     input_shape, X_train, y_train, X_test, y_test,
                     steps_per_execution: int = 100,
                     print_steps: bool = False, time_check: bool = False,
                     filename: str = None, save: bool = False):

  # Check if a file with specifcations exists; if it does, load the existing data
  try:
      result_df = pd.read_csv(filename)
      print(f"Found {filename = }")
  except FileNotFoundError:
      print(f"Could not find {filename =}")
      result_df = pd.DataFrame(columns=["Name", "Kwargs", "Batch_size", "Epochs", "Accuracy"])


  for model in models:
    for optimizer, kwarg in optimizers:

      if print_steps:
        print(f"Fitting model using {optimizer.__name__ = }")

      model1 = make_model(model, optimizer=optimizer(**kwarg),
           steps_per_execution=steps_per_execution, summary_model=False)

      for batch in batch_sizes:
        for epoch in epochs:
          if print_steps:
            print(f"Epochs = {epoch}, batch size = {batch}")

          if time_check:
            start_time = time.time()

          fit_model(model1, X_train.reshape(X_train.shape[0], *input_shape), y_train, 255.0, epochs=epoch,
                            verbose=0, batch=batch, validation_split=0.1)

          accuracy = eval_model(model1, X_test.reshape(X_test.shape[0], *input_shape), y_test, batch=batch, verbose=0)[1]

          if time_check:
            print(f"Time elapsed for {optimizer.__name__ = }, {kwarg = }, {batch = }, {epoch= } is {(time.time()-start_time)/60} min")

          # Write to file
          new_data = {"Name": optimizer.__name__, "Kwargs": kwarg,
                      "Batch_size": batch, "Epochs": epoch,
                      "Accuracy": accuracy}

          # Append to the existing file
          result_df = result_df.append(new_data, ignore_index=True)

          # Save the DataFrame to a .csv file after each iteration
          if save:
            result_df.to_csv(filename, index=False)

    return result_df

# %% [markdown]
# #### Running for convolutional network:

# %%
# If you want to save:
#!cp IDL_A1_convolution_default.csv /content/drive/MyDrive/IDL/A1/IDL_A1_convolution_default.csv

# Or download locally
from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
#files.download('IDL_A1_convolution_default.csv')

df_loaded_from_drive = pd.read_csv('/content/drive/MyDrive/IDL/IDL_A1_convolution_default (1).csv')
df_loaded_from_drive

# %% [markdown]
# Idea: implement batch normalization. It speeds up the process a lot, as well as that it should help to gain higher accuracy for our model. Add Batch Normalization after a sequence of convolutions and maxpooling (this is done often in practise; found online, https://analyticsindiamag.com/everything-you-should-know-about-dropouts-and-batchnormalization-in-cnn/#:~:text=It%20can%20be%20used%20at,the%20classification%20of%20handwritten%20digits.)

# %%
convolution_model_modified ={
    'shape':((28,28, 1), 10),
    'output': ('dense', {'units': 10, 'activation': 'softmax'}),
    'layers':[
        ('conv', {'filters':64, 'kernel_size': 7, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        ('conv', {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('conv', {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        ('conv', {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('conv', {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        ('BN', {}), # New layer of batch normalization
        ('flatten', {}),
        ('dense', {'units': 128, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),
        ('dense', {'units': 64, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),

    ]
}

# %%
df_CNN_BN = pd.read_csv('/content/drive/MyDrive/IDL/IDL_A1_convolution_with_BN.csv')
df_CNN_BN

# %% [markdown]
# Remove batch normalization and rerun the CNN with a learning rate of 0.0005 and Adamax.

# %%
test_networks_V2([convolution_model],
                       [
                        (keras.optimizers.Adamax,  {'learning_rate': 0.0005}),
                       ],

                        batch_sizes = [64, 128, 256, 512],# the smaller size speeds up the process, at the cost of performance
                        epochs = [20], #5, 10, 20
                        input_shape = convolution_model['shape'][0],
                        X_train = X_train_full,
                        y_train = y_train_full,
                        X_test = X_test,
                        y_test = y_test,

                        print_steps=True,
                        time_check=True,
                        filename='/content/drive/MyDrive/IDL/A1/IDL_A1_convolution_default.csv',
                        steps_per_execution=1000,
                        save=True
                       )

# %%
from google.colab import drive
drive.mount('/content/drive')

df_loaded_from_drive = pd.read_csv(r'/content/drive/MyDrive/IDL/IDL_A1_convolution_default (1).csv')
df_loaded_from_drive['Accuracy']

# %%
def Accuracy_Epoch_batch_plot(df_results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Create a colormap based on unique names
    unique_names = df_results['Name'].unique()
    num_colors = len(unique_names)
    colormap = plt.cm.get_cmap('plasma', num_colors)

    # Plot something in the first subplot
    for i, (name, group) in enumerate(df_results.groupby('Name')):
        axes[0].plot(group['Epochs'], group['Accuracy'], ".", label=name, color=colormap(i / (num_colors - 1)))

    axes[0].set_title('Epochs VS Accuracy',fontsize=18)
    axes[0].set_xlabel("Epochs",fontsize=16)
    axes[0].set_ylabel("Accuracy",fontsize=16)
    axes[0].set_facecolor('lightgray')
    axes[0].grid()

    # Plot something in the second subplot
    for i, (name, group) in enumerate(df_results.groupby('Name')):
        axes[1].plot(group['Batch_size'], group['Accuracy'], ".", label=name, color=colormap(i / (num_colors - 1)))

    axes[1].set_title('Batch size VS Accuracy',fontsize=18)
    axes[1].set_xlabel("Batch size",fontsize=16)
    axes[1].legend(loc=(1.05, 0.1),fontsize=16)
    axes[1].set_facecolor('lightgray')
    axes[1].grid()

    plt.tight_layout()
    # Display the plot
    plt.savefig(r'/content/drive/MyDrive/IDL/CNNplot.png',dpi=300,bbox_inches='tight')
    plt.show()



Accuracy_Epoch_batch_plot(df_loaded_from_drive)

# %%
def Top3(results):
  accuracies = results['Accuracy']
  top3 = [np.argsort(accuracies)[-3:][::-1]][0] # [::-1] is for reverse order, such that the highest accuracy is value 1 in the top5
  for i in top3:
    print("\nModel:")
    print(results.iloc[i])

Top3(df_loaded_from_drive)

# %% [markdown]
# # CIFAR 10 dataset

# %% [markdown]
# Use the 3 best models to train new models on the CIFAR 10 data set. This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. The objects are vehicles and animals.
# 
# Apply our best 3 architectures and hyperparameters (incl. optimizer).

# %%
(X_train_CIFAR, y_train_CIFAR), (X_test_CIFAR, y_test_CIFAR) = keras.datasets.cifar10.load_data()

# %%
X_train_CIFAR.shape

# %%
# plot an instance
plt.imshow(X_train_CIFAR[0,:,:,:])
print(y_train_CIFAR[0])

# %%
y_train_CIFAR.shape

# %% [markdown]
# Note we must change our input shape as we now have 32x32x3 images instead of 28x28x1. We don't touch anything else.

# %%
convolution_model_cifar ={
    'shape':((32, 32, 3), 10),
    'output': ('dense', {'units': 10, 'activation': 'softmax'}),
    'layers':[
        ('conv', {'filters':64, 'kernel_size': 7, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        ('conv', {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('conv', {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        ('conv', {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('conv', {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}),
        ('maxpooling', {'pool_size': (2,2)}),
        ('flatten', {}),
        ('dense', {'units': 128, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),
        ('dense', {'units': 64, 'activation': 'relu'}),
        ('dropout', {'rate': 0.5}),

    ]
}

# %%
adamax_b128_e10_le_low = test_networks_V2([convolution_model_cifar],
                       [
                        (keras.optimizers.Adamax,  {'learning_rate': 0.0005}),
                       ],

                        batch_sizes = [128],
                        epochs = [10],
                        input_shape = convolution_model_cifar['shape'][0],
                        X_train = X_train_CIFAR,
                        y_train = y_train_CIFAR,
                        X_test = X_test_CIFAR,
                        y_test = y_test_CIFAR,

                        print_steps=True,
                        time_check=True,
                       filename='none',
                        steps_per_execution=1000,
                        save=False

                       )

# %%
adamax_b256_e10_le_default = test_networks_V2([convolution_model_cifar],
                       [
                        (keras.optimizers.Adamax,  {'learning_rate': 0.001}),
                       ],

                        batch_sizes = [256],
                        epochs = [10],
                        input_shape = convolution_model_cifar['shape'][0],
                        X_train = X_train_CIFAR,
                        y_train = y_train_CIFAR,
                        X_test = X_test_CIFAR,
                        y_test = y_test_CIFAR,

                        print_steps=True,
                        time_check=True,
                       filename='none',
                        steps_per_execution=1000,
                        save=False

                       )

# %%
adamax_b512_e20_le_default = test_networks_V2([convolution_model_cifar],
                       [
                        (keras.optimizers.Adamax,  {'learning_rate': 0.001}),
                       ],

                        batch_sizes = [512],
                        epochs = [20],
                        input_shape = convolution_model_cifar['shape'][0],
                        X_train = X_train_CIFAR,
                        y_train = y_train_CIFAR,
                        X_test = X_test_CIFAR,
                        y_test = y_test_CIFAR,

                        print_steps=True,
                        time_check=True,
                       filename='none',
                        steps_per_execution=1000,
                        save=False

                       )

# %%
top3_on_CIFAR = pd.concat([adamax_b128_e10_le_low,
                           adamax_b256_e10_le_default,
                           adamax_b512_e20_le_default], axis=1)

top3_on_CIFAR


