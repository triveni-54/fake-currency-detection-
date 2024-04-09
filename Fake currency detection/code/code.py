#!/usr/bin/env python
# coding: utf-8

# VGG NET TRANSFER LEARNING - ACCURACY - 86/96 

# In[11]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model (optional)
for layer in base_model.layers:
    layer.trainable = False

# Create a new model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation for training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess data using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    '/Users/sriiakhillessh/Downloads/mini project/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Validation data generator (similar preprocessing as above)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    '/Users/sriiakhillessh/Downloads/mini project/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Evaluate the model on the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/Users/sriiakhillessh/Downloads/mini project/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')


# In[10]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('/Users/sriiakhillessh/Downloads/mini project')

# Constants
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)

# Function to preprocess and predict the custom image
def predict_custom_image():
    # Open a file dialog for the user to select an image
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # Load and preprocess the custom image
        custom_img = load_img(file_path, target_size=IMAGE_SIZE)
        custom_img = img_to_array(custom_img)
        custom_img = np.expand_dims(custom_img, axis=0)
        custom_img = preprocess_input(custom_img)  # Preprocess input as per VGG16
        
        # Perform inference
        predictions = model.predict(custom_img)
        
        # Post-process the output with custom data
        custom_data = ['100', '200', '500', '2000']  # Replace with your custom class labels
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_class = custom_data[predicted_class_index[0]]
        
        # Display the predicted class in the GUI
        result_label.config(text=f'Predicted Class: {predicted_class}')

# Create the GUI window
root = tk.Tk()
root.title('Image Classification')

# Create a button to select an image
select_button = tk.Button(root, text='Select Image', command=predict_custom_image)
select_button.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(root, text='', font=("Helvetica", 16))
result_label.pack()

# Start the GUI main loop
root.mainloop()


# RESNET 50 ACCURACY 58%

# In[12]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50  # Import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Use ResNet50

# Freeze the layers of the pre-trained model (optional)
for layer in base_model.layers:
    layer.trainable = False

# Create a new model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation for training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess data using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    '/Users/sriiakhillessh/Downloads/mini project/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Validation data generator (similar preprocessing as above)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    '/Users/sriiakhillessh/Downloads/mini project/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Evaluate the model on the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/Users/sriiakhillessh/Downloads/mini project/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# The rest of your code for the GUI remains the same.
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('/Users/sriiakhillessh/Downloads/mini project')

# Constants
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)

# Function to preprocess and predict the custom image
def predict_custom_image():
    # Open a file dialog for the user to select an image
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # Load and preprocess the custom image
        custom_img = load_img(file_path, target_size=IMAGE_SIZE)
        custom_img = img_to_array(custom_img)
        custom_img = np.expand_dims(custom_img, axis=0)
        custom_img = preprocess_input(custom_img)  # Preprocess input as per VGG16
        
        # Perform inference
        predictions = model.predict(custom_img)
        
        # Post-process the output with custom data
        custom_data = ['100', '200', '500', '2000']  # Replace with your custom class labels
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_class = custom_data[predicted_class_index[0]]
        
        # Display the predicted class in the GUI
        result_label.config(text=f'Predicted Class: {predicted_class}')

# Create the GUI window
root = tk.Tk()
root.title('Image Classification')

# Create a button to select an image
select_button = tk.Button(root, text='Select Image', command=predict_custom_image)
select_button.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(root, text='', font=("Helvetica", 16))
result_label.pack()

# Start the GUI main loop
root.mainloop()


# VGG WITH DATA AUGMENTATION ACCURACY 46%

# In[16]:


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16


# In[17]:


conv_base = VGG16(
    weights='imagenet',
    include_top = False,
    input_shape=(150,150,3)
)


# In[18]:


model = Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[19]:


conv_base.trainable = False


# In[23]:


from keras.preprocessing.image import ImageDataGenerator


# In[24]:


from tensorflow.keras.utils import img_to_array


# In[25]:


from tensorflow.keras.utils import  array_to_img, img_to_array, load_img


# In[26]:


batch_size = 32

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/Users/sriiakhillessh/Downloads/mini project/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary') 

validation_generator = test_datagen.flow_from_directory(
        '/Users/sriiakhillessh/Downloads/mini project/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')


# In[27]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[30]:


history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator)


# PLOT

# In[31]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()


# In[ ]:




