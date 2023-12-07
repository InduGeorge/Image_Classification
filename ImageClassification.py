import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Specify the path to your dataset
dataset_path = r'C:\Users\indup\Desktop\deep-learning\Image Classification\Dataset_Celebrities\cropped'

# Create train and validation generators
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense

model = Sequential([
    Conv2D(6, (5, 5), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(16, (5, 5), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(5, activation='softmax')  # Adjust the output size based on the number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Make predictions on the validation set
predictions = model.predict(validation_generator)

# Decode one-hot encoded predictions to class labels
predicted_classes = predictions.argmax(axis=1)

# Get the true class labels
true_classes = validation_generator.classes

# Evaluate the model on the validation set
accuracy = model.evaluate(validation_generator)[1]
print(f'Validation Accuracy: {accuracy * 100:.2f}%')



from tensorflow.keras.preprocessing import image
# Specify the path to the image you want to make predictions on
image_path = 'C:/Users/indup/Desktop/deep-learning/Image Classification/Dataset_Celebrities/cropped/roger_federer/roger_federer21.png'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale pixel values

# Make predictions
predictions = model.predict(img_array)

# Decode the one-hot encoded predictions to class labels
predicted_class = np.argmax(predictions)

# Display the predicted class
print(f'Predicted Class: {predicted_class}')

class_indices = train_generator.class_indices

# Invert the dictionary to map indices to class names
class_names = {v: k for k, v in class_indices.items()}

# Get the predicted class name
predicted_class_name = class_names[predicted_class]

# Display the predicted class name
print(f'Predicted Class: {predicted_class_name}')