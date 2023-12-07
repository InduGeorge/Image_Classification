# Image_Classification

## Model Architecture
The implemented image classification model follows the LeNet-5 architecture, a classic convolutional neural network (CNN) designed for handwritten and machine-printed character recognition. The model is structured as follows:

Convolutional Layers:

Convolutional Layer 1: 6 filters, each of size (5, 5), with ReLU activation.
MaxPooling Layer 1: Pooling size of (2, 2).
Convolutional Layer 2: 16 filters, each of size (5, 5), with ReLU activation.
MaxPooling Layer 2: Pooling size of (2, 2).
Flatten Layer:

Flattens the output from the previous layers into a 1D array.
Fully Connected Layers:

Dense Layer 1: 120 neurons with ReLU activation.
Dense Layer 2: 84 neurons with ReLU activation.
Output Layer: Dense layer with 5 neurons (adjust based on the number of classes) and softmax activation.
Model Compilation
The model is compiled using the following configurations:

Optimizer: Adam optimizer.
Loss Function: Categorical Crossentropy.
Metrics: Accuracy.
Training
The model is trained using the specified data generators (train_generator and validation_generator). The training is performed for 10 epochs, with the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric.

### Prediction
After training, the model makes predictions on the validation set. The one-hot encoded predictions are decoded to class labels, and the accuracy is printed.

Sample Image Prediction
A sample image (roger_federer21.png) is provided for prediction. The image is loaded, preprocessed, and fed into the trained model. The predicted class index is decoded to a class name using a mapping from class indices to class names.

Model Evaluation
The model's performance is evaluated on the validation set, and the final accuracy is displayed.

Additional Notes
The model is built using TensorFlow and Keras.
The LeNet-5 architecture is chosen for its simplicity and effectiveness in image classification tasks.
Adjustments may be required based on the specific dataset and the number of classes in your application.
Feel free to experiment with different architectures, hyperparameters, or datasets to further improve the model's accuracy and robustness. The provided code serves as a starting point for image classification tasks using the LeNet-5 architecture.





