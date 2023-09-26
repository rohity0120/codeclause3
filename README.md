import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load a pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([
    hub.KerasLayer(model_url, output_shape=[1001])
])

# Load a sample retinal image for testing (replace with your image)
image_path = "sample_retina.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image = cv2.resize(image, (224, 224))  # Resize to match the model's input size

# Preprocess the image
image = image / 255.0  # Normalize pixel values to [0, 1]
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(image)

# Decode predictions (you need a mapping of class indices to condition names)
class_names = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Macular Degeneration", "Other"]
predicted_class = np.argmax(predictions)

# Print the predicted condition
print("Predicted Condition: " + class_names[predicted_class])
