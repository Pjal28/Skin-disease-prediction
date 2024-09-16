import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_skin_model.keras')

# Dictionary mapping class indices to class names
classes = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'nv', 5: 'vasc', 6: 'mel'}

def predict_skin_disease(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image to match model input shape (28x28)
    img = cv2.resize(img, (28, 28))
    
    # Normalize pixel values
    img = img / 255.0
    
    # Make prediction
    result = model.predict(np.expand_dims(img, axis=0))
    
    # Get the predicted class index
    class_index = np.argmax(result)
    
    # Get the predicted class name
    class_name = classes[class_index]
    
    return class_name

# Example usage:
image_path = 'example_image.jpg'  # Replace 'example_image.jpg' with the path to your image
predicted_class = predict_skin_disease(image_path)
print("Predicted skin disease:", predicted_class)
