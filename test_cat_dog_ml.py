from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.optimizers import RMSprop

# Load the saved model
model = load_model('./model_output/cat_dog_classifier_legacy.h5')

# Iterate over the images in the directory
test_dir = './test_dataset'
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)

    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        # Convert the image to a numpy array and normalize it
        test_image_array = image.img_to_array(img)
        test_image_array = np.expand_dims(test_image_array, axis=0)
        test_image_array /= 255.

        # Predict the class of the image
        prediction = model.predict(test_image_array)

        # Print the predicted class
        print(f"\nPrediction for {img_file}: {prediction}")

        if prediction[0][0] > 0.5:
            print(f"CAT - {prediction[0][0]:.4f}")

        if prediction[0][1] > 0.5:
            print(f"DOG- {prediction[0][1]:.4f}")

    except (UnidentifiedImageError, FileNotFoundError) as e:
        print(f"Skipped {img_file} due to an error: {str(e)}")