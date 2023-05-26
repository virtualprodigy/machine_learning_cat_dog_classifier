import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from datetime import datetime
import coremltools
import os

def save_h5_model(model: Model, file_directory, file_name: str):
    # Save the model
    full_path = os.path.join(file_directory, file_name)
    model.save(full_path)
    print("H5 Model(Legacy) saved to:", full_path)

def save_keras_model(model: Model, file_directory, file_name: str):
    # Save the model
    full_path = os.path.join(file_directory, file_name)
    model.save(full_path)
    print("Keras Model saved to:", full_path)

def save_ios_mlmodel(model: Model, file_directory, file_name: str):
    # add input for images when model is used on iOS
    input_shape = (224, 224, 3)  # Adjust dimensions according to your image size and channels
    input_layer = Input(shape=input_shape, dtype='float32', name='input')

    #creates model that supports inputs
    combined_model = Model(inputs=input_layer, outputs=model(input_layer))
    coreml_model = coremltools.converters.convert(
        combined_model,
        inputs=[coremltools.ImageType()]
    )

    # Save the Core ML model with the .mlmodel extension
    full_path = os.path.join(file_directory, file_name)
    coreml_model.save(full_path)
    print("MLModel(iOS) saved to:", full_path)

def save_tflite_mlmodel(model: Model, file_directory, file_name: str):
    # Convert the keras model
    full_path = os.path.join(file_directory, file_name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # returns a byte array
    tflite_model = converter.convert()
    # Save the TFLite Model
    with open(full_path, "wb") as tflite_file:
        tflite_file.write(tflite_model)
    print("TFLite Model saved to:", full_path)

def save_model(model):
    # Check if the directory exists, create it if not
    output_directory = "./model_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print("Created directory:", output_directory)

    # create a sub directory using the current date and time for versioning
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%I_%M_%p_%m_%d_%Y")
    sub_directory = os.path.join(output_directory, formatted_datetime)
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
        print("Created sub directory:", sub_directory)

    save_h5_model(model, sub_directory, "cat_dog_classifier_legacy.h5")
    save_keras_model(model, sub_directory, "cat_dog_classifier.keras")
    save_ios_mlmodel(model, sub_directory, "cat_dog_classifier_ios.mlmodel")
    save_tflite_mlmodel(model, sub_directory, "cat_dog_classifier_android.tflite")