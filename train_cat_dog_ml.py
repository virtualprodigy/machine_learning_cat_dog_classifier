import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import os
from PIL import Image
from tensorflow.keras.models import Model
from datetime import datetime
import coremltools

# Custom ImageDataGenerator subclass
class CustomImageDataGenerator(ImageDataGenerator):
    def load_img(self, path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest'):
        try:
            img = Image.open(path)
            return np.array(img.convert('RGB'))
        except (UnidentifiedImageError, PIL.UnidentifiedImageError):
            print(f"Skipped {path} due to an error.")
            return None
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
    # Save the model
    full_path = os.path.join(file_directory, file_name)
    coreml_model = coremltools.converters.convert(model)
    # Save the Core ML model with the .mlmodel extension
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

print("TensorFlow version")
print(tf.__version__)
print("--------")

# Specify GPU for Apple Metal 
print("physical device list")
print("*********")

# Check if folder paths are correct
def is_valid_folder_path(path):
    if os.path.isdir(path):
        print("\nThe folder path is valid.\n")
    else:
        print("\nThe folder path is not valid.\n")

# Training photos
path_training_photos = "./dataset/training"

# Validation photos
path_validation_photos = "./dataset/validation"

# Validation directory paths
print("\n\nVerifying Folder Paths")
is_valid_folder_path(path_training_photos)
is_valid_folder_path(path_validation_photos)

trained_model = Sequential()
trained_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
trained_model.add(MaxPooling2D((2, 2)))
trained_model.add(Conv2D(64, (3, 3), activation='relu'))
trained_model.add(MaxPooling2D((2, 2)))
trained_model.add(Conv2D(128, (3, 3), activation='relu'))
trained_model.add(MaxPooling2D((2, 2)))
trained_model.add(Conv2D(128, (3, 3), activation='relu'))
trained_model.add(MaxPooling2D((2, 2)))
trained_model.add(Flatten())
trained_model.add(Dense(512, activation='relu'))
trained_model.add(Dense(2, activation='softmax'))
trained_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = CustomImageDataGenerator(rescale=1./255)
test_datagen = CustomImageDataGenerator(rescale=1./255)
target_size = (224, 224)
batch_size = 30


# Training data generator
train_generator = train_datagen.flow_from_directory(
    directory=path_training_photos,
    class_mode='categorical',
    target_size=target_size,
    batch_size=batch_size
)

# Validation data generator
validation_generator = test_datagen.flow_from_directory(
    directory=path_validation_photos,
    class_mode='categorical',
    target_size=target_size,
    batch_size=batch_size
)


# Fit the model
history = trained_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=1,
    verbose=1
)

save_model(trained_model)



