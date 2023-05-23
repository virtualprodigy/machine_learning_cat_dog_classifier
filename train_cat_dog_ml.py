import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import os
from PIL import Image
from tensorflow.keras.models import Model
from datetime import datetime

# Custom ImageDataGenerator subclass
class CustomImageDataGenerator(ImageDataGenerator):
    def load_img(self, path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest'):
        try:
            img = Image.open(path)
            return np.array(img.convert('RGB'))
        except (UnidentifiedImageError, PIL.UnidentifiedImageError):
            print(f"Skipped {path} due to an error.")
            return None


def save_model_with_directory_check(model: Model, file_path: str):
    # Check if the directory exists, create it if not
    directory = "./model_output"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created directory:", directory)

    # create a sub directory using the current date and time for versioning
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%I_%M_%p_%m_%d_%Y")
    sub_directory = os.path.join(directory, formatted_datetime)
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
        print("Created directory:", sub_directory)
    
    # Save the model
    full_path = os.path.join(sub_directory, file_path)
    model.save(full_path)
    print("Model saved to:", full_path)

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

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=200,
    verbose=1
)

save_model_with_directory_check(model, "cat_dog_classifier.keras")
save_model_with_directory_check(model, "cat_dog_classifier_legacy.h5")
save_model_with_directory_check(model, "cat_dog_classifier_android.tflite")
save_model_with_directory_check(model, "cat_dog_classifier_ios.mlmodel'")
save_model_with_directory_check(model, "cat_dog_classifier_onnx_os.onnx")

