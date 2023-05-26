from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import os
from PIL import Image
from model_saver import save_model
import subprocess

#display version details
subprocess.run(["python", "version_details.py"])

# Custom ImageDataGenerator subclass
class CustomImageDataGenerator(ImageDataGenerator):
    def load_img(self, path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest'):
        try:
            img = Image.open(path)
            return np.array(img.convert('RGB'))
        except (UnidentifiedImageError, PIL.UnidentifiedImageError):
            print(f"Skipped {path} due to an error.")
            return None

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
    epochs=200,
    verbose=1
)

save_model(trained_model)



