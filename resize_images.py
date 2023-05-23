import os
from PIL import Image, UnidentifiedImageError

def resize_images(photo_paths_json):
    """
    Resize images from the original directory and save them in the resized directory.

    Args:
        original_dir (str): Path to the directory containing the original images.
        resized_dir (str): Path to the directory where resized images will be saved.
        target_size (tuple): Desired target size for the resized images, e.g., (224, 224).
    """
    original_dir = photo_paths_json['original_dir']
    resized_dir = photo_paths_json['resized_dir']
    target_size = photo_paths_json['target_size']
    os.makedirs(resized_dir, exist_ok=True)

    # Iterate over the files in the original directory

    # Get the list of files in the original directory and sort them
    ordered_files = sorted(os.listdir(original_dir))

    for filename in ordered_files:
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(original_dir, filename)
            try:
                image = Image.open(image_path)

                # Convert image to RGB mode if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                resized_image = image.resize(target_size)

                # Save the resized image in the resized directory
                resized_image_path = os.path.join(resized_dir, filename)
                resized_image.save(resized_image_path)

                print(f"Resized image {filename} saved to directory {resized_dir}.")
            except (OSError, UnidentifiedImageError) as e:
                print(f"Failed to process image {resized_dir} -> {filename}: {str(e)}")

# Example usage
cat_training = {
    'original_dir': './dataset/preprocessed_images/training/cat',
    'resized_dir': './dataset/training/cat',
    'target_size': (224, 224)
}

cat_validation = {
    'original_dir': './dataset/preprocessed_images/validation/cat',
    'resized_dir': './dataset/validation/cat',
    'target_size': (224, 224)
}

dog_training = {
    'original_dir': './dataset/preprocessed_images/training/dog',
    'resized_dir': './dataset/training/dog',
    'target_size': (224, 224)
}

dog_validation = {
    'original_dir': './dataset/preprocessed_images/validation/dog',
    'resized_dir': './dataset/validation/dog',
    'target_size': (224, 224)
}

# training cats
resize_images(cat_training)
# validation cat
resize_images(cat_validation)

# training dogs
resize_images(dog_training)
# validation dogs
resize_images(dog_validation)
