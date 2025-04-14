import numpy as np
import os
from PIL import Image


def image_generator(folder: str, target_size: tuple[int, int] = None) -> list[np.ndarray]:
    images = []
    print(f"\nProcessing folder: {folder}")

    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                # Resize image if target_size is provided
                if target_size:
                    img = img.resize(target_size)

                # Convert image to a NumPy array
                img_array = np.array(img)
                images.append(img_array)

    return images


# Generate positive and negative images if they don't exist
posExamplesPath = './ClassifiedData/positiveExamples.npy'
negExamplesPath = './ClassifiedData/negativeExamples.npy'

if not os.path.exists(posExamplesPath) or not os.path.exists(negExamplesPath):
    # Load images
    target_size = (224, 224)
    positive_images = image_generator(
        './ClassifiedData/frontfacingImages', target_size)
    negative_images = image_generator(
        './ClassifiedData/facingAwayImages', target_size)

    # Save images as numpy arrays
    np.save(posExamplesPath, positive_images)
    np.save(negExamplesPath, negative_images)
