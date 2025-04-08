from   scipy.io   import  loadmat
import numpy      as      np
import matplotlib.pylab as plt
import os
from PIL import Image


def image_generator(folder, target_size=None):
    images = []

    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                if target_size:
                    img = img.resize(target_size)
                # Convert image to a NumPy array (optionally normalize, etc.)
                img_array = np.array(img)
                images.append(img_array)

    return images

images = image_generator('./classifiedData/frontfacingImages')
images2 = image_generator('./classifiedData/facingAwayImages')

print(len(images))
print(len(images2))

# plt.imshow(images[1000])            # For grayscale images, add cmap='gray'
# plt.axis('off')                    # Optional: turn off the axis labels
# plt.title("My Image")              # Optional: add a title to the plot
# plt.show()                         # Display the image