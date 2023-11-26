import os
import numpy as np
from PIL import Image
import csv
from datakeywords import *

current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Specify the file paths
image_filepath = 'train-images-idx3-ubyte'#'t10k-images-idx3-ubyte'
output_images_directory = image_filepath+'_dir' # Specify the directory where you want to save the images
labels_filepath = 'train-labels-idx1-ubyte'#'t10k-labels-idx1-ubyte'#
# Write data to a CSV file
output_csv_path = labels_filepath+'.csv'

data_path=os.path.join(current_directory,"data")
image_filepath = os.path.join(data_path,image_filepath)
output_images_directory = os.path.join(data_path,output_images_directory)
labels_filepath = os.path.join(data_path,labels_filepath)
output_csv_path = os.path.join(data_path,output_csv_path)

# Create the output directory if it doesn't exist
if not os.path.exists(output_images_directory):
    os.makedirs(output_images_directory)

# Read the binary data
with open(image_filepath, 'rb') as f:
    magic_number = int.from_bytes(f.read(4), 'big')
    num_images = int.from_bytes(f.read(4), 'big')
    num_rows = int.from_bytes(f.read(4), 'big')
    num_cols = int.from_bytes(f.read(4), 'big')
    images_data = f.read()

# Convert binary data to numpy array
images = np.frombuffer(images_data, dtype=np.uint8)
images = images.reshape(num_images, num_rows, num_cols)

# Save images as image files
for idx, image in enumerate(images):
    image_filename = os.path.join(output_images_directory, f'image_{idx}.png')
    image_pil = Image.fromarray(image)
    image_pil.save(image_filename)

print(f'Saved {num_images} images in {output_images_directory}')

# Now for the labels extraction
# Read the labels binary data
with open(labels_filepath, 'rb') as f:
    magic_number = int.from_bytes(f.read(4), 'big')
    num_labels = int.from_bytes(f.read(4), 'big')
    labels_data = f.read()

# Convert binary data to numpy array
labels = np.frombuffer(labels_data, dtype=np.uint8)

# Create a list of image file paths
image_paths = [os.path.join(output_images_directory, f'image_{i}.png') for i in range(num_labels)]

# Combine image paths and labels into a list of tuples
data = list(zip(image_paths, labels))

with open(output_csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([PATHKEY, LABELKEY])  # Write header
    csv_writer.writerows(data)

print(f'Saved {num_labels} image paths and labels to {output_csv_path}')

