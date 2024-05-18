import zipfile
'''

# Function to create a dictionary of pneumonia photos
def create_pneumonia_dict(zip_path):
    pneumonia_dict = {}
    
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for file_name in zip_file.namelist():
            if file_name.startswith("chest_xray/chest_xray/test/NORMAL/") and file_name.endswith(('.png', '.jpg', '.jpeg')):
                pneumonia_dict[file_name] = 0
            elif file_name.startswith("chest_xray/chest_xray/test/PNEUMONIA/") and file_name.endswith(('.png', '.jpg', '.jpeg')):
                pneumonia_dict[file_name] = 1
    
    return pneumonia_dict

# Path to the ZIP file
zip_path = "C:/Users/kasnikov/Desktop/All about coding/Tehisintellekt/archive.zip"

# Create the dictionary of pneumonia photos
pneumonia_photos_dict = create_pneumonia_dict(zip_path)

# Print the dictionary
for file_name, label in pneumonia_photos_dict.items():
    print(f"{file_name}: {label}")

'''


import zipfile
import os

# Path to the ZIP file
zip_path = "C:/Users/kasnikov/Desktop/All about coding/Tehisintellekt/archive.zip"
extract_path = "C:/Users/kasnikov/Desktop/All about coding/Tehisintellekt/extracted_photos"

# Extract the ZIP file
with zipfile.ZipFile(zip_path, "r") as zip_file:
    zip_file.extractall(extract_path)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to training and testing directories
train_dir = os.path.join(extract_path, 'chest_xray/chest_xray/train')
test_dir = os.path.join(extract_path, 'chest_xray/chest_xray/test')

# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Number of batches to yield from the generator per epoch
    epochs=1,
    batch_size = 512,
    validation_data=test_generator,
    validation_steps=50   # Number of batches to yield from the generator for validation
)



test_loss, test_acc = model.evaluate(test_generator, steps=50)
print(f"Test accuracy: {test_acc}")

import numpy as np
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale to 0-1 range
    return img_array

'''
# Path to the new image
# Path to the directory containing new images
# Initialize counters
normal_total_images = 0
normal_correct_predictions = 0

# Path to the directory containing new images
new_image_directory = "C:/Users/kasnikov/Desktop/All about coding/Tehisintellekt/extracted_photos/chest_xray/train/NORMAL"

print("-------------------------------------------------NORMAL dir -------------------------------------------------")

# Iterate through each image in the directory
for image_name in os.listdir(new_image_directory):
    image_path = os.path.join(new_image_directory, image_name)
    
    # Check if the path is a file
    if os.path.isfile(image_path):
        # Load and preprocess the image
        preprocessed_image = load_and_preprocess_image(image_path)

        # Make a prediction
        prediction = model.predict(preprocessed_image)

        # Interpret the prediction
        if prediction[0] > 0.5:
            print(f"Pneumonia detected in {image_name}")
        else:
            print(f"Normal lungs in {image_name}")
            normal_correct_predictions += 1
        
        normal_total_images += 1



# Initialize counters
pneumonia_total_images = 0
pneumonia_correct_predictions = 0

# Path to the directory containing new images
new_image_directory = "C:/Users/kasnikov/Desktop/All about coding/Tehisintellekt/extracted_photos/chest_xray/train/PNEUMONIA"

print("-------------------------------------------------PNEUMONIA dir -------------------------------------------------")

# Iterate through each image in the directory
for image_name in os.listdir(new_image_directory):
    image_path = os.path.join(new_image_directory, image_name)
    
    # Check if the path is a file
    if os.path.isfile(image_path):
        # Load and preprocess the image
        preprocessed_image = load_and_preprocess_image(image_path)

        # Make a prediction
        prediction = model.predict(preprocessed_image)

        # Interpret the prediction
        if prediction[0] > 0.5:
            print(f"Pneumonia detected in {image_name}")
            pneumonia_correct_predictions += 1
        else:
            print(f"Normal lungs in {image_name}")
        
        pneumonia_total_images += 1


# Calculate and print the accuracy
normal_accuracy = (normal_correct_predictions / normal_total_images) * 100
print(f"NORMAL Accuracy: {normal_accuracy}%")

# Calculate and print the accuracy
pneumonia_accuracy = (pneumonia_correct_predictions / pneumonia_total_images) * 100
print(f"PNEUMONIA Accuracy: {pneumonia_accuracy}%")
'''

new_image_directory = "C:/Users/kasnikov/Downloads/Necrotizing-Pneumonia-1908175079.jpg"
preprocessed_image = load_and_preprocess_image(new_image_directory)

# Make a prediction
prediction = model.predict(preprocessed_image)

# Interpret the prediction
if prediction[0] > 0.5:
      print(f"Pneumonia detected in {new_image_directory}")
else:
      print(f"Normal lungs in {new_image_directory}")