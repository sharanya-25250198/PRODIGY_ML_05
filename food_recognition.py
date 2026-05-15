import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# Load Model
model = MobileNetV2(weights="imagenet")

# Calories Dictionary
food_calories = {
    "pizza": 285,
    "banana": 105,
    "cheeseburger": 303,
    "sandwich": 250,
    "hotdog": 290,
    "apple": 95
}

# Foods folder path
image_folder = "foods"

# Read all images from folder
for img_name in os.listdir(image_folder):

    # Full image path
    img_path = os.path.join(image_folder, img_name)

    print("\n========================")
    print("Checking:", img_name)

    # Load Image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess image
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)

    # Decode predictions
    decoded = decode_predictions(predictions, top=3)[0]

    print("\nTop Predictions:")

    for _, label, confidence in decoded:
        print(f"{label}: {confidence*100:.2f}%")

    # Best Prediction
    best_label = decoded[0][1]

    print("\nDetected Food:", best_label)

    # Calories
    if best_label in food_calories:
        print("Estimated Calories:", food_calories[best_label], "kcal")
    else:
        print("Calories data not available")

print("\nAll food images checked successfully!")