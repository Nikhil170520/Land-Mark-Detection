import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import io  # Import io for handling in-memory files

# Load Model
model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'

# Load Labels
labels_file = "D:\\Datasets\\Land mark Detection\\landmarks_classifier_asia_V1_label_map.csv"
df = pd.read_csv(labels_file)
labels = dict(zip(df.id, df.name))

def image_processing(image):
    img_shape = (321, 321)

    # Load the model directly using KerasLayer without Sequential
    classifier = hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")

    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img  # Keep original image for display

    img = np.array(img) / 255.0
    img = img[np.newaxis]  # Expand dimensions to match model input

    # Run prediction
    result = classifier(img)  # Directly call the KerasLayer without Sequential

    return labels[np.argmax(result)], img1


def get_map(loc):
    """Get geolocation of a place"""
    geolocator = Nominatim(user_agent="landmark_recognizer")
    location = geolocator.geocode(loc)
    return location.address, location.latitude, location.longitude

def run():
    """Run the Streamlit app"""
    st.title("🌍 Landmark Recognition")

    # Display Logo
    logo = PIL.Image.open('D:\\Datasets\\Land mark Detection\\logo.png').resize((256, 256))
    st.image(logo)

    # Upload Image
    img_file = st.file_uploader("📷 Upload an Image", type=["png", "jpg", "jpeg"])

    if img_file is not None:
        # Process the image without saving
        prediction, image = image_processing(img_file)

        # Display image and prediction
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.header(f"📍 **Predicted Landmark: {prediction}**")

        try:
            address, latitude, longitude = get_map(prediction)
            st.success(f"📌 Address: {address}")

            # Display Coordinates
            loc_dict = {"Latitude": latitude, "Longitude": longitude}
            st.subheader(f"✅ **Coordinates of {prediction}**")
            st.json(loc_dict)

            # Display Map
            df = pd.DataFrame([[latitude, longitude]], columns=["lat", "lon"])
            st.subheader(f"🗺️ **{prediction} on the Map**")
            st.map(df)
        except:
            st.warning("⚠️ No address found!")

# Run the App
run()
