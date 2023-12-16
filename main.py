import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from PIL import Image, ImageFont
from PIL import ImageDraw
import io
import matplotlib.pyplot as plt

st.title("Car Object Detection")

###
def preprocess_image(image_bytes):
  """
  Preprocesses an uploaded image for vehicle detection.

  Args:
      image_bytes: Bytes of the uploaded image.

  Returns:
      A preprocessed NumPy array representing the image.
  """

  # Decode the image from bytes
  image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

  # Resize the image to the desired dimensions
  resized_image = cv2.resize(image_array, dsize=(676, 380))

  # Normalize the image 
  normalized_image = resized_image / 255.0

  print("nor", normalized_image.shape)

  # Expand the image dimension for model prediction
  # This may need modification based on your model input requirements
  expanded_image = np.expand_dims(normalized_image, axis=0)

  print("exp:", expanded_image.shape)

  return expanded_image

###
def predict_boxes(image):
  """
  Predicts bounding boxes for vehicles in a preprocessed image.

  Args:
      image: A preprocessed NumPy array representing the image.

  Returns:
      A list of predicted bounding boxes. Each box is a tuple of (xmin, ymin, xmax, ymax) coordinates.
  """

  # Load the trained model
  model = load_model('car-object-detection.h5')

  # Make the prediction
  predicted_boxes = model.predict(image)

  adjusted_boxes = []
  for box in predicted_boxes:
    xmin, ymin, xmax, ymax = box

    print("on predict : ", xmin, " + ", ymin, " + ", xmax, " + ", ymax)
    # Convert to integers for drawing on the image
    adjusted_boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))

  return adjusted_boxes

st.markdown("A simple and accurate **CV Project**.")
text = "Check out this project on [Github](https://github.com/)."
st.markdown(text)

st.subheader("Disclaimer")
st.warning('This demo is only dedicated for single car object detection.', icon="⚠️")

### Upload the image
uploaded_file = st.file_uploader("Choose an image for detection", type=['jpg'])
print(" uploaded file: ", uploaded_file)

if uploaded_file is not None:
    # Read image bytes
    image_bytes = uploaded_file.read()

    with st.spinner("Analyzing your image..."):
        # Preprocess image
        image = preprocess_image(image_bytes)

        # Predict bounding boxes
        predicted_boxes = predict_boxes(image)

        
    
        for box in predicted_boxes:
            # draw_bounding_box(annotated_image, box)
            xmin, ymin, xmax,ymax = box
            
            
    
    # Display original image
    # st.image(image, caption="Original Image")

     # Draw bounding boxes on a copy of the image
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.asarray(image)

    # Define the bounding box color and thickness
    color = (255, 0, 0)  
    thickness = 4
    
    rectangle_image = cv2.rectangle(image_array, (xmin, ymin), (xmax, ymax), color, thickness)
    rectangle_image = cv2.resize(image_array, dsize=(676, 380))

    # Add labels or text within the box
    text = "Car"  
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rectangle_image, text, (xmin, ymin - 5), font, 1, color, thickness)
    # st.image(rectangle_image, caption="Image with Bounding Boxes")
    print("dis", rectangle_image.shape)

    # Define heatmap data 
    heatmap_data = np.random.rand(rectangle_image.shape[0], rectangle_image.shape[1])

    # Define colormap
    cmap = plt.cm.viridis

    # Apply colormap to data
    heatmap_image = cv2.applyColorMap(heatmap_data.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # Blend heatmap with image
    blended_image = cv2.addWeighted(rectangle_image, 0.7, heatmap_image, 0.3, 0)

    # Display the result
    # st.image(blended_image, caption="Image with Heatmap, Bounding Boxes and Text")

    # Display images side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image")

    with col2:
        st.image(rectangle_image, caption="Image with Bounding Boxes")

    with col3:
        st.image(blended_image, caption="Image with Heatmap, Bounding Boxes and Text")






  

    



