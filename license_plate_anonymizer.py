import streamlit as st
import torch
from PIL import Image, ImageFilter
import io
from ultralytics import YOLO

@st.cache_data
def get_detections(image_file: str, _model: YOLO, device="cpu"):
    """
    Get the detections (bounding boxes) from the YOLO model.

    Args:
        image_file (str): Path to the input image file.
        model (ultralytics.YOLO): YOLO model.
        device (str): Device to run the model on.

    Returns:
        torch.Tensor: Tensor containing bounding box coordinates, confidences and labels [x1, y1, x2, y2, conf, label]
    """
    image = Image.open(image_file)
    # Perform inference
    results = _model.predict(image, imgsz=1280, device=device, verbose=False)
    detections = results[0].boxes.data
    return detections

def anonymize_image(image: Image.Image, detections: torch.Tensor, blur_radius: float):
    """
    Anonymize the license plate in the input image by blurring the regions defined by bounding boxes.

    Args:
        image (PIL.Image.Image): Input image.
        detections (torch.Tensor): Tensor containing bounding box coordinates and labels.
        blur_radius (int): Radius of the Gaussian blur filter.

    Returns:
        PIL.Image.Image: Anonymized image.
    """
    for bbox in detections:
        x1, y1, x2, y2 = map(int, bbox[:4])
        region = image.crop((x1, y1, x2, y2))
        blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image.paste(blurred_region, (x1, y1, x2, y2))
    return image

@st.cache_resource
def load_model():
    """
    Load the YOLO model from the checkpoint.

    Returns:
        ultralytics.YOLO: YOLO model.
    """
    model_path = "data/models/v8n_lp_v1.pt"
    model = YOLO(model_path)
    return model

def main():
    st.title("License Plate Anonymizer")
    st.write("Easily anonymize license plates in images by uploading them and adjusting the blur strength slider. Anonymized images can be downloaded via the 'Download Anonymized Image' Button.")
    st.markdown(
        """
        <span style='font-size: 16px; padding-right: 10px; vertical-align: middle;'>Made by Lars Ippen</span>
        [<img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" width=30 height=30 style='background-color: white; padding: 3px; border-radius: 3px;'>](https://www.linkedin.com/in/lars-ippen)
        <span style='padding-right: 4px;'></span>
        [<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width=30 height=30 style='background-color: white; padding: 3px; border-radius: 3px;'>](https://github.com/ippen)
        """, unsafe_allow_html=True
    )
    
    # Display a file uploader widget
    uploaded_file = st.file_uploader("Choose a file", type=["jpeg", "jpg", "png", "webp"])

    blur_strength = st.slider("Blur Strength", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.1f", help="Set the strength of the blur effect. 0: no blur, 100: maximum blur.")
    #blur_strength = st.slider("Blur Strength", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    # Check if the session state has 'detections', if not initialize it
    if 'detections' not in st.session_state:
        st.session_state.detections = None

    if uploaded_file is not None:
        # Open the uploaded image

        # Load YOLO model
        yolo_model = load_model()

        # Always recalculate detections for a new uploaded image
        with st.spinner('Detecting license plates...'):
            # Get detections
            st.session_state.detections = get_detections(uploaded_file, yolo_model)

        with st.spinner('Anonymizing...'):
            # Anonymize license plates and dynamically set blur strength
            image = Image.open(uploaded_file)
            anonymized_image = anonymize_image(image, st.session_state.detections, blur_strength)

        # Display the anonymized image
        st.image(anonymized_image, caption="Anonymized Image", use_column_width=True)

        # Download button for the anonymized image
        buffered = io.BytesIO()
        anonymized_image.save(buffered, format="PNG")
        anonymized_image_data = buffered.getvalue()
        uploaded_file_name = uploaded_file.name.split(".")[0]
        st.download_button(
            label="Download Anonymized Image",
            data=anonymized_image_data,
            file_name=uploaded_file_name+"_anonymized.png",
            mime="image/png",
        )

if __name__ == "__main__":
    main()
