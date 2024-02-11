# License Plate Anonymizer

This application is designed to anonymize license plates in images. It utilizes a YOLO (You Only Look Once) object detection model to detect license plates in images and applies a Gaussian blur to anonymize them.

## How it Works

1. **Upload Image**: Users can upload an image containing license plates.
2. **Detect License Plates**: The application detects license plates in the uploaded image using the YOLO object detection model.
3. **Anonymize**: Detected license plates are anonymized by applying a Gaussian blur effect to the corresponding regions in the image.
4. **Download**: Users can download the anonymized image with the blurred license plates.

## Usage


### Website

The application is publicly available at [License Plate Anonymizer](https://license-plate-anonymizer.streamlit.app/). You can visit the website to blur license plates in images.

### Running Locally

To run the application locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/license-plate-anonymizer.git
```

2. Navigate to the project directory:

```bash
cd license-plate-anonymizer
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run license_plate_anonymizer.py
```

5. Access the application in your web browser at `http://localhost:8501`.

## Components

### `license_plate_anonymizer.py`

This Python script contains the main functionality of the application. It includes functions for detecting license plates in images, anonymizing the detected license plates, loading the YOLO model, and the main Streamlit application.

### `requirements.txt`

This file lists all the Python libraries and their versions required to run the application. You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Models

The YOLO model used for license plate detection is stored in `data/models/v8n_lp_v1.pt`.
