# Foreign-Sketch-Generator

## Overview
This project uses Generative Adversarial Networks (GANs) and NLP models to implement a forensic sketch generator. The application allows users to describe a suspect, and it generates a sketch based on the given description. It utilizes:
- **GAN Networks** for realistic sketch generation
- **CLIP Model** for text-image similarity matching
- **DeepFace** for facial analysis
- **Flask** for web-based interaction
- **Sentence Transformers** for text embeddings

The GAN model has been trained on a small dataset of 10 persons' faces from the CUFS dataset.

## Features
- Generate forensic sketches from text descriptions using GANs
- Extract forensic details from images using DeepFace
- Match descriptions to existing forensic sketches using CLIP
- Fully functional website for user interaction

## Project Structure
```
|-- app.py                 # Flask backend for sketch generation
|-- index.html             # Frontend UI
|-- TEXT EMbeddings.py     # Text embedding processing with Sentence Transformers
|-- GAN MODEL TRAINING.ipynb # Jupyter notebook for training GAN networks
```

## Installation
### Prerequisites
- Python 3.7+
- pip installed
- Virtual environment (recommended)

### Steps
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd <repo_name>
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask server:
   ```sh
   python app.py
   ```
2. Open the web application in your browser:
   ```
   http://127.0.0.1:5000/
   ```
3. Enter a description of the suspect and click 'Generate Sketch'.

## Models Used
GANs (Generative Adversarial Networks) are used for realistic forensic sketch generation. They are trained on a small dataset of 10 persons' faces from the CUFS dataset.
- **CLIP (Contrastive Language-Image Pretraining)**: Used for text-to-image similarity matching.
- **DeepFace**: Performs face recognition and analysis (age, gender, emotion, race).
- **Sentence Transformers**: Converts text descriptions into embeddings.

## Future Enhancements
- Improve GAN architecture for higher-quality sketches.
- Integrate real-time image processing and enhancement.
- Optimize model inference speed for better performance.

## Contributors
- **Kesava Datta**
- **Vikram**
- **Shashank**
- **Meghashyam**
- **tanveen**
