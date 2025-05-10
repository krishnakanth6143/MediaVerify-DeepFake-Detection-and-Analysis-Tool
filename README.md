# DeepFake Detector: Preserving Digital Media Integrity

A web-based system that helps users identify and analyze manipulated media using advanced AI technology.

## Overview

The DeepFake Detection System addresses the growing challenge of manipulated media content. As deepfake technology becomes more sophisticated, this tool provides reliable detection methods to preserve trust in digital content, allowing users to distinguish between authentic and synthetic media.

## Features

- **Real-time Detection**: Process images in just 12.7ms on average
- **High Accuracy**: Over 90% accuracy across multiple datasets
- **Explainable AI**: LIME visualization to explain detection decisions
- **User Dashboard**: Track detection statistics and historical analysis
- **AI Chatbot Assistant**: Get information about deepfakes and how they work

## Technology Stack

### AI & Machine Learning
- PyTorch for deep learning models
- ResNet-50 architecture for feature extraction
- LIME for explainable AI visualizations
- Spatial-temporal analysis techniques

### Web Development
- Flask web framework for backend
- Bootstrap 5 for responsive UI
- Chart.js for interactive visualizations
- HTML5 and modern CSS features

## Installation

1. Clone the repository
```
git clone https://github.com/YourUsername/Deep_Fake_detection.git
cd Deep_Fake_detection
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Set up environment variables (recommended for API key)
Create a `.env` file and add:
```
OPENROUTER_API_KEY=your_api_key_here
```

4. Run the application
```
python app.py
```

## Usage

1. Navigate to the Detection page
2. Upload the image you want to analyze
3. View the detection results and explanation visualization
4. Check the dashboard for historical data and statistics

## Project Structure

- `app.py` - Main Flask application
- `chatbot.py` - AI assistant implementation
- `model_train.ipynb` - Notebook with model training process
- `Model/` - Directory containing the trained detection model
- `static/` - Static assets (CSS, images, etc.)
- `templates/` - HTML templates for the web interface

## License

[MIT License](LICENSE)

## Acknowledgments

- This project was developed as part of research at Sona College of Technology
- Special thanks to the open-source community for providing tools and libraries
