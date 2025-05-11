# MediaVerify: DeepFake Detection and Analysis Tool

A web-based system that helps users identify and analyze manipulated media using advanced AI technology.

## Overview

The DeepFake Detection System addresses the growing challenge of manipulated media content. As deepfake technology becomes more sophisticated, this tool provides reliable detection methods to preserve trust in digital content, allowing users to distinguish between authentic and synthetic media.

![image](https://github.com/user-attachments/assets/aa096c49-70f7-4493-9d48-d5a2e4c2f5d8)


## Features

- **Real-time Detection**: Process images in just 12.7ms on average
![image](https://github.com/user-attachments/assets/6b1be6e1-bf84-4676-961c-04f65654a974)

- **High Accuracy**: Over 90% accuracy across multiple datasets
![image](https://github.com/user-attachments/assets/8636799c-e832-4198-92e7-d6344a77002a)

- **Explainable AI**: LIME visualization to explain detection decisions
![image](https://github.com/user-attachments/assets/b1c66e5f-0151-4c6f-a8ed-f41f147eb9cc)

- **User Dashboard**: Track detection statistics and historical analysis
![image](https://github.com/user-attachments/assets/4a687462-9443-4bb2-b403-41dc5c155d34)

- **AI Chatbot Assistant**: Get information about deepfakes and how they work

![image](https://github.com/user-attachments/assets/522f4cc5-f249-42ff-9866-a36dbd4ea6e0)


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
