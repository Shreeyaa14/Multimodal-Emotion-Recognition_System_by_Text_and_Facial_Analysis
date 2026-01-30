**Multimodal Emotion Recognition System by Text and Facial Analysis**

This project is an automated Multimodal Emotion Recognition System that identifies the 7 emotions of humans, namely Happy, Sad, Fear, Anger, Disgust, Surprise, Neutral, by jointly analyzing textual input and facial expressions.
The system leverages Natural Language Processing and Computer Vision techniques to extract emotional cues from multiple modalities, improving prediction accuracy compared to single-source emotion detection. It supports real-time facial emotion recognition and text-based sentiment inference through an interactive web interface.

**System Workflow**
* Accept text input and facial image or live webcam stream
* Preprocess text using NLP cleaning and tokenization
* Detect face regions and normalize facial images
* Extract semantic features from text and spatial features from facial data
* Predict emotions independently using trained deep learning models
* Fuse text and facial emotion predictions using confidence-based aggregation
* Display final emotion label and confidence score in real time

**Technical Details**
* **Programming Language:** Python
* **Text Emotion Analysis:** NLP preprocessing, tokenization, embeddings, LSTM/DNN-based classifier
* **Facial Emotion Recognition:** Convolutional Neural Network (CNN) for facial feature extraction
* **Face Detection:** OpenCV-based detection and preprocessing
* **Multimodal Fusion:** Score-level fusion combining text and facial predictions
* **Real-Time Processing:** Webcam-based facial emotion detection
* **Web Framework:** Streamlit
* **Data Handling:** CSV-based emotion datasets
* **Libraries Used:** TensorFlow/Keras, OpenCV, Pandas, NumPy, NLTK

**Output**
* Predicted emotion labels (Happy, Sad, Angry, Neutral, Fear, Surprise, Disgust)
* Confidence scores for each detected emotion
* Real-time emotion visualization through the web interface
* Input text, facial images, and datasets are included in the project files
