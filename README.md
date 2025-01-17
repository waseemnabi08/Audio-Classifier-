# Audio Classification Project

## Overview
This project focuses on classifying audio signals into three categories: **music**, **speech**, and **noise**. Using a deep learning model trained on MFCC (Mel-Frequency Cepstral Coefficients) features, the system achieves high accuracy and is deployed via a user-friendly web application.

## Key Features
- **Deep Learning Model**: A neural network with layers for feature extraction and classification.
- **Audio Preprocessing**: Utilizes MFCC for robust feature representation.
- **High Performance**: Achieves an accuracy of 96% on test data.
- **Web Deployment**: Provides an intuitive interface for users to upload and classify audio files.

## Dataset
- **Segments Distribution**:
  - Class 0 (Music): 101,280 segments
  - Class 1 (Speech): 144,401 segments
  - Class 2 (Noise): 21,691 segments
- **Test Set Results**:
  - Precision: 95% (Music), 98% (Speech), 89% (Noise)
  - Recall: 97% (Music), 99% (Speech), 74% (Noise)
  - F1-Score: 96% (Music), 99% (Speech), 81% (Noise)

## Neural Network Architecture
```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation='softmax')
])
```

## Results
The model achieves excellent performance metrics, making it suitable for real-world applications in audio analysis.

### Metrics Summary:
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Music     | 95%       | 97%    | 96%      | 20,129  |
| Speech    | 98%       | 99%    | 99%      | 28,973  |
| Noise     | 89%       | 74%    | 81%      | 4,373   |
| **Overall** | **96%**   | **96%**| **96%**  | 53,475  |

## Concepts and Research
1. **Audio Signals**: Analysis of pitch, frequency, and temporal structures.
2. **MFCC Features**: Extraction of mel-frequency coefficients to represent audio data effectively.
3. **Literature Review**: Examined methodologies for audio classification and pitch determination.

## Deployment
The system is deployed as a web application, allowing users to upload audio files and receive instant classification results. The app is designed for ease of use and accessibility.

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/waseemnabi08/Audio-Classifier.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Access the app in your browser at `http://localhost:5000`.


## Future Work
- Enhance noise classification accuracy.
- Extend to more audio categories.
- Integrate real-time audio processing.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any changes.



For any questions or feedback, feel free to open an issue or contact me directly.
