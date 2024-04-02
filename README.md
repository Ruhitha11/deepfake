# Deepfake_Detection 
## What is deepfake?
- A deepfake is a manipulated video or audio where a person's appearance or voice is digitally altered to resemble someone else, often used deceptively to spread misinformation or false content. It is basically like an advanced form of Face Swapping and even Voice cloning , using an AI DeepFake Converter.
- So, Deepfake detection involves analyzing audio and video files to identify digital manipulations by utilizing advanced deep learning techniques for distinguishing genuine content from deepfakes.
## How deepfake detection works?
1) Data Collection :
   - Gather a diverse dataset of both real and fake videos to train the detection model effectively.
2) Preprocessing :
    - First it splits the video into frames next it extracts the every even frame. And then it does face detection followed by face cropping and finally frame resizing.
    - Basically it extracts individual frames from video clips, preprocesses the data by applying techniques like rescaling, augmentation and normalization to enhance the dataset's geenralization.
3) Feature extraction :
    - For video: Deep learning models might analyze facial landmarks, skin texture, blinking patterns or inconsistencies in lighting and motion.
    - For audio: Techniques might extract voice characteristics, analyze speech patterns for anomalies or identify inconsistencies between audio and lip movements.
5) Model selection :
    - Choose appropriate deep learning architectures for the analysis process. The architecture used here is EfficientNetAutoAttB4 among these architectures (EfficientNetB4 , EfficientNetB4ST , EfficientNetAutoAttB4 , EfficientNetAutoAttB4ST , Xception).
5) Model training :
    - Train a deep learning classifier model using the extracted features to distinguish between real and fake videos by splitting the dataset into training, validation and testing sets to evaluate model performance.
6) Model evaluation :
    - Evaluate the model's performance using metrics like confusion matrix , roc curve , precision recall curve , calibration curve , mel spectrogram , accuracy and also F1 score to ensure its effectiveness in detecting deepfakes.
7) Model deployment :
    - This involves integrating the trained model into a production environment where it can be used to detect deepfakes in real-world scenarios.
    - That is basically, it involves implementing the trained model into a web platform or application for users to upload and analyze audio-video content for deepfake detection.
## Model Architecture 
### For video and image : EfficientNetAutoAttB4
  - EfficientNetAutoAttB4 is a model architecture that combines EfficientNet with an Auto-Attention mechanism for enhanced deepfake detection.
  - A model combining EfficientNet with Auto-Attention for deepfake detection. It enhances feature extraction, enabling precise detection of manipulated content. Train from scratch or fine-tune from pre-trained EfficientNet. AutoAttention block focuses on vital features using self/multi-head attention mechanisms. Multi-head attention computes attention weights, emphasizing essential features. Top layers produce final output. Efficient and accurate detection of real/fake content.
### For audio : Convolutional Neural Network
  - A Convolutional Neural Network (CNN) is a type of deep learning architecture particularly well-suited for analyzing grid-like data, such  Mel spectrograms (visual representations of audio data). It excels at extracting spatial features by applying filters that slide across the input data, identifying patterns and relationships.
  - It includes layers such as
      - Input layer : Receives the audio data, typically represented as Mel spectrograms.
      - Convolutional layer : Extracting local features, like inconsistencies in voice texture, irregularities in frequency distribution, or artifacts introduced during deepfake creation , from the Mel spectrogram using convolutional filters. Each layer might have multiple filters, each focusing on specific patterns.
      - Maxpooling layer : Max pooling select the maximum value from a specific region, focusing on the most prominent features. Reduce the dimensionality of the features extracted by the convolutional layers, aiding in feature selection. Performs downsampling to reduce spatial dimensions.
      - Activation layer : It introduce non-linearity, allowing the model to learn complex relationships between features. ReLU (Rectified Linear Unit) activation, which enhances the model's ability to learn complex patterns. It also normalizes activations to stabilize training.
      - Flattening layer : Transforms the multi-dimensional output from previous layers into a single-dimensional vector. This prepares the data for the final classification stage.
      - Dense layer : That is also called as Fully Connected Layer. It processes the features extracted by the convolutional layers to make predictions, distinguishing between genuine and manipulated audio.
      - Output layer : Provides the final classification output, indicating whether the audio is authentic or a deepfake by performing the classification with a sigmoid activation function.
#### Metrics :

![image](https://github.com/Ruhitha11/deepfake/assets/162871309/5da0ece4-55b0-4518-ae61-bb7ffe60e86d)

![image](https://github.com/Ruhitha11/deepfake/assets/162871309/7f8ba948-1584-4504-ae94-d8722b93465e)

![image](https://github.com/Ruhitha11/deepfake/assets/162871309/14b0c79c-bb44-424f-b228-a2a1af325590)

![image](https://github.com/Ruhitha11/deepfake/assets/162871309/c98dc9a3-3211-4fa8-a8c0-4b69ceed093a)
## Dataset :
  - The dataset used for training is DFDC (Deepfake Detection Challenge).
  - This dataset is regarded as one of the best resources for deepfake detection research due to its size, diversity, standardization, annotations, community engagement and impact on the field.
  - It is best suited for both audio and video detections.
```bash
   la -> asvs..train -> flac
```
## Intel resources used :
  - Intel® oneAPI Deep Neural Network Library
  - Intel® Distribution for Python
  - Intel® Developer Cloud
  - Intel® oneAPI Base Toolkit
  - Intel® AI Analytics Toolkit and Libraries
  - PyTorch
  - TensorFlow
  - Scikit-learn



 
        


  
