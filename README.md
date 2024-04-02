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
4) Model selection :
  - Choose appropriate deep learning architectures for the analysis process. The architecture used here is EfficientNetAutoAttB4 among these architectures (EfficientNetB4 , EfficientNetB4ST , EfficientNetAutoAttB4 , EfficientNetAutoAttB4ST , Xception).
5) Model training :
  - Train a deep learning classifier model using the extracted features to distinguish between real and fake videos by splitting the dataset into training, validation and testing sets to evaluate model performance.
6) Model evaluation :
  - Evaluate the model's performance using metrics like confusion matrix , roc curve , precision recall curve , calibration curve and by mel spectrogram to ensure its effectiveness in detecting deepfakes.
7) Model deployment :
  - This involves integrating the trained model into a production environment where it can be used to detect deepfakes in real-world scenarios.
  - That is basically, it involves implementing the trained model into a web platform or application for users to upload and analyze audio-video content for deepfake detection.
## Model Architecture 
### For video : EfficientNetAutoAttB4
  - It is a variant of the standard EfficientNetB4 architecture trained with a classical end-to-end approach with the addition of an attention mechanism that allows the network to concentrate on the most important areas of the feature maps.
  EfficientNetAutoAttB4 is a model architecture that combines EfficientNet with an Auto-Attention mechanism for enhanced deepfake detection.
  The model focuses on relevant features using EfficientNet backbone with Auto-Attention, enabling precise predictions in deepfake detection tasks across different modalities.
  EfficientNetAutoAttB4 is a specific variant of the EfficientNet architecture that includes an additional AutoAttention block. This block is a type of attention mechanism that is applied to the features maps produced by the EfficientNet backbone. The AutoAttention block is designed to selectively focus on the most important features in the image, improving the model's performance on downstream tasks.

  
