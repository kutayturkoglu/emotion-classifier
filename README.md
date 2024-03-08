### Emotion Recognition Model

This repository contains code for training and evaluating an emotion recognition model using facial images. The model aims to classify facial expressions into six different emotions: anger, disgust, fear, happiness, sadness, and surprise.

#### Dataset

The dataset used for training and evaluation is the Facial Recognition Dataset available on Kaggle, collected by the user [apollo2506](https://www.kaggle.com/apollo2506). The dataset consists of grayscale facial images labeled with six different emotions.

#### Preprocessing

- Images are converted to grayscale and resized to 48x48 pixels.
- Data augmentation techniques such as random horizontal and vertical flips, random rotation, random resized crop, and color jittering are applied to increase the diversity of the training data.

#### Models

Two different convolutional neural network (CNN) architectures are used for training:

1. **Model X**: A custom CNN architecture (`model_x`) with multiple convolutional layers followed by batch normalization, ReLU activation, max-pooling, and dropout layers.
2. **Model Y**: Another custom CNN architecture (`model_y`) with a simpler structure consisting of fewer convolutional layers.

#### Training

The models are trained using the training dataset with a softmax cross-entropy loss function and stochastic gradient descent (SGD) optimizer. Learning rate reduction on plateau is applied during training to adjust the learning rate dynamically.

#### Evaluation

- The accuracy of each model is evaluated using the test dataset.
- An ensemble of the trained models is created, and the final prediction is made by taking a weighted average of the individual model predictions.

#### Results

The performance of the models is as follows:

- Model X achieves approximately 50% accuracy on the test set.
- The main challenge is correctly classifying the "neutral" emotion (class 3).

#### Dependencies

To run the code, the following dependencies are required:

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- tqdm
- PIL (Python Imaging Library)

#### Usage

To train the models and evaluate their performance, follow the steps outlined in the code files provided in this repository.

```bash
# Clone the repository
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition

# Install dependencies
pip install -r requirements.txt

# Run the training and evaluation scripts
python train.py
python evaluate.py
