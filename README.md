# PsycheSail Backend Documentation 🧠⛵

Welcome to the PsycheSail Backend repository! This repository hosts the Python server codebase for the accompanying Flutter application, PsycheSail.

## Overview 🌟

- `main.py`: This file comprises the core FastAPI implementation, orchestrating API endpoints and invoking corresponding functionalities.
- `data.json`: Here resides the dataset utilized for training the stress level detection model.
- `trainingTF`/`trainingSVM notebooks`: Jupyter Notebooks dedicated to training and storing the NLP model for stress level prediction in chats.
- `inferencing.py`: A script designed to load and predict stress levels using the saved model, providing insightful analytics.
- `VectorSearch.py`: A program engineered to facilitate text-based search using Vertex AI Vector Search, pinpointing relevant chatrooms based on shared experiences.

### Stress Detection 💆‍♂️

The Stress Detection module predicts user progress over time, furnishing users with analytics showcasing their improvement trajectory, fostering a sense of accomplishment and well-being.

### TensorFlow NLP Model 🧠🔍

Trained on a corpus of 10,000 chats, our NLP model leverages TensorFlow. Saved in the `.h5` format, it efficiently discerns stress levels in user interactions.

Here's a succinct rundown for your README file:

#### Model Architecture:
- **Embedding Layer:** Converts text data into dense vector representations.
- **SimpleRNN Layer:** Captures temporal dependencies in sequential input data.
- **Dense Layers:** Learns intricate patterns within the data.
- **Output Layer:** Generates probability distributions across stress level classes via softmax activation.

#### Training Process:
- **Data Preprocessing:** Tokenizes and pads messages for uniform input size.
- **Model Compilation:** Adam optimizer and sparse categorical cross-entropy loss function are applied.
- **Training:** The model undergoes 20 epochs of training on preprocessed data.

#### Usage:
- **Training:** Prepare a JSON dataset with message-text and corresponding stress-level fields. Customize hyperparameters and execute the training script.
- **Evaluation:** Assess model performance using metrics like accuracy.
- **Inference:** Utilize the trained model to predict stress levels for new textual inputs.

### Vector Search 🔍📊

Our Vector Search module employs text embeddings to index and cluster messages. By identifying main topics and matching user input with relevant chatroom topics, it efficiently returns matching chatroom IDs.

### FastAPI Server 🚀

The FastAPI server facilitates seamless integration between our Flutter application and the Python backend, enabling ML model processing and predictive functionalities.

## Usage 🛠️

1. Clone the repository:

```bash
git clone https://github.com/PSYCHE-SAIL/Backend
```

2. Install the requirements (Recommended Python 3.10.x):

```bash
pip install -r requirements.txt
```

3. Run the FastAPI Server:

```bash
uvicorn main:app --reload
```
