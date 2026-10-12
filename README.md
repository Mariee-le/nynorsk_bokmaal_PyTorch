The dataset contains Norwegian text labelled by language variety:

nob – Bokmål

nno – Nynorsk

The dataset is split into training and test data.

Model architecture

The model is a simple neural network classifier:

Input text
    ↓
Word tokenization
    ↓
Vocabulary lookup
    ↓
Embedding layer (128 dimensions)
    ↓
Mean pooling
    ↓
Linear layer
    ↓
ReLU activation
    ↓
Linear classifier
    ↓
Bokmål / Nynorsk prediction

Technologies used

Python

PyTorch

Hugging Face Datasets

Neural networks

Natural Language Processing (NLP)

Installation

Clone the repository:

git clone https://github.com/Mariee-le/nynorsk-bokmaal-PyTorch.git

Navigate to the project folder:

cd nynorsk-bokmaal-PyTorch

Install dependencies:

pip install torch datasets

Training the model

Run:

python file.py

The model will:

load the dataset

create the vocabulary

train for 5 epochs

evaluate accuracy on the test set

run example predictions

Results

The model achieved approximately:

Accuracy: ~95.5%

on the test set.

Example predictions:

"Kva synast du om å lære spansk?"
→ nynorsk

"hva synst du om å lære spansk"
→ bokmål

Example usage

After training, the model can classify new Norwegian text:

predict("Eg likar å lese bøker")

Output:

nynorsk

Future improvements

Possible extensions:

Use character-level models for better handling of spelling variation

Add more Norwegian language varieties

Experiment with LSTM or Transformer-based models

Use Hugging Face tokenizers

Create a separate inference script

Compare different model architectures

About

This project was created as part of my exploration of NLP and machine learning, combining computational methods with my background in linguistics.
