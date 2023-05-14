# HateSpeechAlert

In today's digital age, social media platforms like Twitter, Facebook, Instagram, and YouTube have become breeding grounds for hate speech. It has caused many instances of hate and division among users, communities etc. Our aim is to make these online spaces safer and more inclusive for everyone.  

Our project takes a text (which may be comment from social platform or any other user feedback), it analyse and  provide a response indicating whether the text contains Hate Speech, Offensive Speech, or neither of them.

## Installation

1. Install the library   
   `pip install -r requirements.txt`

2. Train the model  
   `python ./hatespeechdetector.py`

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [CommandLine](#commandline)
  - [Embedding it](#embedding-it)
- [RoadMap](#roadmap)

## Usage

### CommandLine

`python ./predict.py "I am happy"`  -> Output `No Hate or Offensive Speech Detected`  
`python ./predict.py "I will kill you"`  -> Output `Hate Speach Detected`  

### Embedding it 

```python
from predict import predict
predict("I am happy")
```

OR

Directly Use Our Model
```python
import joblib
cv = joblib.load('vectorizer.pkl')
model = joblib.load('classifier.pkl')
result = cv.transform([text]).toarray()
return model.predict(result)
```

**Note:** The output will be formated as:

| Label | Category           |
|-------|--------------------|
| 0     | Hate Speech        |
| 1     | Offensive Speech   |
| 2     | Neither of Them    |

## RoadMap

1. Create an http API so platform can easily integrate our model in their service.
