import pandas as pd
import numpy as np

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


train_data = pd.read_csv("D:/V2 Python/C117/app/Scripts/static/data_files/tweet_emotions.csv")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "content"]
    training_sentences.append(sentence)

model = load_model("D:/V2 Python/C117/app/Scripts/static/model_files/Tweet_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

emo_code_url = {
    "empty": [0, "D:/V2 Python/C117/venv/Scripts/static/emoticons/Empty.png"],
    "sadness": [1,"D:/V2 Python/C117/venv/Scripts/static/emoticons/Sadness.png" ],
    "enthusiasm": [2, "D:/V2 Python/C117/venv/Scripts/static/emoticons/Enthusiasm.png"],
    "neutral": [3, "D:/V2 Python/C117/venv/Scripts/static/emoticons/Neutral.png"],
    "worry": [4, "D:/V2 Python/C117/venv/Scripts/static/emoticons/Worry.png"],
    "surprise": [5, "D:/V2 Python/C117/venv/Scripts/static/emoticons/Surprise.png"],
    "love": [6, "D:/V2 Python/C117/venv/Scripts/static/emoticons/Love.png"],
    "fun": [7, "D:/V2 Python/C117/venv/Scripts/static/emoticons/fun.png"],
    "hate": [8, "D:/V2 Python/C117/venv/Scripts/static/emoticons/hate.png"],
    "happiness": [9, "D:/V2 Python/C117/venv/Scripts/static/emoticons/happiness.png"],
    "boredom": [10, "D:/V2 Python/C117/venv/Scripts/static/emoticons/boredom.png"],
    "relief": [11, "D:/V2 Python/C117/venv/Scripts/static/emoticons/relief.png"],
    "anger": [12, "D:/V2 Python/C117/venv/Scripts/static/emoticons/anger.png"]
    
    }

def predict(text):

    predicted_emotion=""
    predicted_emotion_img_url=""
    
    if  text!="":
        sentence = []
        sentence.append(text)

        sequences = tokenizer.texts_to_sequences(sentence)

        padded = pad_sequences(
            sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
        )
        testing_padded = np.array(padded)

        predicted_class_label = np.argmax(model.predict(testing_padded), axis=1)        
        print(predicted_class_label)   
        for key, value in emo_code_url.items():
            if value[0]==predicted_class_label:
                predicted_emotion_img_url=value[1]
                predicted_emotion=key
        return predicted_emotion, predicted_emotion_img_url