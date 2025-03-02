import random
import json
import pickle
import numpy as np
import nltk
import tkinter as tk
from tkinter import scrolledtext, PhotoImage
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
#from PIL import Image, ImageTk

# Initialize lemmatizer and load intents file, model, words, and classes
lemmatizer = WordNetLemmatizer()
intents_path = 'C:\\Users\\Asus\\Desktop\\Anuridika akka1\\Chatbot\\intents.json'
intents = json.loads(open(intents_path).read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Define functions for preprocessing, predicting and responding
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I don't understand that."

# Function to send message
def send():
    user_text = entry.get()
    if user_text.lower() == 'quit':
        root.quit()
    if user_text != '':
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, "You: " + user_text + '\n')
        entry.delete(0, tk.END)
        
        ints = predict_class(user_text)
        res = get_response(ints, intents)
        chat_area.insert(tk.END, "Bot: " + res + '\n\n')
        chat_area.config(state=tk.DISABLED)
        chat_area.yview(tk.END)

# Setup GUI with tkinter
root = tk.Tk()
root.title("Chatbot")

# Setting window size
root.geometry("400x700")
root.resizable(width=False, height=False)

# Adding a header image (optional)
header_image_path = 'C:\\Users\\Asus\\Desktop\\Anuridika akka1\\Chatbot\\download.png'  # Make sure this is a GIF image
header_img = tk.PhotoImage(file=header_image_path)
header_label = tk.Label(root, image=header_img)
header_label.pack(pady=10)

frame = tk.Frame(root)
scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

chat_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, state=tk.DISABLED, font=("Arial", 12))
chat_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=chat_area.yview)
frame.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

entry_frame = tk.Frame(root)
entry_frame.pack(padx=10, pady=10, fill=tk.X)

entry = tk.Entry(entry_frame, bd=0, bg="white", font=("Arial", 12))
entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

send_button = tk.Button(entry_frame, text="Send", command=send, font=("Arial", 12), height=10)
send_button.pack( side=tk.RIGHT)

root.bind('<Return>', lambda event: send())

root.mainloop()
