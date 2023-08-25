from flask import *
import pickle    
import io
import praw
import json
from datetime import datetime
from googleapiclient.discovery import build
#!pip install torchtext==0.6.0 --quiet
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
import numpy as np
import pandas as pd
import spacy
import random
from torchtext.data import TabularDataset
import io
import speech_recognition as sr
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import pickle
import io
import speech_recognition as sr
global sp_src_path,sp_trg_path,english,device
from flask_cors import CORS
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import pandas as pd
import tensorflow as tf
import clip
from PIL import Image
from Churn.churn import process_file

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
reddit = praw.Reddit(client_id='LchMksVUmRUeyg', client_secret='gb1XyXX-r0ycV9KKFM-ujFVNOogO_w', user_agent='Data Scraping')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = TFBertForSequenceClassification.from_pretrained('tf_model')
labels = ['Negative', 'Positive']  # (0:negative, 1:positive)

negative_sentiment_words = []
with open('data/neg_words.txt', 'r') as file:
        for line in file:
            line = line.strip().lower()
            negative_sentiment_words.append(line)
positive_sentiment_words = []
with open('data/pos_words.txt', 'r') as file:
        for line in file:
            line = line.strip().lower()
            positive_sentiment_words.append(line)


# Load the CLIP model
clipmodel, preprocess = clip.load("ViT-B/32", "cpu")

def analyze_meme_sentiment(image_path):
    # Load the meme image
    image = Image.open(image_path)

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0)

    # Perform sentiment analysis using the CLIP model
    with torch.no_grad():
        image_features = clipmodel.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text = ["This meme is positive", "This meme is negative"]
        text_inputs = torch.cat([clip.tokenize(text) for _ in range(image_features.shape[0])]).to("cpu")
        text_features = clipmodel.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Get the sentiment label
    sentiment_label = "Positive" if similarity[0][0] > similarity[0][1] else "Negative"

    return sentiment_label

@app.route('/meme', methods=['POST'])
def analyze_meme():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:

        # Analyze the sentiment of the meme
        sentiment = analyze_meme_sentiment(file)

        return jsonify({'sentiment': sentiment})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500



@app.route('/audio', methods=['POST'])
def predict_audio():
    file = request.files['file']
    src_lang = request.form['language']
    print("src " + src_lang)
    print("trg " + trg_lang)
    if src_lang == "English" and trg_lang == "Tamil":
        in_language = 'en-in'
        sp_src_path = "en"
        sp_trg_path = "ta"
        out_language = "ta"
        folder = "data/5Lang/English to Tamil"
    elif src_lang == "Tamil" and trg_lang == "English":
        in_language = 'ta-in'
        sp_src_path = "ta"
        sp_trg_path = "en"
        out_language = "en"
        folder = "data/5Lang/Tamil to English"
    elif src_lang == "Spanish" and trg_lang == "Turkish":
        in_language = 'es'
        sp_src_path = "es"
        sp_trg_path = "tr"
        out_language = "tr"
        folder = "data/5Lang/Spanish to Turkish"
    elif src_lang == "Hindi" and trg_lang == "English":
        in_language = 'hi-in'
        sp_src_path = "hi"
        sp_trg_path = "en"
        out_language = "en"
        folder = "data/5Lang/Hindi to English"
    elif src_lang == "English" and trg_lang == "Hindi":
        in_language = 'en-in'
        sp_src_path = "en"
        sp_trg_path = "hi"
        out_language = "hi"
        folder = "data/5Lang/English to Hindi"
    elif src_lang == "English" and trg_lang == "Tamil":
        in_language = 'en-in'
        sp_src_path = "en"
        sp_trg_path = "ta"
        out_language = "ta"
        folder = "data/5Lang/English to Tamil"
    else:
        print("Invalid")
        exit()
    print(sp_src_path+"\n"+sp_trg_path)
    data_path = folder+"/data.csv"
    model_path = folder+"/model.pkl"
    ts_path = folder+"/ts1.pkl"
    spacy_german = spacy.blank(sp_src_path)
    spacy_english = spacy.blank(sp_trg_path) 
    def tokenize_german(text):
        return [token.text for token in spacy_german.tokenizer(text)]
    def tokenize_english(text):
        return [token.text for token in spacy_english.tokenizer(text)]
    # Take language input from the user
  # replace with the UID you want to search for

    # initialize the recognizer
    r = sr.Recognizer()

    # specify the path to the audio file
    audio_file = file 
    # use the audio file as the source
    with sr.AudioFile(audio_file) as source:
        # read the audio data
        audio = r.record(source)

    # recognize speech using Google Speech Recognition
    try:
        text =  r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    print(text)
    german = Field(tokenize=tokenize_german,
                lower=True,
                init_token="<sos>",
                eos_token="<eos>")

    english = Field(tokenize=tokenize_english,
                lower=True,
                init_token="<sos>",
                eos_token="<eos>")

    train_data, valid_data, test_data = TabularDataset.splits(path="", train=data_path, validation=data_path, test=data_path,
        format="csv", fields=[ ("English", english),("Hindi", german)],
        skip_header=True)
    german.build_vocab(train_data, max_size=10000, min_freq=3)
    english.build_vocab(train_data, max_size=10000, min_freq=3)

    german.build_vocab(train_data, max_size=10000, min_freq=3)
    english.build_vocab(train_data, max_size=10000, min_freq=3)

    e = list(german.vocab.__dict__.values())
    word_2_idx = dict(e[3])
    idx_2_word = {}
    for k,v in word_2_idx.items():
        idx_2_word[v] = k
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                                        batch_size = BATCH_SIZE, 
                                                                        sort_within_batch=True,
                                                                        sort_key=lambda x: len(x.Hindi),
                                                                        device = device)
    count = 0
    max_len_eng = []
    max_len_ger = []
    for data in train_data:
        max_len_ger.append(len(data.Hindi))
        max_len_eng.append(len(data.English))
        if count < 10 :
            count += 1
    count = 0
    for data in train_iterator:
        if count < 1 :
            temp_ger = data.Hindi
            temp_eng = data.English
            count += 1
    temp_eng_idx = (temp_eng).cpu().detach().numpy()
    temp_ger_idx = (temp_ger).cpu().detach().numpy()
    df_eng_idx = pd.DataFrame(data = temp_eng_idx, columns = [str("S_")+str(x) for x in np.arange(1, 33)])
    df_eng_idx.index.name = 'Time Steps'
    df_eng_idx.index = df_eng_idx.index + 1 
    # df_eng_idx.to_csv('/content/idx.csv')
    df_eng_word = pd.DataFrame(columns = [str("S_")+str(x) for x in np.arange(1, 33)])
    df_eng_word = df_eng_idx.replace(idx_2_word)

    input_size_encoder = len(german.vocab)
    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = 0.5

    encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
                            hidden_size, num_layers, encoder_dropout).to(device)

    input_size_decoder = len(english.vocab)
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = 0.5
    output_size = len(english.vocab)

    decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,
                            hidden_size, num_layers, decoder_dropout, output_size).to(device)
    for batch in train_iterator:
        break

    x = batch.English[1]

    learning_rate = 0.001
    step = 0

    model = Seq2Seq(encoder_lstm, decoder_lstm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = english.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    #contents = pickle.load(f) becomes...
    with open(model_path, 'rb') as f:
        model = CPU_Unpickler(f).load()
    # load the saved model from file
    with open(ts_path, 'rb') as f:
        ts1 = pickle.load(f)
    progress  = []
    for i,sen in enumerate(ts1):
        progress.append(TreebankWordDetokenizer().detokenize(sen))
    translated_sentence = translate_sentence(model, text, german, english, device,sp_src_path, max_length=50)
    progress.append(TreebankWordDetokenizer().detokenize(translated_sentence))
    ftext = progress[-1]
    ftext = re.sub(r'<eos>', '', ftext)
    ftext = re.sub(r'<unk>', '', ftext)
    print(ftext)
    predict_input = tokenizer.encode(text,
                                         truncation=True,
                                         padding=True,
                                         return_tensors="tf")

    tf_output = model1.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    probabilities = tf_prediction.numpy()[0]
    label = np.argmax(probabilities)
    confidence = probabilities[label]
    confidence = float(confidence)
    confidence = round(confidence * 100, 2)

    if labels[label] == 'Negative':
        reasons = []
        for word in negative_sentiment_words:
            if word in text.lower():
                reasons.append(word)
        return jsonify({'predictions': [{'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence}]})
    elif labels[label] == 'Positive':
        reasons = []
        for word in positive_sentiment_words:
            if word in text.lower():
                reasons.append(word)
            return json.dumps({'predictions': [{'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence}]})
    else:
        return json.dumps({'predictions': [{'text': text, 'sentiment': labels[label], 'confidence': confidence}]})


@app.route('/social', methods=['POST'])
def get_comments():
    # Set the API key
    api_key = 'AIzaSyDvOvhzBGEHLnDpuOBpJu0L1ALVUATl-HI'

    # Get the keyword, start date, and end date from the request
    keyword = request.json.get('keyword')
    start_date_str = request.json.get('start_date')
    end_date_str = request.json.get('end_date')

    # Convert start date and end date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Set the search parameters
    max_results = 50
    published_after = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    published_before = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Create a YouTube Data API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Call the search.list method to retrieve video results
    search_response = youtube.search().list(
        q=keyword,
        type='video',
        part='id',
        maxResults=max_results,
        publishedAfter=published_after,
        publishedBefore=published_before
    ).execute()

    # Extract the video IDs from the search results
    video_ids = [search_result['id']['videoId'] for search_result in search_response.get('items', [])]

    # Create an empty list to store comments and replies
    video_comments_list = []

    # Function to retrieve video comments and append them to the list
    def video_comments(video_id):
        # Create empty list for storing replies
        replies = []

        # Create YouTube resource object
        youtube = build('youtube', 'v3', developerKey=api_key)

        # Retrieve YouTube video comments
        video_response = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id
        ).execute()

        # Iterate over video response
        while video_response:
            # Extract required info from each result object
            for item in video_response['items']:
                # Extract comment
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']

                # Count the number of replies to the comment
                replycount = item['snippet']['totalReplyCount']

                # If replies exist, iterate through each reply
                if replycount > 0:
                    for reply in item['replies']['comments']:
                        # Extract reply
                        reply_text = reply['snippet']['textDisplay']
                        # Store reply in the list
                        replies.append(reply_text)

                # Append comment and replies to the list
                video_comments_list.append({'comment': comment, 'replies': replies})

                # Empty the replies list
                replies = []

            # Fetch the next page of comments
            if 'nextPageToken' in video_response:
                video_response = youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    pageToken=video_response['nextPageToken']
                ).execute()
            else:
                break

    # Ensure video_ids list is not empty
    if video_ids:
        # Choose the first video ID from the list
        video_id = video_ids[0]

        # Call the function to retrieve video comments
        video_comments(video_id)
    # Reddit data retrieval
    reddit_keyword = keyword
    subreddit_name = 'airtel'

    start = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')
    print("Start")
    print(start)
    print(end)
    print(reddit_keyword)

    # List to store matching comments from Reddit
    reddit_comments_list = []

    # Get subreddit object
    subreddit = reddit.subreddit(subreddit_name)

    # Loop through comments in subreddit and append matching ones to list
    for submission in subreddit.search(reddit_keyword, limit=None):
        print("came")
        submission.comments.replace_more(limit=20)
        for comment in submission.comments.list():
                reddit_comments_list.append({
                    'comment_id': comment.id,
                    'comment': comment.body,
                    'author': str(comment.author),
                    'created_utc': comment.created_utc
                })
                if len(reddit_comments_list) >= 30:
                    break

        if len(reddit_comments_list) >= 30:
            break
    print(reddit_comments_list)
    # Perform sentiment analysis on YouTube comments
    youtube_prediction = []
    youtube_texts = [comment['comment'] for comment in video_comments_list]

    for text in youtube_texts:
        predict_input = tokenizer.encode(text,
                                         truncation=True,
                                         padding=True,
                                         return_tensors="tf")

        tf_output = model.predict(predict_input)[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        probabilities = tf_prediction.numpy()[0]
        label = np.argmax(probabilities)
        confidence = float(probabilities[label])
        confidence = round(confidence * 100, 2)
        
        if labels[label] == 'Negative':
            reasons = []
            for word in negative_sentiment_words:
                if word in text.lower():
                    reasons.append(word)
            youtube_prediction.append({'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence})
        elif labels[label] == 'Positive':
            reasons = []
            for word in positive_sentiment_words:
                if word in text.lower():
                    reasons.append(word)
            youtube_prediction.append({'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence})
        else:
            youtube_prediction.append({'text': text, 'sentiment': labels[label], 'confidence': confidence})

    # Perform sentiment analysis on Reddit comments
    reddit_prediction = []
    reddit_texts = [comment['comment'] for comment in reddit_comments_list]

    for text in reddit_texts:
        predict_input = tokenizer.encode(text,
                                         truncation=True,
                                         padding=True,
                                         return_tensors="tf")

        tf_output = model.predict(predict_input)[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        probabilities = tf_prediction.numpy()[0]
        label = np.argmax(probabilities)
        confidence = float(probabilities[label])
        confidence = round(confidence * 100, 2)
        
        if labels[label] == 'Negative':
            reasons = []
            for word in negative_sentiment_words:
                if word in text.lower():
                    reasons.append(word)
            reddit_prediction.append({'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence})
        elif labels[label] == 'Positive':
            reasons = []
            for word in positive_sentiment_words:
                if word in text.lower():
                    reasons.append(word)
            reddit_prediction.append({'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence})
        else:
            reddit_prediction.append({'text': text, 'sentiment': labels[label], 'confidence': confidence})
    youtube_predictions = youtube_prediction if youtube_prediction else "null"
    reddit_predictions = reddit_prediction if reddit_prediction else "null"
    print("youtube")
    print(youtube_predictions)
    print("reddit")
    print(reddit_predictions)
    return jsonify({'youtube_predictions': youtube_predictions, 'reddit_predictions': reddit_predictions})

@app.route('/text', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided.'}), 400  # Return an error response with a 400 status code
        text = data.get('text')
        if text is None:
            return jsonify({'error': 'No text provided.'}), 400  # Return an error response with a 400 status code

        predict_input = tokenizer.encode(text,
                                         truncation=True,
                                         padding=True,
                                         return_tensors="tf")

        tf_output = model.predict(predict_input)[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        probabilities = tf_prediction.numpy()[0]
        label = np.argmax(probabilities)
        confidence = float(probabilities[label])
        confidence_percent = round(confidence * 100, 2)
        
        if labels[label] == 'Negative':
            reasons = []
            for word in negative_sentiment_words:
                if word in text.lower():
                    reasons.append(word)
            return json.dumps({'predictions': [{'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence_percent}]})
        elif labels[label] == 'Positive':
            reasons = []
            for word in positive_sentiment_words:
                if word in text.lower():
                    reasons.append(word)
            return json.dumps({'predictions': [{'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence_percent}]})
        else:
            return json.dumps({'predictions': [{'text': text, 'sentiment': labels[label], 'confidence': confidence_percent}]})
     
@app.route('/dataset', methods=['POST'])
def predict_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided.'}), 400  # Return an error response with a 400 status code
        df = pd.read_csv(file)
        texts = df['Text']
        predictions = []
        for text in texts:
            predict_input = tokenizer.encode(text,
                                             truncation=True,
                                             padding=True,
                                             return_tensors="tf")

            tf_output = model.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1)
            probabilities = tf_prediction.numpy()[0]
            label = np.argmax(probabilities)
            confidence = round(float(probabilities[label]) * 100, 2)  # Convert confidence to percentage with 2 decimal places

            if labels[label] == 'Negative':
                reasons = []
                for word in negative_sentiment_words:
                    if word in text.lower():
                        reasons.append(word)
                predictions.append({'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence})
            elif labels[label] == 'Positive':
                reasons = []
                for word in positive_sentiment_words:
                    if word in text.lower():
                        reasons.append(word)
                predictions.append({'text': text, 'sentiment': labels[label], 'reasons': reasons, 'confidence': confidence})
            else:
                predictions.append({'text': text, 'sentiment': labels[label], 'confidence': confidence})
        return json.dumps({'predictions': predictions})
    else:
        return "error"


# df_eng_word.to_csv('/content/Words.csv')
class EncoderLSTM(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
    super(EncoderLSTM, self).__init__()

    # Size of the one hot vectors that will be the input to the encoder
    #self.input_size = input_size

    # Output size of the word embedding NN
    #self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Regularization parameter
    self.dropout = nn.Dropout(p)
    self.tag = True

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = nn.Embedding(input_size, embedding_size)
    
    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)

  # Shape of x (26, 32) [Sequence_length, batch_size]
  def forward(self, x):

    # Shape -----------> (26, 32, 300) [Sequence_length , batch_size , embedding dims]
    embedding = self.dropout(self.embedding(x))
    
    # Shape --> outputs (26, 32, 1024) [Sequence_length , batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size]
    outputs, (hidden_state, cell_state) = self.LSTM(embedding)

    return hidden_state, cell_state


class DecoderLSTM(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, output_size):
    super(DecoderLSTM, self).__init__()

    # Size of the one hot vectors that will be the input to the encoder
    #self.input_size = input_size

    # Output size of the word embedding NN
    #self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Size of the one hot vectors that will be the output to the encoder (English Vocab Size)
    self.output_size = output_size

    # Regularization parameter
    self.dropout = nn.Dropout(p)

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = nn.Embedding(input_size, embedding_size)

    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)

    # Shape -----------> (1024, 4556) [embedding dims, hidden size, num layers]
    self.fc = nn.Linear(hidden_size, output_size)

  # Shape of x (32) [batch_size]
  def forward(self, x, hidden_state, cell_state):

    # Shape of x (1, 32) [1, batch_size]
    x = x.unsqueeze(0)

    # Shape -----------> (1, 32, 300) [1, batch_size, embedding dims]
    embedding = self.dropout(self.embedding(x))

    # Shape --> outputs (1, 32, 1024) [1, batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size] (passing encoder's hs, cs - context vectors)
    outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))

    # Shape --> predictions (1, 32, 4556) [ 1, batch_size , output_size]
    predictions = self.fc(outputs)

    # Shape --> predictions (32, 4556) [batch_size , output_size]
    predictions = predictions.squeeze(0)

    return predictions, hidden_state, cell_state


class Seq2Seq(nn.Module):
  def __init__(self, Encoder_LSTM, Decoder_LSTM):
    super(Seq2Seq, self).__init__()
    self.Encoder_LSTM = Encoder_LSTM
    self.Decoder_LSTM = Decoder_LSTM

  def forward(self, source, target, tfr=0.5):
    # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]
    batch_size = source.shape[1]

    # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]
    target_len = target.shape[0]
    target_vocab_size = len(english.vocab)
    
    # Shape --> outputs (14, 32, 5766) 
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

    # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)
    hidden_state, cell_state = self.Encoder_LSTM(source)

    # Shape of x (32 elements)
    x = target[0] # Trigger token <SOS>

    for i in range(1, target_len):
      # Shape --> output (32, 5766) 
      output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)
      outputs[i] = output
      best_guess = output.argmax(1) # 0th dimension is batch size, 1st dimension is word embedding
      x = target[i] if random.random() < tfr else best_guess # Either pass the next word correctly from the dataset or use the earlier predicted word

    # Shape --> outputs (14, 32, 5766) 
    return outputs
# Hyperparameters


def translate_sentence(model, sentence, german, english, device,sp_src_paths, max_length=50):
    spacy_ger = spacy.blank(sp_src_paths)

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)
    text_to_indices = [german.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.Encoder_LSTM(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model1 = TFBertForSequenceClassification.from_pretrained('tf_model')
labels = ['Negative', 'Positive']  # (0:negative, 1:positive)
      
# src_lang = "Tamil"
trg_lang = "English"


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
    