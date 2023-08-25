from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from datetime import datetime
from flask_cors import CORS
import praw
global sp_src_path,sp_trg_path,english,device
from flask_cors import CORS
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = TFBertForSequenceClassification.from_pretrained('tf_model')
labels = ['Negative', 'Positive']  # (0:negative, 1:positive)

negative_sentiment_words = []
with open('neg_words.txt', 'r') as file:
        for line in file:
            line = line.strip().lower()
            negative_sentiment_words.append(line)
positive_sentiment_words = []
with open('pos_words.txt', 'r') as file:
        for line in file:
            line = line.strip().lower()
            positive_sentiment_words.append(line)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
reddit = praw.Reddit(client_id='LchMksVUmRUeyg', client_secret='gb1XyXX-r0ycV9KKFM-ujFVNOogO_w', user_agent='Data Scraping')

def get_comments():    # Set the API key
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

