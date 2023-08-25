from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy
from collections import Counter
from io import BytesIO


def process_file():
    file = request.files['file']
    print(file)
    file_content = file.read()
    data = pd.read_csv(BytesIO(file_content))
    data.dropna(axis="columns", how="any", inplace=True)
    data['Sentiment'] = data['Sentiment'].str.lower()
    df = data.groupby('Sentiment')
    datasets = {}
    for groups, data in df:
        datasets[groups] = data

    wordlist = []
    with open('words.txt', 'r') as file:
        for line in file:
            line = line.strip().lower()
            wordlist.append(line)

    neg = datasets['negative']
    pos = datasets['positive']

    neglist = []
    nlp = spacy.blank("en")
    for sentence in neg['Text']:
        doc = nlp(sentence)
        lowercase_tokens = [token.text.lower() for token in doc]
        for token in lowercase_tokens:
            if token in wordlist:
                neglist.append(token)

    poslist = []
    nlp = spacy.blank("en")
    for sentence in pos['Text']:
        doc = nlp(sentence)
        lowercase_tokens = [token.text.lower() for token in doc]
        for token in lowercase_tokens:
            if token in wordlist:
                poslist.append(token)

    neg_word_freq = Counter(neglist)
    neglength = len(neg)

    # Find the top 5 most frequent words
    top_five = neg_word_freq.most_common(5)

    # Sort the result from highest to lowest frequency
    sorted_result = sorted(top_five, key=lambda x: x[1], reverse=True)

    # Build the result dictionary
    result = {}
    for word, freq in sorted_result:
        percent_freq = (freq / neglength) * 100
        result[word] = percent_freq

    print(result)
    return jsonify(result)

