from flask import *
from flask_cors import CORS
from Churn.churn import process_file
from Sentiment.sentiment import predict_text, predict_file, predict_audio
from Social.social import get_comments
from Fake.fake import predict_fake_text, predict_fake_file
from Meme.meme import analyze_meme


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


############################################################################################
#                 SENTIMENT
############################################################################################

@app.route('/text_sentiment', methods=['POST'])
def text_sentiment():
    return predict_text()

@app.route('/file_sentiment', methods=['POST'])
def file_sentiment():
    return predict_file()

@app.route('/audio_sentiment', methods=['POST'])
def audio_sentiment():
    return predict_audio()


############################################################################################
#                  CHURN
############################################################################################

@app.route('/churn', methods=['POST'])
def churn():
    return process_file()


############################################################################################
#                  SOCIAL
############################################################################################

@app.route('/social', methods=['POST'])
def social_analysis():
    return get_comments()


############################################################################################
#                  Fake
############################################################################################

@app.route('/fake_text', methods=['POST'])
def fake_analysis_text():
    return predict_fake_text()

@app.route('/fake_file', methods=['POST'])
def fake_analysis_file():
    return predict_fake_file()


############################################################################################
#                  Meme
############################################################################################

@app.route('/meme', methods=['POST'])
def meme_analysis():
    return analyze_meme()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    
