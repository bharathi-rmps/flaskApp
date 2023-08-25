from flask import *
from flask_cors import CORS
from Churn.churn import process_file
from Sentiment.sentiment import predict_text, predict_file, predict_audio
from Social.social import get_comments
from Fake.fake import predict_fake_text, predict_fake_file
from Meme.meme import analyze_meme


application = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


############################################################################################
#                 SENTIMENT
############################################################################################

@application .route('/text_sentiment', methods=['POST'])
def text_sentiment():
    return predict_text()

@application .route('/file_sentiment', methods=['POST'])
def file_sentiment():
    return predict_file()

@application .route('/audio_sentiment', methods=['POST'])
def audio_sentiment():
    return predict_audio()


############################################################################################
#                  CHURN
############################################################################################

@application .route('/churn', methods=['POST'])
def churn():
    return process_file()


############################################################################################
#                  SOCIAL
############################################################################################

@application .route('/social', methods=['POST'])
def social_analysis():
    return get_comments()


############################################################################################
#                  Fake
############################################################################################

@application .route('/fake_text', methods=['POST'])
def fake_analysis_text():
    return predict_fake_text()

@application .route('/fake_file', methods=['POST'])
def fake_analysis_file():
    return predict_fake_file()


############################################################################################
#                  Meme
############################################################################################

@application .route('/meme', methods=['POST'])
def meme_analysis():
    return analyze_meme()

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)
    