import clip
from PIL import Image
from flask import *
app = Flask(__name__)
import torch

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
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)