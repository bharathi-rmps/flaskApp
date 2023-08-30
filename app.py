from flask import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['POST'])
def indexPage():
    return "True man it came!"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
    
