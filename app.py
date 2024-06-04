# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import os

# app = Flask(__name__)
# CORS(app)

# # Load the trained model and vectorizer
# model = pickle.load(open('models/finalized_model.pkl', 'rb'))
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
# print(os.path.exists('models/finalized_model.pkl'))
# print(os.path.exists('models/vectorizer.pkl'))
# @app.route('/', methods=['POST'])
# def predict():
#     data = request.json
#     if 'text' not in data:
#         return jsonify({'error': 'Invalid input data'}), 400

#     text = [data['text']]
#     tf_text = vectorizer.transform(text)
#     prediction = model.predict(tf_text)
#     return jsonify({'prediction': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import os

# app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# # Load the trained model and vectorizer
# model = pickle.load(open('models/finalized_model.pkl', 'rb'))
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# # Check if the model files exist
# print(os.path.exists('models/finalized_model.pkl'))
# print(os.path.exists('models/vectorizer.pkl'))

# @app.route('/')
# def home():
#     """Route for the root URL. Returns a simple message."""
#     return "Fake News Detection API is running!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Route for predicting fake news.
    
#     Expects a JSON object with a 'text' field containing the news text.
#     Returns a JSON response with the prediction result.
#     """
#     data = request.json  # Get the JSON data from the request
#     if 'text' not in data:
#         # If 'text' field is missing, return an error response
#         return jsonify({'error': 'Invalid input data'}), 400

#     text = [data['text']]
#     tf_text = vectorizer.transform(text)
#     prediction = model.predict(tf_text)
    
#     # Return a JSON response with the prediction result
#     return jsonify({'prediction': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import os
# app = Flask(__name__)
# CORS(app)

# # Load the trained model and vectorizer
# model = pickle.load(open('models/finalized_model.pkl', 'rb'))
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
# # # Check if the model files exist
# print(os.path.exists('models/finalized_model.pkl'))
# print(os.path.exists('models/vectorizer.pkl'))
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     if 'text' not in data:
#         return jsonify({'error': 'Invalid input data'}), 400

#     text = [data['text']]
#     tf_text = vectorizer.transform(text)
#     prediction = model.predict(tf_text)
#     return jsonify({'prediction': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model and vectorizer
try:
    model = pickle.load(open('models/finalized_model.pkl', 'rb'))
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error("Error loading model or vectorizer: %s", e)
    raise

@app.route('/', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        
        text = [data['text']]
        tf_text = vectorizer.transform(text)
        prediction = model.predict(tf_text)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)



# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle

# app = Flask(__name__)
# CORS(app)

# # Load the trained model and vectorizer
# model = pickle.load(open('models/finalized_model.pkl', 'rb'))
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     if 'text' not in data:
#         return jsonify({'error': 'Invalid input data'}), 400

#     text = [data['text']]
#     tf_text = vectorizer.transform(text)
#     prediction = model.predict(tf_text)
    
#     # Map the prediction to a human-readable label
#     label = 'REAL' if prediction[0] == 1 else 'FAKE'
#     return jsonify({'prediction': label})

# if __name__ == '__main__':
#     app.run(debug=True)
