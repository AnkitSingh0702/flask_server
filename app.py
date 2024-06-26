from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle


app = Flask(__name__)
CORS(app)

try:
    model = pickle.load(open('models/finalized_model.pkl', 'rb'))
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    
except Exception as e:
    
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
      
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)



