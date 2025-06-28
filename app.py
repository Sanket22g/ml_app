from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load your trained model
model = pickle.load(open('placement_model.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask ML Placement Predictor API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form data
        cgpa = float(request.form.get('cgpa', 0))
        iq = float(request.form.get('iq', 0))
        profile_score = float(request.form.get('profile_score', 0))

        # Prepare input for model
        input_query = np.array([[cgpa, iq, profile_score]])

        # Predict
        prediction = model.predict(input_query)

        # Return response
        result = {'placement': '1' if prediction[0] == 1 else '0'}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
