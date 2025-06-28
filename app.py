from flask import Flask, request, jsonify
import pickle
import numpy as np

model=pickle.load(open('placement_model.pkl', 'rb'))
app= Flask(__name__)
@app.route('/')
def home():
    return "welcome to the flask app"

@app.route('/predict',methods=['POST'])
def predict():
    cgpa=request.form.get('cgpa')
    iq=request.form.get('iq')
    profile_score=request.form.get('profile_score')

    input_query=np.array([[cgpa, iq, profile_score]], dtype=float)
    prediction=model.predict(input_query)
    result = {'placement_status': 'Placed' if prediction[0] == 1 else 'Not Placed'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)