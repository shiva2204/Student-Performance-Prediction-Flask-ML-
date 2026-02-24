from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('student_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capture inputs from the web form
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f'Predicted Grade: {output}')

if __name__ == "__main__":
    app.run(debug=True)
