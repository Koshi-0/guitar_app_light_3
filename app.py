from flask import Flask, render_template, request, redirect, url_for, flash
import torch
from model import inference_cnn
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            chord_name = classify_image(filepath)
            return render_template('index.html', chord_name=chord_name)
    return render_template('index.html', chord_name=None)

def classify_image(filepath):
    return inference_cnn(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)