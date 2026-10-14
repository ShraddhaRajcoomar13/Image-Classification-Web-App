from flask import Flask, request, render_template, jsonify
import os
from config import CLASSES
from predict_image import predict, model   # model comes from predict_image's load

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', classes=CLASSES)


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()
        results = predict(image_bytes)
        if 'error' in results:
            return jsonify(results), 400
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)