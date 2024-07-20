from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from googletrans import Translator

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def translate_text(original_text):
    translator = Translator()
    translation = translator.translate(original_text, dest='zh-CN')  # Change 'zh-CN' to the target language code
    return translation.text

@app.route('/')
def index():
    return render_template('index_page.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the content of the file (you'll need to replace this with your model inference logic)
        with open(file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()

        # Perform translation
        translated_text = translate_text(original_text)

        # Pass the original and translated text to the template
        return render_template('index_page.html', original_text=original_text, translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
