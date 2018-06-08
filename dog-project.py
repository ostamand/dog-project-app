# Udacity Deep Learning Nanodegree
# Project: Dog Breed Classifier 
# June 2018

from model import dog_identification as model
from flask import Flask, url_for, request, render_template, flash, redirect
from werkzeug import secure_filename
import os
import pdb

# python dog-project.py

# export FLASK_DEBUG=0
# export FLASK_ENV=development
# export FLASK_APP=dog-project.py
# flask run --no-reload

# Virtual Environment 
# python3 -m virtualenv env
# source env/bin/activate
# which python
# deactivate

IMG_FOLDER = 'static/img'

m = model.DogIdentification()
m.build()

app = Flask(__name__)
app.config['IMG_FOLDER'] = IMG_FOLDER

def _get_img_path(image):
    return os.path.join(app.config['IMG_FOLDER'], image)

def _get_ref_img_path(ref):
    if len(ref.split()) > 1:
        ref = ref.replace(' ', '_')

    return os.path.join('img/ref', ref + '.jpg')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # http://flask.pocoo.org/docs/1.0/patterns/fileuploads/#uploading-files
        # save file in temp folder
        # redirect to result page
        filename = secure_filename(file.filename)
        file.save(_get_img_path(filename))
        return redirect(url_for('result', image=filename))
        
    return render_template('index.html')
    
@app.route('/delete/<image>') 
def delete(image):
    # call to delete selected 
    if request.method == 'POST':
        to_delete = _get_img_path(image)
    
@app.route('/result/<image>')
def result(image):
    # check if file exists in img folder
    # process file
    # render result template 
    # ajax call to delete file?
    code, description, predictions = m.process(image=_get_img_path(image))

    if code == model.ProcessCode.HUMAN_FACE:
        info = 'human_face'
    elif code == model.ProcessCode.NEITHER:
        info = 'neither'
    elif code == model.ProcessCode.DOG:
        info = 'dog'

    return render_template('result.html', info=info, predictions=predictions, 
    img_file = os.path.join('img', image), img_ref_file=_get_ref_img_path(predictions['breeds'][0]))

if __name__ == '__main__':
    app.run()