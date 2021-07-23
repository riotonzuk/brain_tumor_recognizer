# run by typing python3 main.py in a terminal 
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from ai2 import get_yolo_net, yolo_forward, yolo_save_img
from utils import get_base_url, allowed_file, and_syntax

# setup the webserver
'''
    coding center code
    port may need to be changed if there are multiple flask servers running on same server
    comment out below three lines of code when ready for production deployment
'''
# port = 12345
# base_url = get_base_url(port)
# app = Flask(__name__, static_url_path = base_url + 'static')

'''
    cv scaffold code
    uncomment below line when ready for production s
'''
app = Flask(__name__)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# set up yolo net for prediction
# you will need to change names, weights, and configuration files.
names_path = os.path.join('yolo', 'actual_obj.names')
LABELS = open(names_path).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weights_path = os.path.join('yolo', 'yolo.weights')
cfg_path = os.path.join('yolo', 'yolov4-custom.cfg')
net = get_yolo_net(cfg_path, weights_path)


@app.route('/')
# @app.route(base_url)
def home():
    return render_template('Home.html')

@app.route('/Meet-the-Team.html')
# @app.route(base_url + "Meet-the-Team.html")
def meetTheTeam():
    return render_template("Meet-the-Team.html")

@app.route('/Project-Timeline.html')
# @app.route(base_url + "Project-Timeline.html")
def projectTimeline():
    return render_template("Project-Timeline.html")

@app.route('/Definitions-+-YOLO.html')
# @app.route(base_url + 'Definitions-+-YOLO.html')
def yolo_def():
    return render_template('Definitions-+-YOLO.html')

@app.route('/uploads/<filename>', methods=['POST'])
@app.route('/', methods=['POST'])
# @app.route(base_url + '/uploads/<filename>', methods = ['POST'])
# @app.route(base_url, methods=['POST'])
def home_post(filename=None):
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('results', filename = filename))
    else:
        return redirect(url_for('home'))


@app.route('/uploads/<filename>')
# @app.route(base_url + '/uploads/<filename>')
def results(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    new_filename = "yolo_" + filename
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

    # image is a numpy array
    image = cv2.imread(image_path)

    (class_ids, labels, boxes, confidences) = yolo_forward(net, LABELS, image, confidence_level = 0.3)

    label_lst = []
    confidence_lst = []

    if len(class_ids) == 0:
        return render_template('results.html', labels = 'NO DETECTION', confidences = 'N/A' , filename = filename)
    else:
        yolo_save_img(image, class_ids, boxes, labels, confidences, COLORS, result_image_path)
        for label, confidence in zip(labels , confidences):
            label_lst.append(label)
            confidence_str = str(int(confidence * 100)) + '%'
            confidence_lst.append(confidence_str)
        label_str = and_syntax(label_lst)
        confidence_str = and_syntax(confidence_lst)
        return render_template('results.html', labels = label_str, confidences = confidence_str , filename = new_filename)




@app.route('/files/<path:filename>')
# @app.route(base_url + '/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment = True)


if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalc2.ai-camp.org to the site where you are editing this file.
    website_url = 'cocalc6.ai-camp.org'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    print(app.root_path)
    # remove debug=True when deploying it
    app.run(host = '0.0.0.0', port = port, debug=True)
    import sys; sys.exit(0)

    '''
    cv scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
