from flask import Flask, render_template, request, url_for
from flask_wtf import FlaskForm
from tensorflow.keras.models import load_model
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

model = load_model('VGG19##.h5')

class UploadFileForm(FlaskForm):
    file = FileField ("Upload Mammogram :")
    submit = SubmitField ("Detect")

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    results = []
    if request.method == 'POST' and form.validate_on_submit():
        
        
        file = form.file.data

        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        
        img = load_img(file_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        result = model.predict(x)
        
        if result <= 0.5:
         prediction = 'Benign'
        else:
         prediction = 'Malignant'

        results=prediction
        
        print('Results: ', results)
        
        return render_template('index.html', form=form, file_path=url_for('static', filename='files/' + filename), results=results,confidence=result[0][0])

    return render_template('index.html', form=form)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=80 , debug=True)









'''''
from flask import Flask, render_template, request, url_for
from flask_wtf import FlaskForm
from tensorflow.keras.models import load_model
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Load the pre-trained ResNet50V2 model
model = load_model('ResNet98.h5')

class UploadFileForm(FlaskForm):
    file = FileField ("File")
    submit = SubmitField ("Upload File")

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    results = []
    if request.method == 'POST' and form.validate_on_submit():
        
        
        file = form.file.data

        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        
        img = load_img(file_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        
        result = model.predict(x)
        # prediction = 'Benign' if result <= 0.5 else 'Malignant'
        print('result var type: ', type(result))

        # Get the index of the top predicted class
        top_class = np.argmax(result)

        # Convert the predicted class index to a string label
        classes = ['Benign', 'Malignant']
        result_str = classes[top_class]

        print('Result:', result)
        print('Top class:', top_class)
        print('Result str:', result_str)

        results.append(result_str)

        print('Results: ', results)
        
        return render_template('iii.html', form=form, file_path=url_for('static', filename='files/' + filename), confidence=result[0][0], results=results)

    return render_template('iii.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
    '''
