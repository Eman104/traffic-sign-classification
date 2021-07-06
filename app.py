from flask import Flask, render_template, request, send_from_directory
import numpy as np
from PIL import  Image
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')
#graph=tf.compat.v1.get_default_graph()

#graph = tf.get_default_graph()

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index2.html')

# dictionary to label all traffic signs class.
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

@app.route('/home', methods=['POST'])
def home():
    global COUNT
    global graph
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    img_arr = Image.open('static/{}.jpg'.format(COUNT))

    img_arr = img_arr.resize((30,30))
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = np.array(img_arr)


    #with graph.as_default():
    pred = model.predict_classes([img_arr])[0]
    #prediction = model._make_predict_function(img_arr)
    sign = classes[pred + 1]

    COUNT += 1
    return render_template('index2.html',prediction=sign)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)

