from flask import Flask, jsonify, request
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

model = pickle.load(open('pipe (5).pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World'

@app.route('/predict',methods = ['POST'])
def predict():
    area = request.form.get('area')
    bedrooms = request.form.get('bedrooms')	
    bathrooms =request.form.get('bathrooms')
    mainroad =  request.form.get('mainroad')	
    guestroom =  request.form.get('guestroom') 
    basement =  request.form.get('basement')
    hotwaterheating =  request.form.get('hotwaterheating')	
    airconditioning =  request.form.get('airconditioning')
    parking =  request.form.get('parking')
    furnishingstatus =  request.form.get('furnishingstatus')

    dictionary = {'area':area,
              'bedrooms':bedrooms,
              'bathrooms':bathrooms,
              'mainroad':mainroad,
              'guestroom':guestroom,
              'basement':basement,
              'hotwaterheating':hotwaterheating,
              'airconditioning':airconditioning,
              'parking':parking,
              'furnishingstatus':furnishingstatus
    }

        test_input = np.array([area,bedrooms,bathrooms,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,furnishingstatus],dtype = object).reshape(1,10)   
    result  = model.predict(test_input)
    print(result)
    return jsonify({'placement':str(result)})

if __name__ == '__main__':
    app.run(debug = True)
