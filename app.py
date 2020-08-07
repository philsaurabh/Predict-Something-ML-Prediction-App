from flask import Flask,render_template
import joblib
import pickle
from flask import request
import numpy as np
import os

app=Flask(__name__,template_folder='templates')
flag=1
dir_path = os.path.dirname(os.path.realpath(__file__))

model4=pickle.load(open('model4.pkl','rb'))


@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/forest")
def forest():
    return render_template("forest_fire.html")

@app.route("/iris")
def iris():
    return render_template("iris.html")

@app.route("/glass")
def glass():
    return render_template("glass.html")

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif(size==30):#Cancer
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = joblib.load("model2")
        result =loaded_model.predict(to_predict)
    elif(size==4):#Iris
        loaded_model = joblib.load("model3")
        result =loaded_model.predict(to_predict)
    elif(size==9):#Iris
        loaded_model = joblib.load("model5")
        result =loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==30):#Cancer
            result = ValuePredictor(to_predict_list,30)
            flag=0
        elif(len(to_predict_list)==8):#Daiabtes
            result = ValuePredictor(to_predict_list,8)
            flag=0
        elif(len(to_predict_list)==11):#heart
            result = ValuePredictor(to_predict_list,11)
            flag=0
        elif(len(to_predict_list)==4):#iris
            result = ValuePredictor(to_predict_list,4)
            flag=1
        elif(len(to_predict_list)==9):#glass
            result = ValuePredictor(to_predict_list,9)
            flag=2
    if(flag==0):
        if(int(result)==1):
            prediction='Looks like you are Suffering!'
        else:
            prediction='You are Healthy for now. Take Care!' 
    elif(flag==1):
        if(int(result)==0):
            prediction='The variety is Setosa!'
        elif(int(result)==1):
            prediction='The variety is Versicolor!'
        else:
            prediction='The variety is Virginica!'
    else:
        if(int(result)==1):
            prediction='It is type 1 Glass!'
        elif(int(result)==2):
            prediction='It is type 2 Glass!'
        elif(int(result)==3):
            prediction='It is type 3 Glass!'
        elif(int(result)==5):
            prediction='It is type 5 Glass!'
        elif(int(result)==6):
            prediction='It is type 6 Glass!'
        else:
            prediction='It is type 7 Glass!'
    return(render_template("result.html", prediction=prediction))

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model4.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        prediction='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output)
    else:
        prediction='Your Forest is safe.\n Probability of fire occuring is {}'.format(output)
    return(render_template("result.html", prediction=prediction))

@app.route('/predict_em',methods=['POST'])
def predict_em():
    
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)