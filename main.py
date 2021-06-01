import copy
import json
from flask import Flask,jsonify,request
import flask
import os
import urls
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

app = Flask(__name__)

details = []
tree_model = DecisionTreeClassifier()
class_ref = pd.DataFrame()
prediction = ''

@app.route('/model', methods = ['POST'])
def trainingModel():
    global tree_model,class_ref
    class_ref = pd.read_csv('dataset/class.csv').loc[:, ['Class_Number', 'Class_Type']]
    zoo = pd.read_csv('dataset/zoo.csv').iloc[:, 1:]
    zoo_X = zoo.drop('class_type', axis=1)
    zoo_y = zoo.loc[:, 'class_type']
    X_train, X_test, y_train, y_test = train_test_split(zoo_X, zoo_y, test_size=0.15)
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    print("model complete")
    return ""


@app.route('/animaldetails', methods = ['POST'])
def animaldetailsPost():
    global details, prediction
    # request_data = request.data
    # request_data = json.loads(request_data.decode('utf-8'))
    list_stuff = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
    # details = list()
    # for item in list_stuff:
    #     details.append = request_data[item]
    # print(details)
    data = [[ 0,0 ,1,0,0,1,1,0,0,1,0,1,0,1,1,0, ],]

    df = pd.DataFrame(data,columns = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize'])
    predicted_val_tree = tree_model.predict(df)
    print(type(df))

    print(df)
    df.reset_index(drop=True, inplace=True)
    predicted_data_tree = pd.concat((df, pd.Series(predicted_val_tree, name='predicted')), axis=1)
    predicted_data_tree.head()
    class_ref_dct = dict(zip(class_ref['Class_Number'], class_ref['Class_Type']))
    predicted_data_tree['predicted'].replace(class_ref_dct, inplace=True)
    predicted_data_tree.head()
    prediction = str(predicted_data_tree['predicted'])
    
    
    return ""

@app.route('/prediction', methods = ['GET'])
def result():
    return jsonify({'prediction' : prediction})
    

if __name__ == "__main__":
    app.run(host='0.0.0.0',port =5000)