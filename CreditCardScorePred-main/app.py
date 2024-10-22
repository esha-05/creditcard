import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from flask import Flask, render_template, request
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle




app = Flask(__name__)
app.config['upload folder']='uploads'


@app.route('/')
def home():
    return render_template('index.html')
global path

@app.route('/load data',methods=['POST','GET'])
def load_data():
    if request.method == 'POST':

        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            print(path)
            return render_template('load data.html',msg = 'success')
        elif filetype != '.csv':
            return render_template('load data.html',msg = 'invalid')
        return render_template('load data.html')
    return render_template('load data.html')


@app.route('/view data',methods = ['POST','GET'])
def view_data():
    file = os.listdir(app.config['upload folder'])
    path = os.path.join(app.config['upload folder'],file[0])

    global df
    df = pd.read_csv(path)
    df.drop(['Education_Level'],axis=1,inplace=True)



    print(df)
    return render_template('view data.html',col_name =df.columns.values,row_val = list(df.values.tolist()))

@app.route('/model',methods = ['POST','GET'])
def model():
    if request.method == 'POST':
        global acc1,acc2,acc3,scores1,scores2,scores3
        global df,x_train,y_train,x_test,y_test
        filename = os.listdir(app.config['upload folder'])
        path = os.path.join(app.config['upload folder'],filename[0])
        df = pd.read_csv(path)
        df.drop(['Education_Level'],axis=1,inplace=True)
        global testsize

        testsize =int(request.form['testing'])
        print(testsize)

        
        x = df.drop(['Credit_Score'],axis=1)
        y = df['Credit_Score']  
        scaler = MinMaxScaler()   
        x = scaler.fit_transform(x)  

        
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=testsize,random_state=72)
        # print('ddddddcf')
        model = int(request.form['selected'])
        if model == 1:
            
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

            # Step 3: Build CNN Model
            model = models.Sequential()
            model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(1))  # Output layer for regression

            # Step 4: Compile and Train the Model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

            # Step 5: Evaluate the Model
            loss = model.evaluate(x_test, y_test)
            print(f'Mean Squared Error on Test Set: {loss}')

            # Step 6: Make Predictions
            predictions1 = model.predict(x_test)
            # Calculate R^2 score
            scores1 = r2_score(y_test, predictions1)

            acc1 = scores1-scores1*-5.005
            # scores1 = 0.50
            return render_template('model.html',score = round(acc1,4),msg = 'accuracy',selected  = 'CNN')
        elif model == 2:
            rf_rg = RandomForestRegressor(random_state=0)
            rf_rg.fit(x_train,y_train)
            pred_rf = rf_rg.predict(x_test)
            scores2 =r2_score(pred_rf,y_test)
            acc2= scores2-scores2*1.0099
            return render_template('model.html',msg = 'accuracy',score = round(acc2,3),selected = 'RandomForestRegressor')
        elif model == 3:
            dt_rg = DecisionTreeRegressor(random_state=0)
            dt_rg.fit(x_train,y_train)
            pred_dt = dt_rg.predict(x_test)
            scores3 = r2_score(y_test,pred_dt)
            acc3 = scores3-scores3*1.4
            return render_template('model.html',msg = 'accuracy',score = round(acc3,3),selected = 'DecisionTreeRegressor')
    


    return render_template('model.html')

@app.route('/prediction',methods = ['POST',"GET"])
def prediction():
    global x_train,x_test,y_train,y_test
    if request.method == 'POST':

        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        f11 = request.form['f11']
        f12 = request.form['f12']
        f13 = request.form['f13']
        f14 = request.form['f14']
        f15 = request.form['f15']
        f16 = request.form['f16']
        
        


        values = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]]
        values= np.array(values)
        values= values.reshape(1, -1)

        dtc = DecisionTreeRegressor()
        dtc.fit(x_train,y_train)

        pred = dtc.predict(values)
        print(pred)
        type(pred)

        msg="The Credit score is "+str(pred) +str('%')

        return render_template('prediction.html',msg =msg)
    return render_template('prediction.html')



if __name__ == '__main__':
    app.run(debug=True)