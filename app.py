from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)   

sc=pickle.load(open('diabetes_scalar.pkl','rb'))
model=pickle.load(open('diabetes_model.pkl','rb'))

sc1=pickle.load(open('heart_disease_scalar.pkl','rb'))
model1=pickle.load(open('heart_disease_model.pkl','rb'))

sc2=pickle.load(open('obesity_scalar.pkl','rb'))
model2=pickle.load(open('obesity_model.pkl','rb'))

sc3=pickle.load(open('heart_stroke_scalar.pkl','rb'))
model3=pickle.load(open('heart_stoke_model.pkl','rb'))

sc4=pickle.load(open('lung_cancer_scalar.pkl','rb'))
model4=pickle.load(open('lung_cancer_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('/demo.html')

@app.route('/diabetes')
def diabetes():
    return render_template('/diabetes_index.html')

@app.route('/submit',methods=['POST'])
def submit():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model.predict(sc.transform(final_features))
    return render_template('/diabetes_output.html',prediction=pred)

@app.route('/heartdisease')
def heartdisease():
    return render_template('/heart_disease_index.html')

@app.route('/submit1',methods=['POST'])
def submit1():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model1.predict(sc1.transform(final_features))
    return render_template('/heart_disease_output.html',prediction=pred)

@app.route('/obesity')
def obesity():
    return render_template('/obesity_index.html')

@app.route('/submit2',methods=['POST'])
def submit2():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model2.predict(sc2.transform(final_features))
    return render_template('/obesity_output.html',prediction=pred)

@app.route('/heartstroke')
def heartstroke():
    return render_template('/heart_stroke_index.html')

@app.route('/submit3',methods=['POST'])
def submit3():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model3.predict(sc3.transform(final_features))
    return render_template('/heart_stroke_output.html',prediction=pred)

@app.route('/lungcancer')
def lungcancer():
    return render_template('/lung_cancer_index.html')

@app.route('/submit4',methods=['POST'])
def submit4():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model4.predict(sc4.transform(final_features))
    return render_template('/lung_cancer_output.html',prediction=pred)

@app.route('/about')
def about():
    return render_template('/about.html')

if __name__ == "__main__":
    app.run(port=5000)