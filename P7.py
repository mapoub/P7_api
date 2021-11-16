from flask import Flask, jsonify, request
#import flasgger
#from flasgger import Swagger
import pickle
import pandas as pd 
import dill
import matplotlib.pyplot as plt
import lime
from lime import lime_tabular

import numpy as np
import seaborn as sns

from hashlib import sha256

KEY         = "P7"
SEUIL       = 0.725
PATH_HTML   = "/var/www/html/P7/"

app = Flask(__name__)
#Swagger(app)


#@app.route('/')
#def hello():
#    return "Hello world from flask"

#@app.route('/api', methods=['POST', 'DELETE', 'GET'])
#def hello2():
#    return jsonify({'Hello': 'World!'})

#df_test = pickle.load('test.pickle','rb',)
#modele  = pickle.load('model.pickle','rb')

with open('test.pickle', 'rb') as pickle_file:
    X_test = pickle.load(pickle_file)


with open('model.pickle', 'rb') as pickle_file:
    modele = pickle.load(pickle_file)

with open('id_client.pickle', 'rb') as pickle_file:
    id_client = pickle.load(pickle_file)
    df_id_client = pd.DataFrame(id_client)

with open('df_test.pickle', 'rb') as pickle_file:
    df_test = pickle.load(pickle_file)

@app.route('/api/persons')
def persons():
    dico = {'':'', 'client1':'client1', 'client2':'client2', 'client3':'client3', 'client4':'client4' }
    return jsonify(dico)


@app.route('/api/clients')
def clients():
    client= ",103625,105091,105134,104584,102887,104691,104381,103575,105973,101041,105630,100591,104417,106735,106438,106216,101903,106087,105248,102174,103830,101259,102506,103658,101398"
    # unique()
    #return jsonify(client)
    return client



@app.route('/api/person/<person_id>')
def person(person_id):
    dico = {}
    if person_id =="client1":
        dico["score"] = "35"
        response = jsonify(dico)
    elif person_id =="client2":
        response = jsonify({'score': '50'})
    elif person_id =="client3":
        response = jsonify({'score': '65'})
    elif person_id =="client4":
        response = jsonify({'score': '40'})
    #response = jsonify({'Hello': person_id})
    return response


# /api/indicator?id=105091&indic=105091
@app.route('/api/indicator')
def indicator():
    id = request.args.get('id')
    indic = request.args.get('indic')

    id_client = int(id)
    index = list(df_id_client[df_id_client.SK_ID_CURR == id_client ].index)[0]
    print("index:" + str(index))
    

    print("indicator")
    fig2, ax2 = plt.subplots()

    x = df_test[[indic]]  # df_train_target1[col]
    #x2 = df_train_target0[col]
    #ax = sns.distplot(x, color = '#005800')
    ax = sns.distplot(x, color = 'darkgreen')
    #ax2 = sns.distplot(x2)
    x = ax.lines[0].get_xdata()
    y = ax.lines[0].get_ydata()
    #plt.axvline(x[np.argmax(y)], color='red')
    plt.axvline(float(df_test.iloc[index:index+1][indic]), color='red')
    
    ax.lines[0].remove()
    #ax2.lines[0].remove()
    plt.show()
    id_ = id + indic + KEY
    sha =  sha256(id_.encode('utf-8')).hexdigest()
    fig2.savefig(PATH_HTML+sha+'.png')
    
    return sha

@app.route('/api/client/<id_client>')
def client(id_client):
#prediction, proba = predict(id_client, df)

#    dict_final = {
#        'prediction' : int(prediction),
#        'proba' : float(proba[0][0])
#        }

#    print('Nouvelle Prédiction : \n', dict_final)

#    return jsonify(dict_final)
#    #return = jsonify({'client': id_client})
#    return jsonify( id_client)
#'''      



    #index = list(df_test[df_test.SK_ID_CURR == id_client ].index)[0]

    #index = int(id_client) 
    print("id_client:<"+ id_client+">")
    #print(list(df_id_client.SK_ID_CURR))

    id_client = int(id_client)
    index = list(df_id_client[df_id_client.SK_ID_CURR == id_client ].index)[0]
    print("index:" + str(index))
    y_pred = modele.predict(X_test[index:index+1,:])
    y_proba = modele.predict_proba(X_test[index:index+1,:])
    

    #df_testr.iloc[50:51]
    #df_test
    dico = {}
    dico["score"] = int((y_proba[:,1] >= SEUIL).astype(int)) #str(y_pred[0])
    dico["proba0"] = str(round(y_proba[0][0]*100,2))
    dico["proba1"] = str(round(y_proba[0][1]*100,2))
    dico["seuil"] = str(100 - SEUIL*100)

    #dico["DAYS_BIRTH"] = str(int(df_test.iloc[index:index+1].DAYS_BIRTH))
    dico["json"] = (df_test.iloc[index:index+1]).to_json()
 

    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    feature_imp = pd.DataFrame(sorted(zip(modele.feature_importances_,df_test.columns)), columns=['Value','Feature'])
    fig, ax = plt.subplots()

    ax= sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:20], color ='#007500')
    plt.title('Variables importantes')
    plt.tight_layout()
    #plt.show()

    df_variables_princi = feature_imp.sort_values(by="Value", ascending=False)[:20].Feature

    fig.savefig(PATH_HTML+'feature_importance.png')


    with open('lime_.pickle', 'rb') as f: lime1 = dill.load(f)

    exp = lime1.explain_instance(pd.DataFrame(X_test).iloc[index],
                             modele.predict_proba,
                             #class_names=["favorable", "défavorable"],
                             num_samples=100)

    exp.show_in_notebook(show_table=True)
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    fig.savefig(PATH_HTML+'feature_importance_'+str(id_client)+'.png')

    df_variables = pd.DataFrame(exp.as_list())
    
    #list(df_variables[0])

    dico["variables"] = list(df_variables[0])
    dico["variables2"] = df_variables.to_json()

    dico["variables_princ"] = list(df_variables_princi)

    

    '''
    print("indicator")
    fig2, ax2 = plt.subplots()

    x = df_test[["DAYS_BIRTH"]]  # df_train_target1[col]
    #x2 = df_train_target0[col]
    ax = sns.distplot(x)
    #ax2 = sns.distplot(x2)
    x = ax.lines[0].get_xdata()
    y = ax.lines[0].get_ydata()
    #plt.axvline(x[np.argmax(y)], color='red')
    plt.axvline(float(df_testr.iloc[100:101].DAYS_BIRTH), color='red')
    ax.lines[0].remove()
    #ax2.lines[0].remove()
    plt.show()
    fig2.savefig('/var/www/html/P7/indicator.png')
    '''

#DAYS_BIRTH
#AMT_ANNUITY
#DAYS_EMPLOYED
#ANNUITY_INCOME_PERCENT

#dico = {
    #        'score': str(y_pred[0]),
    #        'proba0': str(y_proba[0][0]),
    #        'proba1': str(y_proba[0][1])
    #        }
    
    
    #return jsonify(y_pred)
    return jsonify(dico)


if __name__ == '__main__':
    app.run(debug=True)
