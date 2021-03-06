import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing, metrics


def read_data():
    df_data = pd.read_csv('Datathon2019/Datathon_dataset/-1_train_final.csv')
    df_data_test = pd.read_csv('Datathon2019/Datathon_dataset/mean_train_final.csv')
    data_list = df_data.values.tolist()
    return df_data, df_data_test, data_list

def plotData(data):
    plt.figure()
    plt.scatter([i for i in range(len(data))], [x[10] for x in data], s=3)
    plt.show()

def try_different_method(model,x_train,x_test,y_train,y_test,data_x_test,data_y_test, model_name):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, result)
    print('model:',model)
    print('score:',score)
    print('accuracy:',accuracy)
    print(result)

    # output: classification report
    print (metrics.classification_report(y_test, result))

    resultT = model.predict(data_x_test)
    accuracy_test = metrics.accuracy_score(data_y_test, resultT)
    print('testData_accuracy:',accuracy_test)
    # output: classification report
    print ('testData_F1:',metrics.classification_report(data_y_test, resultT))
    
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('%s\nscore: %f'%(model_name, score))
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    df_data, df_data_test, data_list = read_data()
    df_data.fillna(0, inplace=True)
    df_data_test.fillna(0, inplace=True)

    data_x = df_data[['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE', 'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num']]
    data_y = df_data['die_in_h']

    data_x_test = df_data_test[['nlr', 'lactate', 'bcd', 'GENDER', 'age', 'DISCHARGE_LOCATION', 'Neoplastic_disease', 'LIVER_DISEASE', 'CONGESTIVE_HEART_FAILURE', 'OTHER_NEUROLOGICAL', 'RENAL_FAILURE', 'total', 'resp_rate', 'sys_bp', 'temp', 'hr', 'bun', 'sodium', 'glucose_blood', 'hematocrit', 'o2', 'glucose_pleural', 'pH', 'potassium', 'cre', 'wbc', 'platelets', 'alb', 'row_num']]
    data_y_test = df_data_test['die_in_h']
    data_x_test = data_x_test.values
    data_y_test = data_y_test.values

    # print(data_y_test.head())

    # plotData(data_list)

    x_train,x_test,y_train,y_test = train_test_split(data_x, data_y, test_size = 0.33)

    iratation = 100

    superpa = []
    for i in range(iratation):
        rfc = ensemble.RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
        rfc_s = cross_val_score(rfc,x_train,y_train,cv=10).mean()
        superpa.append(rfc_s)
        print('now step:', i + 1)
    print('max_score:', max(superpa),'\nmax_forest:',superpa.index(max(superpa))+1)#????????????????????????????????????max(superpa))+1??????????????????????????????n_estimators
    plt.figure(figsize=[20,5])
    plt.plot(range(1,iratation+1),superpa)
    plt.show()

    param_test2= {'max_depth':range(2,14,2), 'min_samples_split':range(50,201,20)}
    gsearch2= GridSearchCV(estimator = ensemble.RandomForestClassifier(n_estimators= superpa.index(max(superpa))+1,
                                    min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10),
    param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)
    gsearch2.fit(x_train,y_train)
    print(gsearch2.best_params_)
    
    #?????????????????????????????????????????????????????????????????????
    rf1= ensemble.RandomForestClassifier(n_estimators= superpa.index(max(superpa))+1, max_depth=gsearch2.best_params_['max_depth'], min_samples_split=gsearch2.best_params_['min_samples_split'],
                                    min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10)
    rf1.fit(x_train,y_train)
    print(rf1.oob_score)
    #??????????????????0.984
    #??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????min_samples_split????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????min_samples_split??????????????????????????????min_samples_leaf???????????????
    
    #????????????????????????????????????????????????min_samples_split??????????????????????????????min_samples_leaf????????????
    param_test3= {'min_samples_split':range(50,150,20), 'min_samples_leaf':range(10,60,10)}
    gsearch3= GridSearchCV(estimator = ensemble.RandomForestClassifier(n_estimators= superpa.index(max(superpa))+1,max_depth=gsearch2.best_params_['max_depth'],
                                    max_features='sqrt' ,oob_score=True, random_state=10),
    param_grid = param_test3,scoring='roc_auc',iid=False, cv=5)
    gsearch3.fit(x_train,y_train)
    
    #?????????????????????????????????max_features?????????:
    param_test4= {'max_features':range(3,11,2)}
    gsearch4= GridSearchCV(estimator = ensemble.RandomForestClassifier(n_estimators= superpa.index(max(superpa))+1,max_depth=gsearch2.best_params_['max_depth'], min_samples_split=gsearch3.best_params_['min_samples_split'],
                                    min_samples_leaf=gsearch3.best_params_['min_samples_leaf'] ,oob_score=True, random_state=10),
    param_grid = param_test4,scoring='roc_auc',iid=False, cv=5)
    gsearch4.fit(x_train,y_train)
   
    
    #???????????????????????????????????????????????????????????????????????????
    rf2= ensemble.RandomForestClassifier(n_estimators= superpa.index(max(superpa))+1, max_depth=gsearch2.best_params_['max_depth'], min_samples_split=gsearch3.best_params_['min_samples_split'],
                                    min_samples_leaf=gsearch3.best_params_['min_samples_leaf'],max_features=gsearch4.best_params_['max_features'] ,oob_score=True, random_state=10)
    rf2.fit(x_train,y_train)
    print(rf2.oob_score)

    # model_LinearRegression = linear_model.LinearRegression()
    # model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
    # model_DecisionTreeClassifier = tree.DecisionTreeClassifier()
    # model_SVR = svm.SVR()
    model_RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators=superpa.index(max(superpa))+1,max_depth=gsearch2.best_params_['max_depth'], min_samples_split=gsearch3.best_params_['min_samples_split'],min_samples_leaf=gsearch3.best_params_['min_samples_leaf'],max_features=gsearch4.best_params_['max_features']) 

    # try_different_method(model_LinearRegression,x_train,x_test,y_train,y_test, 'LinearRegression')
    # try_different_method(model_KNeighborsRegressor,x_train,x_test,y_train,y_test, 'KNeighborsRegressor')
    # try_different_method(model_DecisionTreeClassifier,x_train,x_test,y_train,y_test,data_x_test,data_y_test, 'DecisionTreeClassifier')
    # try_different_method(model_SVR,x_train,x_test,y_train,y_test, 'SVR')
    try_different_method(model_RandomForestClassifier,x_train,x_test,y_train,y_test,data_x_test,data_y_test, 'RandomForestClassifier')

    
