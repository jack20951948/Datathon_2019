import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing, metrics

def read_data():
    df_data_test = pd.read_csv('Datathon2019/Datathon_dataset/mean_train_final.csv')
    df_data = pd.read_csv('Datathon2019/Datathon_dataset/non_null_data.csv')
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
    
    plt.figure(figsize=[20,5])
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

    x_train,x_test,y_train,y_test = train_test_split(data_x, data_y, test_size = 0.2)


    # superpa = []
    # for i in range(100):
    #     rfc = ensemble.RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
    #     rfc_s = cross_val_score(rfc,data_x,data_y,cv=10).mean()
    #     superpa.append(rfc_s)
    #     print('now step:',i)
    # print('max_forest:', max(superpa),superpa.index(max(superpa))+1)#????????????????????????????????????max(superpa))+1??????????????????????????????n_estimators
    # plt.figure(figsize=[20,5])
    # plt.plot(range(1,101),superpa)
    # plt.show()

    # model_LinearRegression = linear_model.LinearRegression()
    # model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
    # model_DecisionTreeClassifier = tree.DecisionTreeClassifier()
    # model_SVR = svm.SVR()
    model_RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators=45) 

    # try_different_method(model_LinearRegression,x_train,x_test,y_train,y_test, 'LinearRegression')
    # try_different_method(model_KNeighborsRegressor,x_train,x_test,y_train,y_test, 'KNeighborsRegressor')
    # try_different_method(model_DecisionTreeClassifier,x_train,x_test,y_train,y_test, 'DecisionTreeClassifier')
    # try_different_method(model_SVR,x_train,x_test,y_train,y_test, 'SVR')
    try_different_method(model_RandomForestClassifier,x_train,x_test,y_train,y_test,data_x_test,data_y_test, 'RandomForestClassifier')

    feat_labels = data_x.columns[:]

    # n_estimators???????????????????????? # n_jobs ?????? ???????????????=1??? ???????????????????????????????????????????????????-1???????????????????????????????????? 
    forest = ensemble.RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1) 
    forest.fit(x_train, y_train) 
    # ????????????????????????????????????????????????????????? 
    # feature_importances_ ???????????????????????????????????? 
    importances = forest.feature_importances_ 
    print("????????????",importances) 
    x_columns = data_x.columns[:] 
    indices = np.argsort(importances)[::-1] 
    for f in range(x_train.shape[1]): 
        # ??????????????????????????????????????????????????????????????????????????????????????????????????? # ?????????????????????????????????????????? 
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])) # ??????????????????????????????????????????????????? 
    threshold = 0.01 
    # x_selected = x_train[:,importances > threshold] 
    # ????????? import matplotlib.pyplot as plt 
    plt.figure(figsize=(10,6)) 
    plt.title("Rank of features",fontsize = 18) 
    plt.ylabel("import level",fontsize = 15,rotation=90) 
    plt.rcParams['font.sans-serif'] = ["SimHei"] 
    plt.rcParams['axes.unicode_minus'] = False 
    for i in range(x_columns.shape[0]): 
        plt.bar(i,importances[indices[i]],color='orange',align='center') 
    plt.xticks(np.arange(x_columns.shape[0]),x_columns,rotation=90,fontsize=15) 
    plt.show()

    

    

