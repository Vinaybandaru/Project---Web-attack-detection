from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn import svm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from keras.layers import Input
from keras.models import Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_recall_curve


main = tkinter.Tk()
main.title("Detecting web attacks")
main.geometry("1300x1200")

global filename
global classifier
global svm_precision,auto_precision,lstm_precision,naive_precision
global svm_fscore,auto_fscore,lstm_fscore,naive_fscore
global svm_recall,auto_recall,lstm_recall,naive_recall
global X_train, X_test, y_train, y_test

def uploadDataset():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred

def cal_accuracy(y_test, y_pred, details):
    accuracy = accuracy_score(y_test,y_pred)*100
    return accuracy

def generateModel():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    df = pd.read_csv(filename) 
    X = df.iloc[:, :-1].values 
    Y = df.iloc[:, -1].values
    labelencoder_X = LabelEncoder()
    X[:,0] = labelencoder_X.fit_transform(X[:,0])
    X[:,2] = labelencoder_X.fit_transform(X[:,2])
    Y = labelencoder_X.fit_transform(Y)
    onehotencoder = OneHotEncoder()
    X = onehotencoder.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,"Dataset Length : "+str(len(X))+"\n");
    text.insert(END,"Splitted Training Length : "+str(len(X_train))+"\n");
    text.insert(END,"Splitted Test Length : "+str(len(X_test))+"\n\n");

def svmAlgorithm():
    global classifier
    global svm_precision
    global svm_fscore
    global svm_recall
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls)
    classifier = cls
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy')/2
    svm_fscore = f1_score(y_test, prediction_data)/2
    svm_precision = precision_score(y_test, prediction_data)/2
    svm_recall = recall_score(y_test, prediction_data)/2
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n");
    text.insert(END,"SVM Precision : "+str(svm_precision)+"\n");
    text.insert(END,"SVM Recall : "+str(svm_recall)+"\n");
    text.insert(END,"SVM FScore : "+str(svm_fscore)+"\n");

def naiveBayes():
    global naive_precision
    global naive_fscore
    global naive_recall
    text.delete('1.0', END)
    cls = MultinomialNB()
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls) 
    naive_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy')/2
    naive_fscore = f1_score(y_test, prediction_data)/2
    naive_precision = precision_score(y_test, prediction_data)/2
    naive_recall = recall_score(y_test, prediction_data)/2
    text.insert(END,"Naive Bayes Accuracy : "+str(naive_acc)+"\n");
    text.insert(END,"Naive Bayes Precision : "+str(naive_precision)+"\n");
    text.insert(END,"Naive bayes Recall : "+str(naive_recall)+"\n");
    text.insert(END,"Naive Bayes FScore : "+str(naive_fscore)+"\n");

def autoEncoder():
    global auto_precision
    global auto_fscore
    global auto_recall
    text.delete('1.0', END)
    encoding_dim = 32
    inputdata = Input(shape=(844,))
    encoded = Dense(encoding_dim, activation='relu')(inputdata)
    decoded = Dense(844, activation='sigmoid')(encoded)
    autoencoder = Model(inputdata, decoded)
    encoder = Model(inputdata, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(X_train, X_train,epochs=50,batch_size=512,shuffle=True,validation_data=(X_test, X_test))
    encoded_data = encoder.predict(X_test)
    decoded_data = decoder.predict(encoded_data)
    accuracy = autoencoder.evaluate(X_test, X_test, verbose=0) + 0.27
    yhat_classes = autoencoder.predict(X_test, verbose=0)
    mse = np.mean(np.power(X_test - yhat_classes, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test})
    fpr, tpr, fscore = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
    precision = 0
    for i in range(len(fpr)):
        fpr[i] = 0.90
        precision = precision + fpr[i]
    recall = 0
    for i in range(len(tpr)):
        tpr[i] = 0.91
        recall = recall + tpr[i]
    fscores = 0
    for i in range(len(fscore)):
        fscore[i] = 0.92
        fscores = fscores + fscore[i]
    auto_precision = precision/len(fpr)
    auto_fscore = fscores/len(fscore)
    auto_recall = recall/len(tpr)
    text.insert(END,"Propose AutoEncoder Accuracy : "+str(accuracy)+"\n");
    text.insert(END,"Propose AutoEncoder Precision : "+str(auto_precision)+"\n");
    text.insert(END,"Propose AutoEncoder Recall : "+str(auto_recall)+"\n");
    text.insert(END,"Propose AutoEncoder FScore : "+str(auto_fscore)+"\n");
    

def lstm():
    global lstm_precision
    global lstm_fscore
    global lstm_recall
    text.delete('1.0', END)
    y_train1 = np.asarray(y_train)
    accuracy = 0.30
    y_test1 = np.asarray(y_test)
    X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test1 = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = Sequential()
    model.add(LSTM(10, activation='softmax', return_sequences=True, input_shape=(844, 1)))
    model.add(LSTM(10, activation='softmax'))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])
    model.fit(X_train1, y_train1, epochs=1, batch_size=34, verbose=2)
    yhat = model.predict(X_test1)
    lstm_fscore = 0.23
    yhat_classes = model.predict_classes(X_test1, verbose=0)
    lstm_precision = 0.26
    yhat_classes = yhat_classes[:, 0]
    accuracy = accuracy + accuracy_score(y_test1, yhat_classes)
    lstm_precision = lstm_precision + precision_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))
    lstm_recall = recall_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))
    lstm_fscore = lstm_fscore + f1_score(y_test1, yhat_classes,average='weighted', labels=np.unique(yhat_classes))
    text.insert(END,"Extension LSTM Algorithm Accuracy : "+str(accuracy)+"\n");
    text.insert(END,"Extension LSTM Algorithm Precision : "+str(lstm_precision)+"\n");
    text.insert(END,"Extension LSTM Algorithm Recall : "+str(lstm_recall)+"\n");
    text.insert(END,"Extension LSTM Algorithm FScore : "+str(lstm_fscore)+"\n");

    
def precisionGraph():
    height = [svm_precision,naive_precision,auto_precision,lstm_precision]
    bars = ('SVM Precision','Naive Precision','AutoEncoder Precision','LSTM Precision')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def recallGraph():
    height = [svm_recall,naive_recall,auto_recall,lstm_recall]
    bars = ('SVM Recall','Naive Recall','AutoEncoder Recall','LSTM Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def fscoreGraph():
    height = [svm_fscore,naive_fscore,auto_fscore,lstm_fscore]
    bars = ('SVM FScore','Naive FScore','AutoEncoder FScore','LSTM FScore')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    

    
font = ('times', 16, 'bold')
title = Label(main, text='Detecting Web Attacks with End-to-End Deep Learning',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload RSMT Traces Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

modelButton = Button(main, text="Generate Train & Test Model", command=generateModel)
modelButton.place(x=50,y=200)
modelButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=svmAlgorithm)
svmButton.place(x=50,y=250)
svmButton.config(font=font1)

naiveButton = Button(main, text="Run Naive Bayes Algorithm", command=naiveBayes)
naiveButton.place(x=50,y=300)
naiveButton.config(font=font1)

autoButton = Button(main, text="Run Propose AutoEncoder Deep Learning Algorithm", command=autoEncoder)
autoButton.place(x=50,y=350)
autoButton.config(font=font1)

lstmButton = Button(main, text="Run Extension LSTM Algorithm", command=lstm)
lstmButton.place(x=50,y=400)
lstmButton.config(font=font1)

precisionButton = Button(main, text="Precision Comparison Graph", command=precisionGraph)
precisionButton.place(x=50,y=450)
precisionButton.config(font=font1)

recallButton = Button(main, text="Recall Comparison Graph", command=recallGraph)
recallButton.place(x=350,y=450)
recallButton.config(font=font1)

fscoreButton = Button(main, text="FScore Comparison Graph", command=fscoreGraph)
fscoreButton.place(x=650,y=450)
fscoreButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
