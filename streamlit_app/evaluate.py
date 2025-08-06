import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import viz
import matplotlib.pyplot as plt

def evaluate(model, history, xtest, ytest, labels):    
    score = model.evaluate(xtest,ytest, verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')
    y_pred = model.predict(xtest)
    print('\n', classification_report(np.where(ytest > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())), sep='') 
    Y_pred_classes = np.argmax(y_pred,axis = 1) 
    Y_true = np.argmax(ytest,axis = 1) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    viz.plotKerasLearningCurve()
    plt.show()
    viz.plot_learning_curve(history)
    plt.show()
    viz.plot_confusion_matrix(confusion_mtx, classes = list(labels.values()))
    plt.show()
    return model