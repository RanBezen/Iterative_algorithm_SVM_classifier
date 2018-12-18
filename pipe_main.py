from pipeLineOBJ import pipe
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

if __name__== "__main__":
    class_indices = range(11, 21)
    #class_indices = []
    data_path = 'dataset'
    # create pipr object
    pipe = pipe(class_indices=class_indices,dir=data_path,s=80,nbins= 32, cellSize=8,kernelType='rbf', norm=True)

    # tuning pipe params
    #pipe.tunning_params()

    # train classifier
    pipe.n_class_SVM_train(x_train = pipe.x_train,y_train = pipe.y_train, gamma = pipe.gamma,C = pipe.c)
    pipe.n_class_SVM_predict(pipe.n_class_SVM,x_test=pipe.x_test,y_test=pipe.y_test)
    acc_n_class_SVM = accuracy_score(pipe.y_test, pipe.predictions)
    print('Accurecy of SVM kernel one vs rest:',np.round(acc_n_class_SVM, decimals=5, out=None))

    pipe.get_failed_preds()

    #create confusion matrix
    conf_matrix = confusion_matrix(pipe.y_test, pipe.predictions)
    # Plot non-normalized confusion matrix
    pipe.plot_confusion_matrix(conf_matrix, classes=pipe.get_classes_by_ind(pipe.class_indices), title='Confusion matrix')
    # Plot normalized confusion matrix
    pipe.plot_confusion_matrix(conf_matrix, classes=pipe.get_classes_by_ind(pipe.class_indices), normalize=True, title='Normalized confusion matrix')
    plt.show()


