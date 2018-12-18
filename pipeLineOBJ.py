import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class pipe():

    def __init__(self,class_indices, dir='dataset',s=80, max_imgs=20, nbins= 32, cellSize=8, c = 46.416, gamma = 0.04,kernelType='rbf', norm=True):
        self.kernelType=kernelType
        self.class_indices = class_indices
        self.norm = norm
        self.dir = dir
        self.s = s
        self.max_imgs = max_imgs
        self.nbins = nbins
        self.cellSize = cellSize
        self.c = c
        self.gamma = gamma
        self.train, self.test, self.test_dirs = self.upload_data(self.class_indices,self.s)
        self.x_train, self.x_test, self.y_train, self.y_test = self.preperData(self.train, self.test, nbins=self.nbins,
                                                           cellSize=self.cellSize)  # calc hog histograms

    def get_classes_by_ind(self,class_indices):
        """
        :param dir: dataset directory
        :param class_indices: input classes
        :return: list of classes names string
        """
        classes = os.listdir(self.dir)
        classes_choosen = []
        for ind in class_indices:
            classes_choosen.append(classes[ind - 1])
        return classes_choosen

    def upload_data(self,class_indices, s):
        """
        uplaod and resize data by class indices, return x data and labels
        :param class_indices: class_indices
        :param base_dir:
        :param size_col:
        :param size_row:
        :return: lists of train, test, test_dirs
        """
        #initialize
        train_array = []
        test_array = []
        classes = os.listdir(self.dir)
        train_labels_num = []
        test_labels_num = []
        test_dirs = []

        for ind in class_indices:       #run on folders
            class_ = classes[ind - 1]
            class_dir = os.path.join(self.dir, class_)
            img_name_list = os.listdir(class_dir)
            img_per_label = 0
            for img_name in img_name_list:     #run on images- folder contents
                if img_per_label < self.max_imgs:   #limiting train images per class
                    img_dir = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_dir, 0)
                    img = cv2.resize(img, (s, s))
                    #normalization
                    if self.norm:
                        normalizedImg = np.zeros((s, s))
                        image_resized = cv2.normalize(img, normalizedImg, 0, 127, cv2.NORM_MINMAX)
                        img = image_resized.copy()
                    train_array.append(img)
                    train_labels_num.append(ind)
                    img_per_label += 1
                elif img_per_label < self.max_imgs*2: #limiting test images per class
                    img_dir = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_dir, 0)
                    img = cv2.resize(img, (s, s))
                    # normalization
                    if self.norm:
                        normalizedImg = np.zeros((s, s))
                        image_resized = cv2.normalize(img, normalizedImg, 0, 127, cv2.NORM_MINMAX)
                        img = image_resized.copy()
                    test_array.append(img)
                    test_labels_num.append(ind)
                    img_per_label += 1
                    test_dirs.append(img_dir)
        train = [train_array, train_labels_num]
        test = [test_array, test_labels_num]
        return train, test, test_dirs

    def calc_hogs(self,image_array,nbins = 32, cellSize = 8):
        """
        calculate image hog transform
        :param image_array: input image
        :return: histogram vector
        """
        histograms = []
        for image in image_array:
            winSize = (64, 64)
            blockSize = (16, 16)
            blockStride = (8, 8)
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = 2.0000000000000001e-01
            gammaCorrection = 0
            nlevels = 64
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, (cellSize,cellSize), nbins, derivAperture, winSigma,
                                    histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
            winStride = (8, 8)
            padding = (8, 8)
            locations = ((10, 20),)
            hist = hog.compute(image, winStride, padding, locations)
            histograms.append(np.squeeze(hist))
        return histograms

    def preperData(self,train,test, nbins, cellSize, toThuffle=False):
        """
        preper data with hogs histograms, and shuffle the data
        :return:
        """
        x_train = self.calc_hogs(train[0], nbins=nbins, cellSize=cellSize) #calculation of hogs
        y_train = train[1]
        x_test = self.calc_hogs(test[0], nbins=nbins, cellSize=cellSize)
        y_test = test[1]
        if toThuffle: #data shuffling
            x_train, y_train = shuffle(x_train, train[1], random_state=0)
            x_test, y_test = shuffle(x_test, test[1], random_state=0)

        return x_train, x_test, y_train, y_test

    def get_best_S(self,gamma, C,  nbins, cellSize):
        """
        tuning image size

        :param gamma: gamma of svm
        :param C: cost of svm
        :param nbins: n bins of hog
        :param cellSize: cell size of hog
        :return: optimized S
        """
        class_indices = range(1, 11)    #load 1-10 classes
        s_vec = range(76,140,2)
        # training
        acc_vec = []
        ind_s_vec = []
        i=1
        for s in s_vec:
            train, test, _ = self.upload_data(class_indices, s=s)
            x_train, x_test, y_train, y_test = self.preperData(train, test, nbins=nbins, cellSize=cellSize)  # calc hog histograms
            n_class_SVM = self.n_class_SVM_train(x_train, y_train, gamma=gamma, C=C)  #train on 10 one vs all svm non linear
            preds, _ = self.n_class_SVM_predict(n_class_SVM, x_test, y_test)
            acc = accuracy_score(y_test, preds)
            print('step:%d from %d'%(i, len(s_vec)))
            print('s Size:', s,'accuracy:', acc)
            acc_vec.append(acc)
            ind_s_vec.append(s)
            i+=1
        max_acc = max(acc_vec)
        print('max_acc:', max_acc)
        ind_max = acc_vec.index(max_acc)
        s_max_from_vec = ind_s_vec[ind_max]
        print('s_max_from_vec', s_max_from_vec)
        #1-D visualization
        s_acc = np.stack((np.squeeze(ind_s_vec), np.squeeze(acc_vec)))
        s_acc = np.transpose(s_acc)
        self.visualization(y=s_acc[:,1], x=s_acc[:,0], x_title='size', y_title='Accuracy')
        plt.close()
        #plt.show()
        return max_acc, s_max_from_vec

    def get_best_hog(self,gamma, C, s):
        """
        tuning parameters for hog
        :param gamma: gamma of svm
        :param C: cost of svm
        :param s: image size
        :return: prarams
        """
        class_indices = range(1, 11)    #load 1-10 classes
        train, test, _ = self.upload_data(class_indices, s=s)
        # training
        nbins_vec = [4,8,16,32,64]
        cellSize_vec = [4,8,16]
        acc_vec = []
        ind_nbins_vec = []
        ind_cellSize_vec = []
        i=1
        for cellSize in cellSize_vec:
            for nbins in nbins_vec:
                x_train, x_test, y_train, y_test = self.preperData(train, test, nbins=nbins, cellSize=cellSize)  # calc hog histograms
                n_class_SVM = self.n_class_SVM_train(x_train, y_train, gamma=gamma, C=C)  #train on 10 one vs all svm non linear
                preds, _ = self.n_class_SVM_predict(n_class_SVM, x_test, y_test)
                acc = accuracy_score(y_test, preds)
                print('step:%d from %d'%(i, len(nbins_vec)*len(cellSize_vec)))
                print('cellSize:', cellSize, 'nbins:', nbins,'accuracy:', acc)
                acc_vec.append(acc)
                ind_nbins_vec.append(nbins)
                ind_cellSize_vec.append(cellSize)
                i+=1
        max_acc = max(acc_vec)
        print('max_acc:', max_acc)
        ind_max = acc_vec.index(max_acc)
        nbins_max_from_vec = ind_nbins_vec[ind_max]
        cellSize_max_from_vec = ind_cellSize_vec[ind_max]


        print('nbins_max_from_vec', nbins_max_from_vec)
        print('cellSize_max_from_vec', cellSize_max_from_vec)

        #2-D visualization
        nbins_cellSize_acc = np.stack((np.squeeze(ind_cellSize_vec), np.squeeze(ind_nbins_vec), np.squeeze(acc_vec)))
        nbins_cellSize_acc = np.transpose(nbins_cellSize_acc)

        indx_nbins_by_cellSize = np.where(nbins_cellSize_acc[:, 0] == cellSize_max_from_vec)
        nbins_by_cellSize = np.squeeze(nbins_cellSize_acc[indx_nbins_by_cellSize, :])

        indx_cellSize_by_nbins = np.where(nbins_cellSize_acc[:, 1] == nbins_max_from_vec)
        cellSize_by_nbins = np.squeeze(nbins_cellSize_acc[indx_cellSize_by_nbins, :])

        self.visualization(y=cellSize_by_nbins[:,2], x=cellSize_by_nbins[:,0], x_title='cellSize', y_title='Accuracy')
        self.visualization(y=nbins_by_cellSize[:,2], x=nbins_by_cellSize[:,1], x_title='nbins', y_title='Accuracy')

        #3-D visualization
        fig = plt.figure()
        ax = Axes3D(fig)  # <-- Note the difference from your original code...

        ax.scatter3D(ind_nbins_vec, ind_cellSize_vec, acc_vec, cmap='Greens')

        ax.set_xlabel('nbins')
        ax.set_ylabel('cellSize')
        ax.set_zlabel('acc')

        plt.title('Accuracy vs cellSize and nbins')
        plt.legend()
        plt.savefig('Accuracy vs cellSize and nbins.jpg')

        plt.close()
        #plt.show()

        return max_acc, nbins_max_from_vec,cellSize_max_from_vec

    def get_best_params_SVM(self,  nbins, cellSize, s):
        """
        tuning parameters c, gamma for brf svm, one vs all
        :param s: image size
        :param nbins: n bins of hog
        :param cellSize: cell size of hog
        :return: optimized params
        """
        class_indices = range(1, 11)    #load 1-10 classes
        train, test, _ = self.upload_data(class_indices, s=s)
        x_train, x_test, y_train, y_test = self.preperData(train, test, nbins=nbins,
                                                           cellSize=cellSize)  # calc hog histograms
        # training
        gamma_vec = np.arange(0.01, 0.1, 0.01)
        #c_vec = range(1, 6, 1)
        c_vec = np.logspace(-1,7,7).tolist()
        acc_vec = []
        ind_c_v = []
        ind_g_v = []
        i=1
        for gamma in gamma_vec:
            for c in c_vec:
                gamma = round(gamma, 4)
                n_class_SVM = self.n_class_SVM_train(x_train, y_train, gamma=gamma, C=c)  #train on 10 one vs all svm non linear
                preds, _ = self.n_class_SVM_predict(n_class_SVM, x_test, y_test)
                acc = accuracy_score(y_test, preds)
                print('step:%d from %d'%(i, gamma_vec.shape[0]*len(c_vec)))
                print('C:', c, 'gamma:', gamma,'accuracy:', acc)
                acc_vec.append(acc)
                ind_c_v.append(c)
                ind_g_v.append(gamma)
                i+=1
        max_acc = max(acc_vec)
        print('max_acc:', max_acc)
        ind_max = acc_vec.index(max_acc)
        c_max_from_vec = ind_c_v[ind_max]
        g_max_from_vec = ind_g_v[ind_max]

        print('c_max_from_vec', c_max_from_vec)
        print('g_max_from_vec', g_max_from_vec)

        #2-D visualization
        gamma_C_acc = np.stack((np.squeeze(ind_g_v), np.squeeze(ind_c_v), np.squeeze(acc_vec)))
        gamma_C_acc = np.transpose(gamma_C_acc)

        indx_C_by_gamma = np.where(gamma_C_acc[:, 0] == g_max_from_vec)
        C_by_gamma = np.squeeze(gamma_C_acc[indx_C_by_gamma, :])

        indx_gamma_by_C = np.where(gamma_C_acc[:, 1] == c_max_from_vec)
        gamma_by_C = np.squeeze(gamma_C_acc[indx_gamma_by_C, :])

        self.visualization(y=gamma_by_C[:,2], x=gamma_by_C[:,0], x_title='Gamma', y_title='Accuracy')
        self.visualization(y=C_by_gamma[:,2], x=C_by_gamma[:,1], x_title='C', y_title='Accuracy')

        #3-D visualization
        fig = plt.figure()
        ax = Axes3D(fig)  # <-- Note the difference from your original code...
        ax.scatter3D(ind_g_v, ind_c_v, acc_vec, cmap='Greens')
        ax.set_xlabel('gamma')
        ax.set_ylabel('C')
        ax.set_zlabel('acc')
        plt.title('Accuracy vs C and Gamma')
        plt.legend()
        plt.savefig('Accuracy vs C and Gamma.jpg')
        plt.close()
        #plt.show()
        return max_acc, c_max_from_vec,g_max_from_vec

    def tunning_params(self):
        """
        maximize acc by tuning nbins, cell size, image size, c, gamma
        iterations method
        :return: optimized nbins, cellSize, c,  gamma, s
        """
        nbins = 16
        cellSize = 4
        converging = False
        max_acc = 0
        s = 80
        i = 0
        while not converging:
            max_acc1, c_max, g_max = self.get_best_params_SVM(nbins=nbins, cellSize=cellSize,s=s)
            if max_acc1 > max_acc:
                c = c_max
                gamma = g_max
                max_acc = max_acc1
                i = 0
            else:
                print('break 1')
                i += 1
                if self.find_break(i):
                    break
            max_acc2, nbins_max, cellSize_max = self.get_best_hog(gamma=gamma, C=c,s=s)
            if max_acc2 > max_acc:
                nbins = nbins_max
                cellSize = cellSize_max
                max_acc = max_acc2
                i = 0
            else:
                print('break 2')
                i += 1
                if self.find_break(i):
                    break
            max_acc3, s_max = self.get_best_S(gamma=gamma, C=c, nbins=nbins, cellSize=cellSize)
            if max_acc3 > max_acc:
                s=s_max
                max_acc = max_acc3
                i = 0
            else:
                print('break 3')
                i += 1
                if self.find_break(i):
                    break

        print('nbins', nbins, '\n', 'cellSize', cellSize, "\n" + 'c', c, "\n", 'gamma', gamma, '\n', 'max_acc', max_acc,  '\n', 's', s)
        #update object params
        self.nbins=nbins
        self.cellSize=cellSize
        self.c=c
        self.gamma=gamma
        self.s=s
        self.train, self.test, self.test_dirs = self.upload_data(self.class_indices,self.s)
        self.x_train, self.x_test, self.y_train, self.y_test = self.preperData(self.train, self.test, nbins=self.nbins,
                                                                               cellSize=self.cellSize)  # calc hog histograms
        return nbins, cellSize, c,  gamma, s

    def labeling_oneVSall(self,labels, one):
        """
        lebaling 1 and -1 for one vs all training
        :param labels: labels vector
        :param one: choosen label
        :return: new 1,-1 labels vector
        """
        new_labels = []
        for label in labels:
            if label == one:
                new_labels.append(1)
            else:
                new_labels.append(-1)
        return new_labels

    def n_class_SVM_train(self,x_train, y_train,gamma, C):
        """
        loops through the M classes and trains M binary classifiers in one-versus-all method
        :param x_train:
        :param y_train:
        :param gamma:
        :param C:
        :return: vector of 10 binaries classifiers
        """
        classes = set(y_train)
        clfs = []
        for class_ in classes:
            y_one = self.labeling_oneVSall(y_train, class_)
            clf = svm.SVC(gamma=gamma, C=C, kernel=self.kernelType, degree=3)
            #clf = svm.LinearSVC()
            clf.fit(x_train, y_one)
            clfs.append([clf,class_])
        self.n_class_SVM = clfs
        return clfs

    def n_class_SVM_predict(self,MClassSVM, x_test, y_test):
        """
        make cross binaries prediction and return the preds classes and the pred confideces
        :param MClassSVM: N svm keranel classifiers
        :param x_test:
        :param y_test:
        :return: predictions, confideces
        """
        distances = []
        classes = []
        predictions = []
        confidences = []
        for svm in MClassSVM:
            svm_one = svm[0]
            svm_class = svm[1]
            dist = svm_one.decision_function(x_test)
            distances.append(dist)
            classes.append(svm_class)
        distances_matrix = np.asanyarray(distances)
        distances_matrix = np.transpose(distances_matrix)
        max_dist_indx = np.argmax(distances_matrix, axis=1)

        #cofidence calculation
        j=0
        for i in max_dist_indx:
            pred = classes[i]
            pred_dist = distances_matrix[j,i]
            currect_y = y_test[j]
            class_indx = classes.index(currect_y)
            currect_dist = distances_matrix[j,class_indx]
            if currect_dist == pred_dist:   #if that is currect predictio than use the seconed max distance
                temp = np.sort(distances_matrix[j,:])
                pred_sec_dist = temp[temp.size-2]
                confid = currect_dist - pred_sec_dist
            else:
                confid = currect_dist - pred_dist
            confidences.append(confid)
            predictions.append(pred)
            j += 1
        #save results to xlsx
        my_df = pd.DataFrame((distances_matrix),columns=self.get_classes_by_ind(classes))
        my_df.to_csv('results_data.csv', index=True, header=True)
        self.predictions = predictions
        self.confidences = confidences
        return predictions, confidences

    def get_failed_preds(self):
        """
        get summaries of false predictions
        first- creating false predictions matrix
        second- save top two largest errors images per class
        third- save and review image grid
        :param y_test:
        :param preds:
        :param img_dirs:
        :param confidences:
        :return:
        """
        y_test = self.y_test
        confidences = self.confidences
        preds = self.predictions
        i=0
        faileds = []
        failed_matrix = []
        #creating false prediction matrix
        for y in y_test:
            if y != preds[i]:
                true_val = self.get_classes_by_ind([y])[0]
                prediction = self.get_classes_by_ind([preds[i]])[0]
                faileds.append([i,true_val,prediction,confidences[i]])
                failed_matrix.append([i,y,preds[i],confidences[i]])
            i += 1
        #find classes with full prediction success, and create titles from them
        y_failed_preds=np.asanyarray(failed_matrix,dtype=np.uint8)[:,1].tolist()
        fully_currected_class = [str(self.get_classes_by_ind([item])[0])+' All is currect' for item in set(y_test) if item not in set(y_failed_preds)]
        #save results to xlsx
        my_df = pd.DataFrame(faileds,columns=['index','true value','prediction','confidance'])
        my_df.to_csv('failed_preds.csv', index=False, header=True)
        images, titles = self.SaveGet_top_two_errors(failed_matrix,self.test_dirs) #Get 2 top errors images
        titles = titles + fully_currected_class
        self.save_images_as_grid(images,titles)  #save grid

    def SaveGet_top_two_errors(self,failed_matrix,img_dirs,):
        """
        get the false predictions matrix and save top two errors per class
        :param failed_matrix:
        :return: flase prediction images, titles
        """
        failed_dir='top 2 failed preds by class/'
        if not os.path.exists(failed_dir):
            os.makedirs(failed_dir)
        else:
            for root, dirs, files in os.walk(failed_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        images = []
        titles = []
        failed_matrix = np.asarray(failed_matrix)
        failed_matrix_classes = failed_matrix[:, 1]
        y_set = failed_matrix_classes.tolist()
        y_Fale_set = set(y_set)
        for class_ in y_Fale_set:   #run on every class on false preds
            #creating matrix for each clas
            indx = np.where(failed_matrix_classes==class_)
            specific = failed_matrix[indx,:]
            specific_dist = specific[:,:,3]
            specific_dist = np.sort(specific_dist)
            specific_dist = np.squeeze(specific_dist)
            #if specific_dist.size!=0:
            if specific_dist.size == 1:     #if there is only one false prediction in the class
                specific = np.squeeze(specific)
                one_min = specific[0]
                m = int(one_min)
                false_class_indx = int(specific[2])
                true_class_indx = int(specific[1])
                true_val = self.get_classes_by_ind([true_class_indx])[0]
                prediction = self.get_classes_by_ind([false_class_indx])[0]
                img = cv2.imread(img_dirs[m])                      #read original image
                d = np.round(specific[3], decimals=3, out=None)
                title = true_val + ' failed ' + prediction + ' ' + str(d)
                cv2.imwrite(failed_dir + title + '.jpg', img)
                images.append(img)
                titles.append(title)
            else:           #if there is more then one false prediction in the class
                indx_top2 = np.where(specific[:, :, 3] <= specific_dist[1])[1]
                specific = np.squeeze(specific)
                two_max_err = specific[indx_top2, 0]
                z = 0
                for im in two_max_err:
                    ind_min = indx_top2[z]
                    m = int(im)
                    true_class_indx = int(specific[ind_min, 1])
                    false_class_indx = int(specific[ind_min, 2])
                    true_val = self.get_classes_by_ind([true_class_indx])[0]
                    prediction = self.get_classes_by_ind([false_class_indx])[0]
                    img = cv2.imread(img_dirs[m])               #read original image
                    d = np.round(specific[ind_min, 3], decimals=3, out=None)
                    title = true_val + ' failed ' + prediction + ' ' + str(d)
                    cv2.imwrite(failed_dir + title + '.jpg', img)
                    images.append(img)
                    titles.append(title)
                    z += 1
        return images, titles

    def save_images_as_grid(self,images,titles):
        """
        save top 2 errors per class in large grid
        :param images:
        :param titles:
        :return:
        """
        r=4
        c=5
        fig, axs = plt.subplots(r, c)
        fig.set_size_inches(18.5, 10.5)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if cnt < len(images):
                    img = images[cnt]
                    img_resized = cv2.resize(img, (400, 400))
                    axs[i, j].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
                    axs[i, j].set_title(titles[cnt],fontsize=10)
                elif cnt < len(titles):
                    img_resized = np.zeros([400, 400, 3], dtype=np.uint8)
                    img_resized.fill(200)
                    axs[i, j].imshow(img_resized)
                    axs[i, j].set_title(titles[cnt],va ='bottom')
                axs[i, j].axis('off')
                cnt += 1
        fig.suptitle('Top 2 errors for each class',fontsize=30)
        fig.savefig("errors grid.png")

    def visualization(self,y,x, x_title, y_title):
        """
        2-D plotting
        :param y:
        :param x:
        :param x_title:
        :param y_title:
        :return:
        """
        plt.plot(x,y, label='prediction values')
        plt.ylabel(y_title)
        plt.xlabel(x_title)
        plt.title(x_title + ' VS ' + y_title)
        plt.legend()
        plt.savefig(x_title + ' VS ' + y_title + '.jpg')
        plt.close()
        #plt.show()

    def find_break(self,i):
        flag=False
        if i==2:
            flag=True
        return flag

    def plot_confusion_matrix(self,cm, classes, normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure(figsize=(8, 6))
        np.set_printoptions(precision=2)
        if normalize:
            div = cm.sum(axis=1)[:, np.newaxis]
            div = np.where(div != 0, div, 0.1)
            cm = cm.astype('float') / div
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        # plt.savefig("conf.png")


    #####only for the linear method#########

    def get_best_hog_linear(self,dir = 'dataset', C=3, s=80):
        """
        tuning parameters for hog
        :param dir:
        :return: prarams
        """
        class_indices = range(1, 11)    #load 1-10 classes
        train, test, _ = self.upload_data(class_indices, s=s)


        # training
        nbins_vec = [4,8,16,32,64]
        cellSize_vec = [4,8,16]

        acc_vec = []
        ind_nbins_vec = []
        ind_cellSize_vec = []
        i=1
        for cellSize in cellSize_vec:
            for nbins in nbins_vec:
                x_train, x_test, y_train, y_test = self.preperData(train, test, nbins=nbins, cellSize=cellSize)  # calc hog histograms
                clfLinear = svm.LinearSVC(multi_class='ovr', C=C)
                clfLinear.fit(x_train, y_train)
                predsLinear = clfLinear.predict(x_test)
                acc = accuracy_score(y_test, predsLinear)
                print('step:%d from %d'%(i, len(nbins_vec)*len(cellSize_vec)))
                print('cellSize:', cellSize, 'nbins:', nbins,'accuracy:', acc)
                acc_vec.append(acc)
                ind_nbins_vec.append(nbins)
                ind_cellSize_vec.append(cellSize)
                i+=1
        max_acc = max(acc_vec)
        print('max_acc:', max_acc)
        ind_max = acc_vec.index(max_acc)
        nbins_max_from_vec = ind_nbins_vec[ind_max]
        cellSize_max_from_vec = ind_cellSize_vec[ind_max]


        print('nbins_max_from_vec', nbins_max_from_vec)
        print('cellSize_max_from_vec', cellSize_max_from_vec)

        #1-D visualization
        nbins_cellSize_acc = np.stack((np.squeeze(ind_cellSize_vec), np.squeeze(ind_nbins_vec), np.squeeze(acc_vec)))
        nbins_cellSize_acc = np.transpose(nbins_cellSize_acc)

        indx_nbins_by_cellSize = np.where(nbins_cellSize_acc[:, 0] == cellSize_max_from_vec)
        nbins_by_cellSize = np.squeeze(nbins_cellSize_acc[indx_nbins_by_cellSize, :])

        indx_cellSize_by_nbins = np.where(nbins_cellSize_acc[:, 1] == nbins_max_from_vec)
        cellSize_by_nbins = np.squeeze(nbins_cellSize_acc[indx_cellSize_by_nbins, :])

        self.visualization(y=cellSize_by_nbins[:,2], x=cellSize_by_nbins[:,0], x_title='cellSize', y_title='Accuracy')
        self.visualization(y=nbins_by_cellSize[:,2], x=nbins_by_cellSize[:,1], x_title='nbins', y_title='Accuracy')

        #2-D visualization
        fig = plt.figure()
        ax = Axes3D(fig)  # <-- Note the difference from your original code...

        ax.scatter3D(ind_nbins_vec, ind_cellSize_vec, acc_vec, cmap='Greens')

        ax.set_xlabel('nbins')
        ax.set_ylabel('cellSize')
        ax.set_zlabel('acc')

        plt.title('Accuracy vs cellSize and nbins')
        plt.legend()
        plt.savefig('Accuracy vs cellSize and nbins.jpg')

        plt.close()
        #plt.show()

        return max_acc, nbins_max_from_vec,cellSize_max_from_vec


    def get_best_S_linear(self,dir='dataset', C=3, nbins=16, cellSize=4):
        """
        tuning parameters for hog
        :param dir:
        :return: prarams
        """
        class_indices = range(1, 11)  # load 1-10 classes
        s_vec = range(76, 180, 2)

        # training

        acc_vec = []
        ind_s_vec = []
        i = 1
        for s in s_vec:
            train, test, _ = self.upload_data(class_indices, s=s)
            x_train, x_test, y_train, y_test = self.preperData(train, test, nbins=nbins, cellSize=cellSize)

            clfLinear = svm.LinearSVC(multi_class='ovr', C=C)
            clfLinear.fit(x_train, y_train)
            predsLinear = clfLinear.predict(x_test)
            acc = accuracy_score(y_test, predsLinear)

            print('step:%d from %d' % (i, len(s_vec)))
            print('s Size:', s, 'accuracy:', acc)
            acc_vec.append(acc)
            ind_s_vec.append(s)
            i += 1
        max_acc = max(acc_vec)
        print('max_acc:', max_acc)
        ind_max = acc_vec.index(max_acc)
        s_max_from_vec = ind_s_vec[ind_max]

        print('s_max_from_vec', s_max_from_vec)

        # 1-D visualization
        s_acc = np.stack((np.squeeze(ind_s_vec), np.squeeze(acc_vec)))
        s_acc = np.transpose(s_acc)

        self.visualization(y=s_acc[:, 1], x=s_acc[:, 0], x_title='size', y_title='Accuracy')

        # plt.show()

        return max_acc, s_max_from_vec

    def get_best_params_SVM_linear(self, nbins=16, cellSize=4, s=80):
        """
        tuning parameters c, gamma for brf svm, one vs all
        :param dir:
        :return: prarams
        """
        class_indices = range(1, 11)    #load 1-10 classes
        train, test, _ = self.upload_data(class_indices, s=s)
        x_train, x_test, y_train, y_test = self.preperData(train, test, nbins=nbins,
                                                           cellSize=cellSize)  # calc hog histograms
        # training
        c_vec = np.arange(0.5, 10, 0.5)
        acc_vec = []
        ind_c_v = []
        i=1
        for c in c_vec:
            clfLinear = svm.LinearSVC(multi_class='ovr', C=c)
            clfLinear.fit(x_train, y_train)
            predsLinear = clfLinear.predict(x_test)
            acc = accuracy_score(y_test, predsLinear)

            print('step:%d from %d'%(i,c_vec.shape[0]))
            print('C:', c,'accuracy:', acc)
            acc_vec.append(acc)
            ind_c_v.append(c)
            i+=1
        max_acc = max(acc_vec)
        print('max_acc:', max_acc)
        ind_max = acc_vec.index(max_acc)
        c_max_from_vec = ind_c_v[ind_max]

        print('c_max_from_vec', c_max_from_vec)

        #1-D visualization
        ind_c_v=np.squeeze(ind_c_v)
        acc_vec=np.squeeze(acc_vec)
        C_acc = np.stack((ind_c_v, acc_vec))
        C_acc = np.transpose(C_acc)

        C_by_acc = np.squeeze(C_acc)


        self.visualization(y=C_by_acc[:,1], x=C_by_acc[:,0], x_title='C', y_title='Accuracy')


        plt.close()
        #plt.show()

        return max_acc, c_max_from_vec

    def tunning_params_linear(self):
        nbins = 16
        cellSize = 4
        converging = False
        max_acc = 0
        s = 80
        i = 0
        while not converging:

            max_acc1, c_max = self.get_best_params_SVM_linear(nbins=nbins, cellSize=cellSize, s=s)
            if max_acc1 > max_acc:
                c = c_max
                max_acc = max_acc1
                i = 0
            else:
                print('break 1')
                i += 1
                if self.find_break(i):
                    break

            max_acc2, nbins_max, cellSize_max = self.get_best_hog_linear(dir='dataset', C=c, s=s)
            if max_acc2 > max_acc:
                nbins = nbins_max
                cellSize = cellSize_max
                max_acc = max_acc2
                i = 0
            else:
                print('break 2')
                i += 1
                if self.find_break(i):
                    break

            max_acc3, s_max = self.get_best_S_linear(dir='dataset', C=c, nbins=nbins, cellSize=cellSize)
            if max_acc3 > max_acc:
                s = s_max
                max_acc = max_acc3
                i = 0
            else:
                print('break 3')
                i += 1
                if self.find_break(i):
                    break

        print('nbins', nbins, '\n', 'cellSize', cellSize, "\n" + 'c', c, '\n', 'max_acc', max_acc,
              '\n', 's', s)
        return nbins, cellSize, c, s

