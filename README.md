# Iterative algorithm for SVM classifier
This project is implementation of one vs all SVM kernel based on HOG transform feutures. In additional of an iterative algorithm was developed to find the optimal parameters

As part of the project, different models of SVM (kernel polynomial-based, kernel RBF-based, linear) were compared. An attempt was made to deduce how each parameter contributes to the accuracy of the machine and how normalization of the data affects accuracy.

The program displays the prediction result, a confusen matrix, and a table with the most incorrect classification in each category. In addition, the system maintains the classification and error results for Excel files on the user's computer.

## Normalization of histogram of each image in the dataset was.
![Image description](https://github.com/RanBezen/Iterative_algorithm_SVM_classifier/blob/master/images/norm_cal.PNG)

      
## MODELS
The models we tested are:

- SVM with Radial basis function kernel, one vs rest
- SVM with polynomial kernel, one vs rest
- linear SVM, one vs rest

## Parameters tunning
The function to be maximized has 5 parameters, and consists of three different parts:
-	SVM: C, Gamma
-	Hog transform: cell size, n bins
-	Data: images size

- Part 1 - parameters of the SVM: C and gamma.
  - C: parameter related to the cost function added
  - Gamma: parameter related to kernel sensitivity to changes.
- Part 2 -  parameters of the Hog transform: cell size and n bins.
  - N bins: the distribution areas where the gradients are divided
  - Cell size: The size of the cells to which the image is divided.
- Part 3 - parameter of dataset
  - image size
  
##Iterative algorithm
 	
argmax(f(C,gamma,cellSize,nbins,S))≈
1.argmax⁡(f(C,gamma | cellSize,nbins,S)
2.argmax⁡(f(cellSize,nbins | C,gamma ,S)
3.argmax⁡(f(S | C,gamma ,cellSize,nbins )
    if did not covrge- back to 1.

 In each iteration, two parts are set to constant, and search is made for the free parameters that maximize the target function using a grid search, for example:
 
Parameters initialize
a. constant parameters: n bins, cell size, S. The free parameters to be found are: gamma and c. We searched for gamma and c that maximize the target function using a grid search
b. constant parameters: gamma, c, S. The free parameters to be found are: n bins, cell size. We search for nbins, cell size, maximizing the target function using grid search.
c. constant Parameters: All parameters except S. The free parameters to be found are: S. Search for S that maximizes the target function.
d. If the accuracy does not converge (does not exceed two iterations), go back to step a.

In this project the parameters were adjusted on the data set 1-10 when the training was done on the first 20 images in each class, and the 20 remaining images were generated in the same class.
