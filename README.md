# Project 1 CS-433 course

## Higgs Boson:
This machine learning project predicts collision events of Higgs bosons from a CERN dataset. 

### Preprocessing of the data:
Prior review of the data is needed to get the reevant informations. All the functions gave been implemented in the proj1_helpers.py file. 

The steps are the following:
- Load the data set.
- Remove invalid values and fill in the missing values and correlated features.
- Standardising the data by column.
- Use of machine learning models

### Machine learning models

We implemented 6 different methods of machine learning in the implementations.py file :

- Linear regression with gradient descent with learning rate gamma.
- Linear regression with stochastic gradient descent with learning rate gamma.
- Least squares.
- Ridge regression with hyperparameter lambda.
- Logistic regression with learning rate gamma
- Regularized logistic regression with hyperparameter lambda and learning rate gamma.
The best values for the parameters were determined by cross validation in the project1.ipynb. Of all the methods, we found that the ridge regression one was the most accurate.

## Run the script
You can run the run.ipynb to compute the CVS_method_name.csv that will give the clssification of the data set according to the same of the method. You will need the train dataset and the test dataset outside the folder, or to add the data inside the folder and change the following code. 

```python
yb, input_data, ids = load_csv_data("../train.csv")
```
 This line should be changed into the following line (as well as a similar line to load the test set) in otrder to be run within the same folder.

```python
yb, input_data, ids = load_csv_data("train.csv")
```
### Optimize the results
In order to get the best results, we optimized the hyperparameters and plotted some data to see where we were going. The functions can be found in tuning_plotting.py and are used in the run.ipynb file. 


#### Authors
Florence Crozat, Eva Cramatte, Océane Mauroux