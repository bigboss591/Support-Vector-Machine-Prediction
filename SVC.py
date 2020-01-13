import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing


#creates a data frame using data in data.csv
flower_data = pd.read_csv('data.csv')

#sets x equal to the values in row 1 - 3
X = flower_data.iloc[:, [1, 2, 3]].values

#sets y equal to the values in row 0
y = flower_data.iloc[:, 0].values

info = dict()
#splits the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Support vector machines (SVMs)  methods are incooperated for classification, regression and outliers detection.
classifier = SVR(kernel = 'rbf',gamma='scale')

#Passing the train set to the fit method
classifier.fit(X_train, y_train)

#Evaluate a score by cross-validation
scores =  cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1)

#Missing values can be replaced by the mean
scores.mean()

#Compute the standard deviation of the given data
scores.std()

lab_enc = preprocessing.LabelEncoder()
#lab_enc2 -

#displays the scores
y_test_predicted = classifier.predict(X_test)
info['accuracy'] = min(scores)
info['acuracy_test'] = mean_squared_error(y_test ,y_test_predicted)
info['accuracy_fold'] =scores

#cutoff = 0.7                              # decide on a cutoff limit
#y_pred_classes = classifier.zeros_like(y_pred)    # initialise a matrix full with zeros
#y_pred_classes[y_pred > cutoff] = 1       # add a 1 if the cutoff was breached

#y_test_classes = classi.zeros_like(y_pred)
#y_test_classes[y_test > cutoff] = 1

#confusion_matrix(y_test_classes, y_pred_classes)

training_yscorestest_encoded = lab_enc.fit_transform(y_test)
training_yscorespre_encoded = lab_enc.fit_transform(y_test_predicted)
info['confusion_matrix'] = confusion_matrix(training_yscorestest_encoded,training_yscorespre_encoded)


print("Accuracy: ",info['accuracy'])
print("Accuracy test: ",info['acuracy_test'])
print("Accuracy fold: ",info['accuracy_fold'])
print("confusion matrix: ",info['confusion_matrix'])
