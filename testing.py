from preprocess import *
from training import *
from scipy.stats import mode 
'''Fitting the model on the whole data'''
svc.fit(X, y)
gaussian.fit(X, y)
random_forest.fit(X, y)
# Reading the test data
test_data=pd.read_csv("D:\\Studying\\Visual Studio 2022\\DiseasePrediction\\archive\\Testing.csv").dropna(axis =1)
# Split testing set
test_X=test_data.iloc[:, :-1]
test_y=test_data.iloc[:, -1]
test_y=label_encoder.fit_transform(test_y)
# Making predictions by taking mode of predictions making by all classifiers 
svc_preds=svc.predict(test_X)
gaussian_preds=gaussian.predict(test_X)
random_forest_preds=random_forest.predict(test_X)
final_preds=[mode([i, j, k])[0][0] for i, j, k in zip(svc_preds, gaussian_preds, random_forest_preds)]
 
# print(f"Accuracy on Test dataset by the combined model\
# : {accuracy_score(test_y, final_preds)*100}")
 
# cf_matrix_final = confusion_matrix(test_y, final_preds)
# plt.figure(figsize=(12,8))
# sns.heatmap(cf_matrix_final, annot = True)
# plt.title("Confusion Matrix for Combined Model on Test Dataset")
# plt.show()