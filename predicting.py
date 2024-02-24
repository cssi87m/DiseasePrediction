'''Creating a function that can take symptoms as input and generate predictions for disease '''
from preprocess import *
from testing import svc, gaussian, random_forest, mode

symptomns=X.columns.values

# Creating a symptomns index dictionary to encode the input symptomns into numerical forms 
symptomn_index={}
for index, value in enumerate(symptomns):
    symptomn=' '.join([i.capitalize() for i in value.split('_')])
    symptomn_index[symptomn]=index

data_dict={
    'symptomn_index': symptomn_index,
    'prediction_classes':label_encoder.classes_
}
# Define a function 
# input: string containing symptomns separated by commas (,)
# output: generated predictions by models 
def predict_disease(symptomns):
    symptomns=symptomns.split(',')
    # creating input data for the model:
    input_data=[0]*len(data_dict['symptomn_index'])
    for symptomn in symptomns:
        index = data_dict['symptomn_index'][symptomn]
        input_data[index]=1
    # reshaping the input data and converting it into 
    # suitable format for model prediction
    input_data=np.array(input_data).reshape(1, -1)
    # generating individual output
    rf_prediction =data_dict['prediction_classes'][random_forest.predict(input_data)[0]]
    nb_prediction =data_dict['prediction_classes'][gaussian.predict(input_data)[0]]
    svc_prediction =data_dict['prediction_classes'][svc.predict(input_data)[0]]
    # making final prediction by taking mode of all predictions
    final_prediction=mode((rf_prediction, nb_prediction, svc_prediction))[0][0]
    prediction={
        'rf_prediction': rf_prediction,
        'nb_prediction': nb_prediction,
        'svc_prediction': svc_prediction,
        'final_prediction': final_prediction
    }
    return prediction
print(predict_disease("Itching,Skin Rash,Nodal Skin Eruptions"))