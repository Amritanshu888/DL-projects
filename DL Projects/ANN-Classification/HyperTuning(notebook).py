## Tab 1
## Determining the optimal number of hidden layers and neurons for an Artificial Neural Network (ANN)

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle

## Tab 2
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)

# Label encode 'Gender'
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

# One-hot encode 'Geography'
onehot_encoder_geo = OneHotEncoder(handle_unknown='ignore')
geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine the encoded data with the main dataset
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

# Define features and target variable
X = data.drop('Exited', axis=1)
y = data['Exited']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the encoders and scaler for later use
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)

with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

## Tab 3
# Define a function to create the model and try different parameters(KerasClassifier)

def create_model(neurons=32,layers=1):
    model = Sequential()
    model.add(Dense(neurons,activation='relu',input_shape=(X_train.shape[1],)))

    for _ in range(layers-1):
        model.add(Dense(neurons,activation='relu'))

    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model    

## Tab 4
# Create a Keras Classifier (to create entire neural network)
model = KerasClassifier(layers=1,neurons=32,build_fn=create_model,epochs=50,batch_size=10,verbose=0)

## Tab 5
# Define the grid search parameters
param_grid = {
    'neurons':[16,32,64,128],
    'layers':[1,2],
    'epochs':[50,100]
}

## Tab 6
# Perform grid search
grid = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,cv=3,verbose=1)
grid_result = grid.fit(X_train,y_train)

#Print the best parameters
print("Best: %f using %s"% (grid_result.best_score_,grid_result.best_params_)) #Ur machine processor gpu is not compatible to get output here
#use google colab as it has gpu's