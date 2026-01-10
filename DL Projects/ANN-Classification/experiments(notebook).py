## Tab1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle ##to reuse files during deployment

## Tab2
#Load the dataset
data = pd.read_csv("Churn_Modelling.csv")
data.head()

## Tab3
## Preprocess the data
## Drop irrelevant columns
data = data.drop(['RowNumber','CustomerId','Surname'],axis=1) ## Axis = 1 means column wise
data

## Tab4
## Encode categorical variables
label_encoder_gender = LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
data

## Tab5
## Onehot encode 'Geography'
from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo=OneHotEncoder()
geo_encoder= onehot_encoder_geo.fit_transform(data[['Geography']])
geo_encoder

## Tab6
geo_encoder.toarray()

## Tab7
onehot_encoder_geo.get_feature_names_out(['Geography']) 

## Tab8
geo_encoded_df=pd.DataFrame(geo_encoder.toarray(),columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df

## Tab9
# Combine one hot encoder columns with the original data
data=pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)
data.head()

## Tab10
## Sve the encoders and scaler
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file) ##file me dump kar diya

with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)

## Tab11
data.head()

## Tab12
# Divide the dataset into independent and dependent features
X = data.drop('Exited',axis=1)
y = data['Exited']

# Split the data in training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

## Scale these features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Tab13
X_train

## Tab14
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

## Tab15
data



### ANN Implementation
## Tab 16
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

## Tab 17
(X_train.shape[1],)  #it says it has single dimension and 12 inputs

## Tab 18
#Build our ANN model
model = Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), ## (HL1)First hidden layer and is connected with input layer
    Dense(32,activation='relu'),## Input layer me hi sirf pass karna hai input shape (HL2)
    Dense(1,activation='sigmoid') #Output layer
]

)

## Tab 19
model.summary()

## Tab 20
import tensorflow
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss=tensorflow.keras.losses.BinaryCrossentropy()
loss

## Tab 21
# Compile the model
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

## Tab 22
## Set up the Tensorboard
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
#Tensorboard to visualize all the logs u have while training ur model
log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callbacks = TensorBoard(log_dir=log_dir,histogram_freq=1)

## Tab 23
## Set up Early Stopping
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
## Patience ki agar update na ho weight toh 5 epochs tak aur try karna hai

## Tab 24
## Train the model
history = model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callbacks,early_stopping_callback]## Two callbacks
)

## Tab 25
model.save('model.h5')

## Tab 26
# Load Tensorboard Extension
%load_ext tensorboard

## Tab 27
%tensorboard --logdir logs/fit