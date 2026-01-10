## Tab 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

## Tab 2
data = pd.read_csv('Churn_Modelling.csv') #Estimated salary is output feature (hence its a regression problem statement)
data.head()

## Tab 3
# Preprocess the data
data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)

## Tab 4
#Encode categorical variables
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

## Tab 5
# One-hot encode 'Geography'
onehot_encoder_geo = OneHotEncoder(handle_unknown='ignore')
geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df

## Tab 6
# Combine one-hot encoded columns with original data
data = pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)
data.head()

## Tab 7
# Split the data into features and target
X = data.drop('EstimatedSalary',axis=1)
y = data['EstimatedSalary']

## Tab 8
## Split the data in training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

## Tab 9
# Scale these features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Tab 10
# Save the encoders and scaler for later use
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)

with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

## Tab 11
## ANN Regression Problem Statement
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## Tab 12
# Build the model
model = Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
    Dense(32,activation='relu'),
    Dense(1) #Output layer for regression(when u don't apply any activation function the default function linear is applied)
])

# Compile the model
model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mae'])

model.summary()

## Tab 13
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

#Set up Tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir = log_dir,histogram_freq=1)

## Tab 14
# Set up Early Stopping
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

## Tab 15
# Train the model
history = model.fit(
    X_train,y_train,
    validation_data=(X_test,y_test),
    epochs=100,
    callbacks=[early_stopping_callback,tensorboard_callback]
)

## Tab 16
%load_ext tensorboard

## Tab 17
%tensorboard --logdir logs/fit

## Tab 18
## Evaluate model on the test data
test_loss,test_mae = model.evaluate(X_test,y_test)
print(f'Test MAE:{test_mae}')

## Tab 19
model.save('regression_model.h5')