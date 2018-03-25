from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

data = pickle.load(open("model_data.pkl","rb"))

Train_X = data['train_x']
Train_Y = data['train_y'] 
Test_X = data['test_x']
Test_Y = data['test_y']

model = {}

for nodes in [64,100,128,192]:
	for drop in [0,0.25]:

		model[(nodes,drop)] = Sequential()
		model[(nodes,drop)].add(LSTM(64, return_sequences=True, input_shape=(18,6), dropout=0.2))
		model[(nodes,drop)].add(Dense(1,activation='relu'))

		opt = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999)

		model[(nodes,drop)].compile(loss='mse', optimizer=opt)

		#tbCallBack = TensorBoard(log_dir='./model_logs', histogram_freq=1, write_graph=True, write_images=True)

		model[(nodes,drop)].fit(Train_X, Train_Y, epochs=100, batch_size=1, validation_data=(Test_X, Test_Y), shuffle=False)#,callbacks=[tbCallBack])

		model[(nodes,drop)].save('LSTM '+str((nodes,drop))+'.h5')
