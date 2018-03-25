from keras.models import load_model
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

model = load_model("LSTM.h5")

data = pickle.load(open("model_data.pkl","rb"))

train_yp = model.predict(data['train_x'])
test_yp  = model.predict(data['test_x'])
predict_y  = model.predict(data['prediction_x'])

plot = pd.DataFrame()

NRx_1_to_18_actual     = data['train_y'].sum(axis=0)
NRx_1_to_18_predicted  = train_yp.sum(axis=0)

NRx_7_to_24_actual     = data['test_y'].sum(axis=0)
NRx_7_to_24_predicted  = test_yp.sum(axis=0)

NRx_13_to_30_predicted = predict_y.sum(axis=0)

# plot['actual']    = [0]*30
# plot['predicted'] = [0]*30

for i in range(18):
	plot.loc[i,'actual']    = NRx_1_to_18_actual[i,0]
	plot.loc[i,'predicted'] = NRx_1_to_18_predicted[i,0]

for i in range(18,24):
	plot.loc[i,'actual']    = NRx_7_to_24_actual[i-6,0]
	plot.loc[i,'predicted'] = NRx_7_to_24_predicted[i-6,0]

for i in range(24,30):
	plot.loc[i,'predicted'] = NRx_13_to_30_predicted[i-12,0]

# plot.plot(kind='line')
# plt.show()

plot.to_excel("plot.xlsx",index=None)


