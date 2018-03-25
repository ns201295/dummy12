import pandas as pd
import pickle

d = pd.read_csv("HVHR.csv")
d = d.sort_values(['SHS_ID','YEARMONTH']).reset_index(drop=True)

input_vars = ['PDE_transformed','DSI_SAMPLES_transformed','CSS_Samples_transformed','tmot_supplied_transformed','EMAIL_OPEN','sales_contest']


train_data = d[d['YEARMONTH'] <= 201702].reset_index(drop=True)
train_x = train_data[input_vars].as_matrix().reshape(-1,1,len(input_vars))
train_y = train_data['NRx'].as_matrix().reshape(-1,1)


test_data = d[d['YEARMONTH'] >= 201603].reset_index(drop=True)
test_x = d[input_vars].as_matrix().reshape(-1,1,len(input_vars))
test_y = d['NRx'].as_matrix().reshape(-1,1)

mean_mon = [201703,201704,201705,201706,201707,201708]
pred_mon = [201709,201710,201711,201712,201801,201802]

mean_vals = d[d['YEARMONTH'].isin(mean_mon)].groupby('SHS_ID')[input_vars].mean().reset_index()
mean_vals['key'] = 'k'

temp = pd.DataFrame()
temp['YEARMONTH'] = pred_mon
temp['key'] = 'k'

future_x = pd.merge(temp,mean_vals,on='key')
del future_x['key']

prediction_x = d
del prediction_x['NRx']

prediction_x = pd.concat([prediction_x,future_x]).sort_values(['SHS_ID','YEARMONTH']).reset_index(drop=True)
prediction_x = prediction_x[input_vars].as_matrix().reshape(-1,1,len(input_vars))


data = {'train_x':train_x,'train_y':train_y,'test_x':test_x,'test_y':test_y,'prediction_x':prediction_x}

pickle.dump(data,open("model_data.pkl","wb"))

