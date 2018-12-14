# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 01:46:50 2018

@author: Matej Pavlovic
"""
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib.mlab as mlab

def FeatureEng(train,tournament,target="bernie"):
	model=XGBClassifier(nthread = 4,max_depth= 5,silent=False, \
	                    n_estimators= 900, gamma=0.1 ,subsample = 0.5 )
	
	validation = tournament[tournament['data_type']=='validation']
	
	train_bernie = train.drop(['id', 'era', 'data_type'], axis=1)
	
	features = [f for f in list(train_bernie) if "feature" in f]
	
	
	y = train_bernie['target_'+target]
	y_valid = validation['target_'+target]
	br=0
	
	print(br)
	X = train_bernie[features]
	X_valid=validation[features]
	X_test=tournament[features]
	
	model.fit(X,y,eval_set= [[X_valid, y_valid]],eval_metric="logloss",early_stopping_rounds=10,verbose=0)
	
	print("train logloss ",log_loss(y,model.predict_proba(X)[:,1]))
	print("val logloss ",log_loss(y_valid,model.predict_proba(X_valid)[:,1]))
	
	rj=pd.DataFrame()
	rj["Feature"]=list(X)
	rj["Importance"]=model.feature_importances_
	rj=rj.sort_values("Importance",ascending=False)
	rj.to_csv("./Feature/"+str(br)+".csv",index=False)
	br+=1
	
	print(br)
	X = train_bernie[features]
	X_valid=validation[features]
	for i in tqdm(range(1,51)):
		for j in range(i,51):
			X[str(i)+'*'+str(j)]=X["feature"+str(i)]*X["feature"+str(j)]
			X_valid[str(i)+'*'+str(j)]=X_valid["feature"+str(i)]*X_valid["feature"+str(j)]
	
	
	model.fit(X,y,eval_set= [[X_valid, y_valid]],eval_metric="logloss",early_stopping_rounds=10,verbose=0)
	
	print("train logloss ",log_loss(y,model.predict_proba(X)[:,1]))
	print("val logloss ",log_loss(y_valid,model.predict_proba(X_valid)[:,1]))
	
	rj=pd.DataFrame()
	rj["Feature"]=list(X)
	rj["Importance"]=model.feature_importances_
	rj=rj.sort_values("Importance",ascending=False)
	rj.to_csv("./Feature/"+str(br)+".csv",index=False)
	br+=1
	
	
	print(br)
	X = train_bernie[features]
	X_valid=validation[features]
	for i in tqdm(range(1,51)):
		for j in range(i,51):
			X[str(i)+'/'+str(j)]=X["feature"+str(i)]/X["feature"+str(j)]
			X_valid[str(i)+'/'+str(j)]=X_valid["feature"+str(i)]/X_valid["feature"+str(j)]
	
	
	model.fit(X,y,eval_set= [[X_valid, y_valid]],eval_metric="logloss",early_stopping_rounds=10,verbose=0)
	
	print("train logloss ",log_loss(y,model.predict_proba(X)[:,1]))
	print("val logloss ",log_loss(y_valid,model.predict_proba(X_valid)[:,1]))
	
	rj=pd.DataFrame()
	rj["Feature"]=list(X)
	rj["Importance"]=model.feature_importances_
	rj=rj.sort_values("Importance",ascending=False)
	rj.to_csv("./Feature/"+str(br)+".csv",index=False)
	br+=1
	
	
	
	print(br)
	X = train_bernie[features]
	X_valid=validation[features]
	for i in tqdm(range(1,51)):
		for j in range(i,51):
			X[str(i)+'**'+str(j)]=np.abs(X["feature"+str(i)].values)**X["feature"+str(j)].values
			X_valid[str(i)+'**'+str(j)]=np.abs(X_valid["feature"+str(i)].values)**X_valid["feature"+str(j)].values
	
	
	model.fit(X,y,eval_set= [[X_valid, y_valid]],eval_metric="logloss",early_stopping_rounds=10,verbose=0)
	
	print("train logloss ",log_loss(y,model.predict_proba(X)[:,1]))
	print("val logloss ",log_loss(y_valid,model.predict_proba(X_valid)[:,1]))
	
	rj=pd.DataFrame()
	rj["Feature"]=list(X)
	rj["Importance"]=model.feature_importances_
	rj=rj.sort_values("Importance",ascending=False)
	rj.to_csv("./Feature/"+str(br)+".csv",index=False)
	br+=1
	
	
	
	
	print(br)
	X = train_bernie[features]
	X_valid=validation[features]
	for i in tqdm(range(1,51)):
		for j in range(i,51):
			X[str(i)+'*/'+str(j)]=np.abs(X["feature"+str(i)].values)**(1/X["feature"+str(j)].values)
			X_valid[str(i)+'*/'+str(j)]=np.abs(X_valid["feature"+str(i)].values)**(1/X_valid["feature"+str(j)].values)
	
	
	model.fit(X,y,eval_set= [[X_valid, y_valid]],eval_metric="logloss",early_stopping_rounds=10,verbose=0)
	
	print("train logloss ",log_loss(y,model.predict_proba(X)[:,1]))
	print("val logloss ",log_loss(y_valid,model.predict_proba(X_valid)[:,1]))
	
	rj=pd.DataFrame()
	rj["Feature"]=list(X)
	rj["Importance"]=model.feature_importances_
	rj=rj.sort_values("Importance",ascending=False)
	rj.to_csv("./Feature/"+str(br)+".csv",index=False)
	br+=1
	
	
	print(br)
	X = train_bernie[features]
	X_valid=validation[features]
	for i in tqdm(range(1,51)):
		for j in range(i,51):
			if i!=j:
				X[str(i)+'L'+str(j)]=np.log(np.abs(X["feature"+str(i)].values))/np.log(np.abs(X["feature"+str(j)].values))
				X_valid[str(i)+'L'+str(j)]=np.log(np.abs(X_valid["feature"+str(i)].values))/np.log(np.abs(X_valid["feature"+str(j)].values))
			else:
				X[str(i)+'L'+str(j)]=np.log(np.abs(X["feature"+str(i)].values))
				X_valid[str(i)+'L'+str(j)]=np.log(np.abs(X_valid["feature"+str(i)].values))
				
	
	model.fit(X,y,eval_set= [[X_valid, y_valid]],eval_metric="logloss",early_stopping_rounds=10,verbose=0)
	
	print("train logloss ",log_loss(y,model.predict_proba(X)[:,1]))
	print("val logloss ",log_loss(y_valid,model.predict_proba(X_valid)[:,1]))
	
	rj=pd.DataFrame()
	rj["Feature"]=list(X)
	rj["Importance"]=model.feature_importances_
	rj=rj.sort_values("Importance",ascending=False)
	rj.to_csv("./Feature/"+str(br)+".csv",index=False)
	br+=1
	
	
	X = train_bernie[features]
	X_valid=validation[features]
	X_test=tournament[features]
	
	
	
	feat=pd.DataFrame()
	for i in range(6):
		temp=pd.read_csv("./Feature/"+str(i)+".csv")
		temp=temp[temp["Importance"]!=0]
		temp=temp.iloc[:int(len(temp)*0.2),:]
		feat=feat.append(temp)
	
	feat=feat[feat["Importance"]!=0]
	feat=feat.groupby("Feature").first()
	
	feat=feat.reset_index()
	
	for i in tqdm(range(len(feat))):
		string=feat["Feature"].iloc[i]
		temp=string.split("*/")
		if len(temp)!=1:
	
			X[temp[0]+'*/'+temp[1]]=np.abs(X["feature"+temp[0]].values)**(1/X["feature"+temp[1]].values)
			X_valid[temp[0]+'*/'+temp[1]]=np.abs(X_valid["feature"+temp[0]].values)**(1/X_valid["feature"+temp[1]].values)
			X_test[temp[0]+'*/'+temp[1]]=np.abs(X_test["feature"+temp[0]].values)**(1/X_test["feature"+temp[1]].values)
			continue
	
		if len(temp)==1:
			temp=string.split("**")
			#print(temp,len(temp))
			if len(temp)!=1:
	
				X[temp[0]+'**'+temp[1]]=np.abs(X["feature"+temp[0]].values)**(X["feature"+temp[1]].values)
				X_valid[temp[0]+'**'+temp[1]]=np.abs(X_valid["feature"+temp[0]].values)**(X_valid["feature"+temp[1]].values)
				X_test[temp[0]+'**'+temp[1]]=np.abs(X_test["feature"+temp[0]].values)**(X_test["feature"+temp[1]].values)
				continue
			
		if len(temp)==1:
			temp=string.split("L")
			#print(string,temp)
			if len(temp)!=1:
				i=int(temp[0])
				j=int(temp[1])
	
				if i!=j:
					X[str(i)+'L'+str(j)]=np.log(np.abs(X["feature"+str(i)].values))/np.log(np.abs(X["feature"+str(j)].values))
					X_valid[str(i)+'L'+str(j)]=np.log(np.abs(X_valid["feature"+str(i)].values))/np.log(np.abs(X_valid["feature"+str(j)].values))
					X_test[str(i)+'L'+str(j)]=np.log(np.abs(X_test["feature"+str(i)].values))/np.log(np.abs(X_test["feature"+str(j)].values))
	
				else:
					X[str(i)+'L'+str(j)]=np.log(np.abs(X["feature"+str(i)].values))
					X_valid[str(i)+'L'+str(j)]=np.log(np.abs(X_valid["feature"+str(i)].values))
					X_test[str(i)+'L'+str(j)]=np.log(np.abs(X_test["feature"+str(i)].values))
				continue
		if len(temp)==1:
			temp=string.split("*")
			if len(temp)!=1:
				i=int(temp[0])
				j=int(temp[1])
				X[str(i)+'*'+str(j)]=X["feature"+str(i)]*X["feature"+str(j)]
				X_valid[str(i)+'*'+str(j)]=X_valid["feature"+str(i)]*X_valid["feature"+str(j)]
				X_test[str(i)+'*'+str(j)]=X_test["feature"+str(i)]*X_test["feature"+str(j)]
				continue
			
		if len(temp)==1:
			temp=string.split("/")
			if len(temp)!=1:
				i=int(temp[0])
				j=int(temp[1])
				X[str(i)+'/'+str(j)]=X["feature"+str(i)]/X["feature"+str(j)]
				X_valid[str(i)+'/'+str(j)]=X_valid["feature"+str(i)]/X_valid["feature"+str(j)]
				X_test[str(i)+'/'+str(j)]=X_test["feature"+str(i)]/X_test["feature"+str(j)]
				continue
		else:
			temp=string.split("feature")
			#print(temp)
	
	
	for stup in tqdm(list(X)):
		temp=X[stup].values
		maksi=np.max(temp[np.where(temp!=np.inf)])
		mini=np.min(temp[np.where(temp!=-np.inf)])
		if maksi>1:
			rang=maksi-mini
			X[stup]=X[stup]/rang
			X_valid[stup]=X_valid[stup]/rang
			X_test[stup]=X_test[stup]/rang
			
			X[stup][X[stup]==np.inf]=2*maksi/rang
			X_valid[stup][X_valid[stup]==np.inf]=2*maksi/rang
			X_test[stup][X_test[stup]==np.inf]=2*maksi/rang
			
			X[stup][X[stup]==-np.inf]=-2*maksi/rang
			X_valid[stup][X_valid[stup]==-np.inf]=-2*maksi/rang
			X_test[stup][X_test[stup]==-np.inf]=-2*maksi/rang
	
	return X, X_valid, X_test
	
	


def DAE_augment(X_train,trashold=0.2):
	
	mask=np.random.uniform(0,1,np.shape(X_train))
	mask=np.where(mask<trashold)
	
	temp=X_train.values
	temp[mask]=0
	
	return temp




X_input=DAE_augment(X_tr)

input_vec = Input(shape=(np.shape(X_train)[1],))
encoded = Dense(32, activation='relu')(input_vec)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(8, activation='linear')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(np.shape(X_train)[1], activation='sigmoid')(decoded)


autoencoder = Model(input_vec, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')


encoder = Model(input_vec, encoded)

encoded_input = Input(shape=(8,))


autoencoder.fit(X_input, X_tr,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))#,callbacks=[es])

X_tr_2=encoder.predict(X_train)
X_te_2=encoder.predict(X_test)
#########################
dubina=[3,4,5]
gamma=[0.1,0.2,0.05]
stabla=[100,800,900,1000]


model=XGBClassifier(nthread = 3,max_depth= 4,silent=False, \
                    n_estimators= 500, gamma=10, subsample = 0.3 )



kf = KFold(n_splits=cv_fold,shuffle=True)
for i,(train_index, test_index )in enumerate(kf.split(X_train)):
    
    X_train_kf, X_val = X_tr_2[train_index,:], X_tr_2[test_index,:]
    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

    model.fit(X_train_kf,y_train_kf,eval_set= [[X_val, y_val]], eval_metric="logloss",early_stopping_rounds=10,verbose=0)
    
    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict_proba(X_train_kf)[:,1]))
    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict_proba(X_val)[:,1]))









##############################################
X_tr=X_train.append(X_valid)

ere_tr=ere_train.append(ere_valid)
ere_tr_unique=pd.unique(ere_tr)

y=np.zeros((len(X_tr),1))
y[:len(X_train)]=1

udio=sum(y)/len(y)

tezine=[1-udio]*len(X_train)+[udio]*len(X_valid)
tezine=np.reshape(np.array(tezine),(-1))

input_vec = Input(shape=(np.shape(X_train)[1],))
encoded = Dense(32, activation='sigmoid')(input_vec)
#decoded = Dense(16, activation='sigmoid')(encoded)
decoded = Dense(8, activation='sigmoid')(encoded)
decoded = Dense(1, activation='sigmoid')(decoded)


autoencoder = Model(input_vec, decoded)
es=EarlyStopping(monitor='val_loss',min_delta=0,patience=8, verbose=2, mode='auto',restore_best_weights=True)

output=pd.DataFrame(y.copy())
output["Klasa"]=0

for k in range(4):
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in tqdm(enumerate(kf.split(ere_tr_unique))):
		
		pr_index=[]
		for er in train_index:
			pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
		train_index=pr_index
		
		pr_index=[]
		for er in test_index:
			pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
		test_index=pr_index
		
		
		X_train_kf, X_val = X_tr.iloc[train_index,:], X_tr.iloc[test_index,:]
		y_train_kf, y_val = y[train_index], y[test_index]
		tezine_train, tezine_test = tezine[train_index], tezine[test_index]
		
		
		autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=["acc"])
	
		autoencoder.fit(X_train_kf, y_train_kf, sample_weight=tezine_train,
							 epochs=40,verbose=0,
							batch_size=256,callbacks=[es],
							shuffle=True, validation_data=(X_val, y_val,tezine_test))
	
		#print("Fold ",i,", train logloss ",log_loss(y_train_kf,autoencoder.predict(X_train_kf)[:,0]))
		#print("Fold ",i,", val logloss ",log_loss(y_val,autoencoder.predict(X_val)[:,0]))
		output["Klasa"].iloc[test_index]=autoencoder.predict(X_val)[:,0]
	
	temp=output[output[0]==1]
	
	tezine=tezine*np.array((1-output["Klasa"])/output["Klasa"])
	tezine[output[0]==0]=1
	tezine=np.array(tezine)

out=autoencoder.predict(X_tr)
temp=np.reshape(np.maximum(y-out,0)+1,(-1))
tezine=tezine*temp





	
temp=np.zeros((len(list(X_train)),4))
opis=pd.DataFrame(temp,columns=["Ime","Mean","Std","Skew"])
for i,stup in tqdm(enumerate(list(X_train))):
	temp=X_train[stup].values
	maksi=np.mean(temp)
	mini=np.std(temp)

		
	opis["Ime"].iloc[i]=stup
	opis["Mean"].iloc[i]=maksi
	opis["Std"].iloc[i]=mini
	opis["Skew"].iloc[i]=sp.stats.skew(temp)
	plt.figure()
	plt.hist(temp,bins=40,normed=True)
	plt.plot(np.linspace(0,1),mlab.normpdf(np.linspace(0,1), maksi, mini))
	plt.show()
	









temp=np.zeros((len(list(X)),4))
opis=pd.DataFrame(temp,columns=["Ime","Mean","Std","Skew"])

for i,stup in tqdm(enumerate(list(X))):
	temp=X[stup].values
	maksi=np.mean(temp)
	mini=np.std(temp)

		
	opis["Ime"].iloc[i]=stup
	opis["Mean"].iloc[i]=maksi
	opis["Std"].iloc[i]=mini
	opis["Skew"].iloc[i]=sp.stats.skew(X[stup].values)
	#pinf=np.isinf(X[stup].values)
	#minf=np.i