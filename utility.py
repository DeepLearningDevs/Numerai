
from preprocess import preprocess
import numpy as np
from sklearn import metrics, preprocessing, linear_model
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import random
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from keras.layers import Dropout, Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from keras.models import Sequential



def build_keras_model_1(X_train):

    model = Sequential()
    model.add(Dense(64,input_dim=X_train.shape[1],activation='relu'))

    # "encoded" is the encoded representation of the input

    model.add( Dense(32, activation='relu'))
    if np.random.uniform()>0.4:
        model.add( Dropout(0.3))
        model.add( Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
    model.add( Dropout(0.2))
    if np.random.uniform()>0.2:
        model.add( Dense(16, activation='relu'))
        model.add( Dropout(0.25))
    model.add( Dense(16, activation='relu'))
    model.add( Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.0003), loss='binary_crossentropy',metrics=['binary_crossentropy'])
    return model


def fit_predict(X_train,y_train,X_test, X_train_s, X_test_s,tezine ,ere_train,cv_fold=5):
	
	ere_tr=ere_train
	#ere_tr=ere_train.append(ere_valid)
	ere_tr_unique=pd.unique(ere_tr)
	
	#####################################################
	#####    XGB
	print("XGB")
	
	dubina=[3,4,5]
	gamma=[0.1,0.2,0.05]
	stabla=[100,800,900,1000]
	
	
	model=XGBClassifier(nthread = 4,max_depth= random.choice(dubina),silent=False, \
	                    n_estimators= random.choice(stabla), gamma=random.choice(gamma), subsample = 0.5 )
	
	X_train_s["XGB"]=0
	X_test_s["XGB"]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):

	    pr_index=[]
	    for er in train_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    train_index=pr_index
	
	    pr_index=[]
	    for er in test_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    test_index=pr_index

	    X_train_kf, X_val = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
	    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
	    tezine_train, tezine_test = tezine[train_index], tezine[test_index]
	
	    model.fit(X_train_kf,y_train_kf,eval_set= [[X_val, y_val,tezine_test]],eval_metric="logloss",early_stopping_rounds=10,verbose=0, sample_weight=tezine_train)
	    
	    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict_proba(X_train_kf)[:,1]))
	    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict_proba(X_val)[:,1]))
	
	    X_train_s["XGB"].iloc[test_index]=model.predict_proba(X_val)[:,1]
	    X_test_s["XGB"]+=model.predict_proba(X_test)[:,1]
	
	X_test_s["XGB"]=X_test_s["XGB"]/cv_fold
	
	
	
	########################
	# Logisticka
	print("Logistic regression")
	model=LogisticRegression(random_state=0,n_jobs=-1,solver="sag",C=2)
	
	X_train_s["Log"]=0
	X_test_s["Log"]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):

	    pr_index=[]
	    for er in train_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    train_index=pr_index
	
	    pr_index=[]
	    for er in test_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    test_index=pr_index

	    X_train_kf, X_val = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
	    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
	    tezine_train, tezine_test = tezine[train_index], tezine[test_index]
	
	    model.fit(X_train_kf,y_train_kf,sample_weight=tezine_train)
	    
	    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict_proba(X_train_kf)[:,1]))
	    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict_proba(X_val)[:,1]))
	
	    X_train_s["Log"].iloc[test_index]=model.predict_proba(X_val)[:,1]
	    X_test_s["Log"]+=model.predict_proba(X_test)[:,1]
	
	X_test_s["Log"]=X_test_s["Log"]/cv_fold
	

	

	##############################################
	# ExtraTree
	print("ExtraTree")
	model=ExtraTreesClassifier(random_state=0,n_estimators=800,criterion= 'entropy',min_samples_split= 5,
	                            max_depth= 6, min_samples_leaf= 5,n_jobs=-1) 
	
	X_train_s["XT"]=0
	X_test_s["XT"]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):
	    
	    pr_index=[]
	    for er in train_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    train_index=pr_index
	
	    pr_index=[]
	    for er in test_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    test_index=pr_index

	    X_train_kf, X_val = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
	    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
	    tezine_train, tezine_test = tezine[train_index], tezine[test_index]
	
	    model.fit(X_train_kf,y_train_kf,sample_weight=tezine_train)
	
	    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict_proba(X_train_kf)[:,1]))
	    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict_proba(X_val)[:,1]))
	
	    X_train_s["XT"].iloc[test_index]=model.predict_proba(X_val)[:,1]
	    X_test_s["XT"]+=model.predict_proba(X_test)[:,1]
	
	X_test_s["XT"]=X_test_s["XT"]/cv_fold
	
	
	#################################
	## LightGBM
	print("LGBM")
	model=LGBMClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=300, max_depth=10)
	X_train_s["LGBM"]=0
	X_test_s["LGBM"]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):

	    pr_index=[]
	    for er in train_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    train_index=pr_index
	
	    pr_index=[]
	    for er in test_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    test_index=pr_index
	    
	    X_train_kf, X_val = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
	    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
	    tezine_train, tezine_test = tezine[train_index], tezine[test_index]
	
	    tezine_test=[np.array(i) for i in list(tezine_test)]
	    model.fit(X_train_kf,y_train_kf,eval_set= [[X_val, y_val]],eval_metric="logloss",early_stopping_rounds=10,verbose=0,sample_weight=list(tezine_train),eval_sample_weight=[tezine_test])
	    
	    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict_proba(X_train_kf)[:,1]))
	    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict_proba(X_val)[:,1]))
	
	    X_train_s["LGBM"].iloc[test_index]=model.predict_proba(X_val)[:,1]
	    X_test_s["LGBM"]+=model.predict_proba(X_test)[:,1]
	
	X_test_s["LGBM"]=X_test_s["LGBM"]/cv_fold
	
	
	##################################3
	### NN1
	print("Neural Netowk 1")
	es=EarlyStopping(monitor='val_loss',min_delta=0,patience=8, verbose=2, mode='auto',restore_best_weights=True)
	model=build_keras_model_1(X_train)
	
	X_train_s["NN1"]=0
	X_test_s["NN1"]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):
	    
	    pr_index=[]
	    for er in train_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    train_index=pr_index
	
	    pr_index=[]
	    for er in test_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    test_index=pr_index

	    X_train_kf, X_val = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
	    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
	    tezine_train, tezine_test = tezine[train_index], tezine[test_index]
	
	    model.fit(X_train_kf,y_train_kf,epochs=50, batch_size=256, shuffle=True, verbose=0,  sample_weight=tezine_train,
	                validation_data=(X_val, y_val,tezine_test),callbacks=[es])
	
	    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict(X_train_kf)))
	    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict(X_val)))
	
	    X_train_s["NN1"].iloc[test_index]=model.predict(X_val)[:,0]
	    X_test_s["NN1"]+=model.predict(X_test)[:,0]
	
	X_test_s["NN1"]=X_test_s["NN1"]/cv_fold
	
	##################################3
	### NN2
	print("Neural Netowk 2")
	es=EarlyStopping(monitor='val_loss',min_delta=0,patience=8, verbose=2, mode='auto',restore_best_weights=True)
	model=build_keras_model_1(X_train)
	
	X_train_s["NN2"]=0
	X_test_s["NN2"]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):
	    
	    pr_index=[]
	    for er in train_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    train_index=pr_index
	
	    pr_index=[]
	    for er in test_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    test_index=pr_index

	    X_train_kf, X_val = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
	    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
	    tezine_train, tezine_test = tezine[train_index], tezine[test_index]
	
	    model.fit(X_train_kf,y_train_kf,epochs=50, batch_size=256, shuffle=True, verbose=0,
	                 sample_weight=tezine_train,validation_data=(X_val, y_val,tezine_test),callbacks=[es])
	
	    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict(X_train_kf)))
	    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict(X_val)))
	
	    X_train_s["NN2"].iloc[test_index]=model.predict(X_val)[:,0]
	    X_test_s["NN2"]+=model.predict(X_test)[:,0]
	
	X_test_s["NN2"]=X_test_s["NN2"]/cv_fold
	
	##################################3
	### NN3
	print("Neural Netowk 3")
	es=EarlyStopping(monitor='val_loss',min_delta=0,patience=8, verbose=2, mode='auto',restore_best_weights=True)
	model=build_keras_model_1(X_train)
	
	X_train_s["NN3"]=0
	X_test_s["NN3"]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):
	    

	    pr_index=[]
	    for er in train_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    train_index=pr_index
	
	    pr_index=[]
	    for er in test_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    test_index=pr_index

	    X_train_kf, X_val = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
	    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
	    tezine_train, tezine_test = tezine[train_index], tezine[test_index]
	
	    model.fit(X_train_kf,y_train_kf,epochs=50, batch_size=256, shuffle=True, verbose=0,
	                 sample_weight=tezine_train, validation_data=(X_val, y_val,tezine_test),callbacks=[es])
	
	    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict(X_train_kf)))
	    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict(X_val)))
	
	    X_train_s["NN3"].iloc[test_index]=model.predict(X_val)[:,0]
	    X_test_s["NN3"]+=model.predict(X_test)[:,0]
	
	X_test_s["NN3"]=X_test_s["NN3"]/cv_fold
	
	#####################
	###### LEVEL 2  #####
	#####################

	print("LEVEL 2")
	
	
	model=XGBClassifier(nthread = 4,max_depth= 5,silent=False, \
	                    n_estimators= 800, gamma=0.3, subsample = 0.5 )
	
	X_train_s["LVL2"]=0
	X_test_s["LVL2"]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):
	    
	    pr_index=[]
	    for er in train_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    train_index=pr_index
	
	    pr_index=[]
	    for er in test_index:
	        pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
	    test_index=pr_index

	    X_train_kf, X_val = X_train_s.iloc[train_index,:], X_train_s.iloc[test_index,:]
	    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
	    tezine_train, tezine_test = tezine[train_index], tezine[test_index]
	
	    model.fit(X_train_kf,y_train_kf,eval_set= [[X_val, y_val]],eval_metric="logloss",early_stopping_rounds=10,verbose=0)
	    
	    print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict_proba(X_train_kf)[:,1]))
	    print("Fold ",i,", val logloss ",log_loss(y_val,model.predict_proba(X_val)[:,1]))
	
	    X_train_s["LVL2"].iloc[test_index]=model.predict_proba(X_val)[:,1]
	    X_test_s["LVL2"]+=model.predict_proba(X_test_s)[:,1]
	
	X_test_s["LVL2"]=X_test_s["LVL2"]/cv_fold
	

	return X_train_s, X_test_s

def master_nn(cv_fold=5,setovi=['bernie','elizabeth','jordan','ken','charles']):
	
	
	
	train = pd.read_csv('numerai_training_data.csv', header=0)
	tournament = pd.read_csv('numerai_tournament_data.csv', header=0)
	validation = tournament[tournament['data_type']=='validation']
	
	train_bernie = train.drop(['id', 'era', 'data_type'], axis=1)
	features = [f for f in list(train_bernie) if "feature" in f]
	ere_train=train["era"]
	ere_valid=validation["era"]
	X = train_bernie[features]
	X_valid=validation[features]
	
	Y = train_bernie[['target_'+target for target in setovi]]
	Y_valid = validation[['target_'+target for target in setovi]]
	
	X_test=tournament[features]
	
	
	X_train=X.append(X_valid)
	y_train=Y.append(Y_valid)
	
	
	X_train_s=X_train.copy()
	X_test_s=X_test.copy()
	
	X_train=X_train-0.5
	
	model = Sequential()
	model.add(Dense(64,input_dim=X_train.shape[1],activation='sigmoid'))
	model.add( Dense(32, activation='sigmoid'))
	model.add( Dropout(0.2))
	model.add(Dense(16, activation='sigmoid'))
	model.add( Dropout(0.2))
	model.add( Dense(16, activation='sigmoid'))
	model.add( Dense(8, activation='sigmoid'))

	model.add(Dense(5, activation='sigmoid'))

	ere_tr=ere_train.append(ere_valid)
	ere_tr_unique=pd.unique(ere_tr)
	es=EarlyStopping(monitor='val_loss',min_delta=0,patience=8, verbose=2, mode='auto',restore_best_weights=True)
	for target in setovi:
		X_train_s["Master_nn_"+target]=0
		X_test_s["Master_nn_"+target]=0
	
	kf = KFold(n_splits=cv_fold,shuffle=True)
	for i,(train_index, test_index )in enumerate(kf.split(ere_tr_unique)):
		
		pr_index=[]
		for er in train_index:
			pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])
		train_index=pr_index
	
		pr_index=[]
		for er in test_index:
			pr_index=pr_index+list(np.where(np.array(ere_tr)==ere_tr_unique[er])[0])

		test_index=pr_index
		X_train_kf, X_val = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
		y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
		model.compile(optimizer=Adam(lr=0.003), loss='binary_crossentropy')
		model.fit(X_train_kf,y_train_kf,epochs=50, batch_size=256, shuffle=True, verbose=0,
						validation_data=(X_val, y_val),callbacks=[es])
	
		print("Fold ",i,", train logloss ",log_loss(y_train_kf,model.predict(X_train_kf)))
		print("Fold ",i,", val logloss ",log_loss(y_val,model.predict(X_val)))
	
		pred_train=model.predict(X_val)
		pred_test=model.predict(X_test)
		for i,target in enumerate(setovi):
			X_train_s["Master_nn_"+target].iloc[test_index]=pred_train[:,i]
			X_test_s["Master_nn_"+target]+=pred_test[:,i]
	for i,target in enumerate(setovi):
		X_test_s["Master_nn_"+target]=X_test_s["Master_nn_"+target]/cv_fold
		
	return X_train_s, X_test_s


