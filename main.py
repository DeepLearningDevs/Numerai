from utility import build_keras_model_1, fit_predict, master_nn
import pandas as pd

if __name__ == '__main__':


	setovi=['bernie','elizabeth','jordan','ken','charles']

	X_train_s,X_test_s=master_nn()


	for target in setovi:
		X_train,y_train,X_valid,y_valid,X_test,ere_train,ere_valid,ere_test, ids = preprocess(target)
		ere_train=ere_train.append(ere_valid)
		X_train=X_train.append(X_valid)
		y_train=y_train.append(y_valid)
		tezine=np.ones((len(X_train)))
		_,out=fit_predict(X_train,y_train,X_test, X_train_s[list(X_train)+["Master_nn_"+target]], X_test_s[list(X_train)+["Master_nn_"+target]],tezine,ere_train)
		predaja=pd.DataFrame(ids,columns=["id"])
		predaja["probability_"+target]=out["LVL2"].values
		predaja.to_csv(target+"_sub.csv",index=False)
		
