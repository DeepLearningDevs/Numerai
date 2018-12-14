import pandas as pd
import numpy as np

def preprocess(target="bernie"):
	print("# Loading data...")
	# The training data is used to train your model how to predict the targets.
	train = pd.read_csv('numerai_training_data.csv', header=0)
	# The tournament data is the data that Numerai uses to evaluate your model.
	tournament = pd.read_csv('numerai_tournament_data.csv', header=0)
	
	
	validation = tournament[tournament['data_type']=='validation']
	
	train_bernie = train.drop(['id', 'era', 'data_type'], axis=1)
	ere_train=train["era"]
	ere_valid=validation["era"]
	ere_test=tournament["era"]
	features = [f for f in list(train_bernie) if "feature" in f]
	
	X_train = train_bernie[features]
	X_valid=validation[features]
	
	
	y_train = train_bernie['target_'+target]
	y_valid = validation['target_'+target]
	
	X_test=tournament[features]
	
	ids = tournament['id']
	
	return X_train, y_train, X_valid, y_valid, X_test, ere_train, ere_valid, ere_test,ids
