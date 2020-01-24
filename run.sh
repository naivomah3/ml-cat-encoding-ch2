#----------------------------------------
# Training scripts
#----------------------------------------
#### Load ENV
#export TRAINING_DATA=input/categorical_train_folds.csv
#export TEST_DATA=input/categorical_test.csv
#export MODEL=$1 # pass as an argument
#### Load scripts
#export FOLD=0 && python -m src.train
#export FOLD=1 && python -m src.train
#export FOLD=2 && python -m src.train
#export FOLD=3 && python -m src.train
#export FOLD=4 && python -m src.train

#----------------------------------------
# Testing scripts
#----------------------------------------
### Load ENV
export TEST_DATA=input/categorical_test.csv
export MODEL=$1 # pass as an argument
### Load scripts
python -m src.predict

#----------------------------------------
# HOW TO TRAIN?
# sh src.train randomforest
# sh src.train extratrees
# HOW TO PREDICT?
# sh src.test randomforest
# sh src.test extratrees
#----------------------------------------