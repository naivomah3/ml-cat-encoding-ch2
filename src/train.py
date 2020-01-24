import os
import pandas as pd
import joblib

from sklearn import preprocessing
from sklearn import metrics

from . import dispatcher

# Get env variables
TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4], # Train on [1, 2, 3, 4], Validate on [0]
    1: [0, 2, 3, 4], # Train on [0, 2, 3, 4], Validate on [1]
    2: [0, 1, 3, 4], # Train on [0, 1, 3, 4], Validate on [2]
    3: [0, 1, 2, 4], # ...
    4: [0, 1, 2, 3],
}

if __name__ == '__main__':
    # Get Train data as per FOLD
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df['kfold'].isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df['kfold']==FOLD].reset_index(drop=True)

    # Load the dataset
    y_train = train_df['target'].values
    y_valid = valid_df['target'].values

    train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

    # Adjust the order of variables
    valid_df = valid_df[train_df.columns]

    # Encode categorical variables
    # Approach: simple ordinal mapping using LabelEncoder
    # Further improve by using different approaches
    label_encoders = [] # To store labels
    for col in train_df.columns:
        label_encoder = preprocessing.LabelEncoder()
        # Get mapping
        label_encoder.fit(train_df[col].values.tolist() + valid_df[col].values.tolist() + df_test[col].values.tolist())
        # Transform col
        train_df.loc[:, col] = label_encoder.transform(train_df[col].values.tolist())
        valid_df.loc[:, col] = label_encoder.transform(valid_df[col].values.tolist())
        # Save labels
        label_encoders.append((col, label_encoder))


    # Dataset is ready to train, define classifier
    clf = dispatcher.MODELS[MODEL] # Load model upon the value of MODEL
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(y_valid, preds))

    # Save model
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder_mapping.pkl") # To later encode testing data
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl") # Save the model
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl") # Saved columns used for training
