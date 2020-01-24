import os
import pandas as pd
import joblib
import numpy as np

# Get env variables
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    # Get Test
    df = pd.read_csv(TEST_DATA)
    test_index = df['id'].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        # Encode categorical variable by using the same mapping on training
        # Load labels to encode testing data
        encoders = joblib.load(os.path.join('models', f"{MODEL}_{FOLD}_label_encoder_mapping.pkl"))

        # Get columns to predict
        cols = joblib.load(os.path.join('models', f"{MODEL}_{FOLD}_columns.pkl"))

        # Walk through columns to encode them all
        for (col, label_encoder) in encoders:
            # Transform col
            df.loc[:, col] = label_encoder.transform(df[col].values.tolist())

        # Load classifier
        clf = joblib.load(os.path.join('models', f"{MODEL}_{FOLD}.pkl"))

        # Predict
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]
        # Save prediction probs and Add them up together
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds


    # Take the mean of each row(predictions)
    predictions /= 5

    # DataFrame for Kaggle submission
    preds_submission = pd.DataFrame(np.column_stack((test_index, predictions)), columns=['id', 'target'])

    return preds_submission

if __name__ == '__main__':
    submission = predict()
    submission.to_csv(os.path.join('models', f"{MODEL}.csv"), index=False)
