import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('../input/categorical_train.csv')
    df['kfold'] = -1
    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    # print(type(df['target'].values))

    # create K-Fold
    sk_fold = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    # Create Fold and Train/Val indices
    for fold, (train_index, val_index) in enumerate(sk_fold.split(X=df, y=df['target'].values)):
        print(len(train_index), len(val_index))
        df.loc[val_index, 'kfold'] = fold

    df.to_csv('input/categorical_train_folds.csv', index=False)



