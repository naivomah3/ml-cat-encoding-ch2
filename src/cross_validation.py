import pandas as pd
from sklearn import model_selection

class CrossValidation:
    def __init__(self,
                 dataframe,
                 target_cols,
                 problem_type="binary_classification",
                 multilabel_delimiter=",",  # in case of "multi label classification" -> target labels delimiter
                 no_folds=5,
                 shuffle=True,
                 random_sate=42
                 ):

        self.dataframe = dataframe
        self.target_cols = target_cols     # list of names of all target variable
        self.no_targets = len(target_cols)
        self.problem_type = problem_type
        self.multilabel_delimiter = multilabel_delimiter
        self.no_folds = no_folds
        self.shuffle = shuffle
        self.random_state = random_sate


        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        # Create a new column kfold to hold folding key
        self.dataframe['kfold'] = -1

    def split(self):
        # Problem 1:
        # --> "binary classification"
        # --> "multi class classification"
        if self.problem_type in ('binary_classification', 'multiclass_classification'):
            if self.no_targets != 1: # More than 1 target cols
                raise Exception("Invalid number of target columns")

            target = self.target_cols[0] # Get the name of the first col
            unique_values = self.dataframe[target].nunique()  # unique class values
            if unique_values == 1:    # Only one class is present
                raise Exception("Only one unique value found")
            elif unique_values > 1:  # Means "multi" class classification
                sk_fold = model_selection.StratifiedKFold(n_splits=self.no_folds,
                                                          shuffle=False,
                                                          random_state=self.random_state)  # Random state has no effect

                # Create Fold and Train/Val indices
                for fold, (train_index, val_index) in enumerate(sk_fold.split(X=self.dataframe, y=self.dataframe.target.values)):
                    self.dataframe.loc[val_index, 'kfold'] = fold

        # Problem 2:
        # --> "Single column regression"
        # --> "Multi column regression"
        elif self.problem_type in ('single_col_regression', 'multi_col_regression'):
            if self.no_targets != 1 and self.problem_type == 'single_col_regression': # More than 1 target col -> single_col_reg
                raise Exception("Invalid number of target columns")

            if self.no_targets < 2 and self.problem_type == 'multi_col_regression': # 1 target col -> multi_col_reg
                raise Exception("Invalid number of target columns")

            k_fold = model_selection.KFold(n_splits=self.no_folds)
            for fold, (train_idx, val_idx) in enumerate(k_fold.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        # Problem 3:
        # --> "holdout": just keep few last part of data for validation -> time series data
        elif self.problem_type.startswith('holdout_'):
            houldout_percentage = int(self.problem_type.split('_')[1])
            no_holdout_samples = int(len(self.dataframe) * (houldout_percentage/100))
            print(no_holdout_samples)
            self.dataframe.loc[:len(self.dataframe) - no_holdout_samples, 'kfold'] = 0
            self.dataframe.loc[len(self.dataframe) - no_holdout_samples:, 'kfold'] = 1    # this part is to holdout

        # Problem 4:
        # --> "multi label classification"
        elif self.problem_type == 'multi_label_classification':
            if self.no_targets != 1:
                raise Exception("Invalid number of target columns")

            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))

            # Create Fold and Train/Val indices
            sk_fold = model_selection.StratifiedKFold(n_splits=self.no_folds,
                                                      shuffle=False,
                                                      random_state=self.random_state)  # Random state has no effect if set to False
            for fold, (train_index, val_index) in enumerate(sk_fold.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_index, 'kfold'] = fold

        # Case not matching
        else:
            raise Exception("Problem type not understood")


        return self.dataframe


if __name__ == '__main__':
    df = pd.read_csv("../input/imet_train.csv")
    cv = CrossValidation(df,
                         shuffle=True,
                         target_cols=['attribute_ids'],
                         problem_type='multi_label_classification',
                         multilabel_delimiter=" ")

    df_split = cv.split()
    print(df_split.head())
    print(df_split['kfold'].value_counts())
