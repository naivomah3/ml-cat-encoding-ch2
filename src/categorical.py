from sklearn import preprocessing

"""
Different approaches:
    - label encoding
    - one hot encoding
    - binarization
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: DataFrame
        categorical_features: list of column names e.g.["ord_1", "nom_0", ...]
        encoding_type: label, onehot, binary
        handle_na: True/False
        """
        self.df = df
        self.output_df = self.df.copy(deep=True)
        self.categorical_features = categorical_features
        self.enc_type =  encoding_type
        self.label_encoders = dict()

        # basic imputation: works for LabelEncoding
        if handle_na:
            for col in self.categorical_features:
                self.df.loc[:, col] = self.df.loc[:, col].astype(str).fillna("-9999999")

    def _label_encoding(self):
        for col in self.categorical_features:
            label = preprocessing.LabelEncoder()
            label.fit(self.df[col].values)
            self.output_df.loc[:, col] = label.transform(self.df[col].values)
            self.label_encoders[col] = label
        return self.output_df

    def _label_binarization(self):
        for col in self.categorical_features:
            label = preprocessing.LabelBinarizer()
            label.fit(self.df[col].values)


    def transform(self):
        if self.enc_type == 'label':
            return self._label_encoding()
        else:
            raise Exception("Encoding type not understood")


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../input/categorical_2_train.csv")
    train_cols = [c for c in df.columns if c not in ['id', 'target']]
    print(train_cols)
    cat_features = CategoricalFeatures(df,
                                       categorical_features=train_cols,
                                       encoding_type='label',
                                       handle_na=True)

    output_df = cat_features.transform()

    print(output_df.head())
