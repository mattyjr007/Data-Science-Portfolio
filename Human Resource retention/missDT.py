from sklearn.base import BaseEstimator, TransformerMixin

class missingDataTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        self.mean = X.mean()
        return self
    
    def transform(self, X, y=None):
        # Perform mean imputation on the data
      #  X = X.fillna(self.mean)
        return X