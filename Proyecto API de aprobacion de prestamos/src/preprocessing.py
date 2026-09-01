from sklearn.base import BaseEstimator, TransformerMixin


class CreditPreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["cuota_ingreso"] = (
            X["cuota_estimada"] /
            X["ingreso_mensual"]
        )

        variables = [
            "personas_a_cargo",
            "ingreso_mensual",
            "score_crediticio",
            "dti_previo",
            "cuota_ingreso"
        ]

        return X[variables]