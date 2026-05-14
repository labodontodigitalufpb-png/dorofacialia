import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier


class HierarchicalTextClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador hierárquico para dor orofacial.

    Etapa 1: prediz o grupo clínico.
    Etapa 2: prediz o diagnóstico dentro de cada grupo.
    A probabilidade final é P(grupo) * P(diagnóstico | grupo).
    """

    def __init__(self, group_model, diagnosis_model):
        self.group_model = group_model
        self.diagnosis_model = diagnosis_model

    def fit(self, X, y, groups):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=object)
        groups = np.asarray(groups, dtype=object)

        self.group_model_ = clone(self.group_model)
        self.group_model_.fit(X, groups)

        self.classes_ = np.array(sorted({str(label) for label in y}), dtype=object)
        self.groups_ = np.array(sorted({str(group) for group in groups}), dtype=object)
        self.group_to_classes_ = {}
        self.diagnosis_models_ = {}

        for group in self.groups_:
            mask = groups == group
            labels = np.array([str(label) for label in y[mask]], dtype=object)
            classes = np.array(sorted(set(labels)), dtype=object)
            self.group_to_classes_[str(group)] = classes

            if len(classes) == 1:
                model = DummyClassifier(strategy="constant", constant=classes[0])
            else:
                model = clone(self.diagnosis_model)

            model.fit(X[mask], labels)
            self.diagnosis_models_[str(group)] = model

        return self

    def _probabilities(self, model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)

        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict_group_proba(self, X):
        X = np.asarray(X, dtype=object)
        probs = self._probabilities(self.group_model_, X)
        return probs, np.array(self.group_model_.classes_, dtype=object)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=object)
        grupo_probs, grupo_classes = self.predict_group_proba(X)
        saida = np.zeros((len(X), len(self.classes_)), dtype=float)
        class_index = {classe: idx for idx, classe in enumerate(self.classes_)}

        for grupo_idx, grupo in enumerate(grupo_classes):
            grupo = str(grupo)
            model = self.diagnosis_models_[grupo]
            local_probs = self._probabilities(model, X)

            for local_idx, classe in enumerate(model.classes_):
                saida[:, class_index[str(classe)]] += grupo_probs[:, grupo_idx] * local_probs[:, local_idx]

        soma = saida.sum(axis=1, keepdims=True)
        return np.divide(saida, soma, out=np.zeros_like(saida), where=soma != 0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
