import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from services.config import COL_CLASSE, COL_TEXTO
from services.datasetService import dataset_completo


def topk_accuracy(model, X, y, k=3):
    classes = np.array(model.classes_)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
    else:
        return accuracy_score(y, model.predict(X))

    indices = np.argsort(scores, axis=1)[:, -k:]
    topk = classes[indices]
    return float(np.mean([truth in row for truth, row in zip(np.asarray(y), topk)]))


def word_features(ngram_range=(1, 3)):
    return TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True, min_df=1)


def mixed_features():
    return FeatureUnion(
        [
            ("word", word_features((1, 3))),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
        ]
    )


def modelos_candidatos():
    return [
        ("ComplementNB word12", word_features((1, 2)), ComplementNB(alpha=0.2)),
        ("ComplementNB word13", word_features((1, 3)), ComplementNB(alpha=0.2)),
        ("MultinomialNB word13", word_features((1, 3)), MultinomialNB(alpha=0.05)),
        (
            "LogReg word13",
            word_features((1, 3)),
            LogisticRegression(max_iter=2000, C=4, class_weight="balanced", solver="liblinear"),
        ),
        (
            "LogReg mix",
            mixed_features(),
            LogisticRegression(max_iter=2000, C=4, class_weight="balanced", solver="liblinear"),
        ),
        (
            "SGD log_loss mix",
            mixed_features(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                max_iter=3000,
                tol=1e-4,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "Calibrated LinearSVC mix",
            mixed_features(),
            CalibratedClassifierCV(
                LinearSVC(C=1.5, class_weight="balanced", dual="auto", random_state=42),
                cv=3,
            ),
        ),
        ("Ridge word13", word_features((1, 3)), RidgeClassifier(class_weight="balanced")),
    ]


def main():
    df = dataset_completo()
    X = df[COL_TEXTO].astype(str)
    y = df[COL_CLASSE].astype(str)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    resultados = []
    for nome, features, classifier in modelos_candidatos():
        model = Pipeline([("features", features), ("classifier", classifier)])
        model.fit(X_treino, y_treino)
        y_pred = model.predict(X_teste)
        resultados.append(
            {
                "modelo": nome,
                "top1": accuracy_score(y_teste, y_pred),
                "balanced": balanced_accuracy_score(y_teste, y_pred),
                "f1_macro": f1_score(y_teste, y_pred, average="macro"),
                "top3": topk_accuracy(model, X_teste, y_teste, k=3),
            }
        )

    resultados.sort(key=lambda item: (item["top1"], item["top3"]), reverse=True)

    for item in resultados:
        print(
            f"{item['modelo']:<28}"
            f" top1={item['top1'] * 100:5.2f}%"
            f" top3={item['top3'] * 100:5.2f}%"
            f" balanced={item['balanced'] * 100:5.2f}%"
            f" f1_macro={item['f1_macro'] * 100:5.2f}%"
        )


if __name__ == "__main__":
    main()
