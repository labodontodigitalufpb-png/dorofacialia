from services.datasetService import dataset_completo
import numpy as np
from services.config import COL_CLASSE, COL_GRUPO, COL_SPLIT, COL_TEXTO
from services.hierarchical_model import HierarchicalTextClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC



#Passo 1 - 
# Buscar os dados - X, Y

#Passo 2 - 
# Separar os dados em treino (75%) (X_treino, Y_treino) e teste (25%) (X_teste, Y_teste)


#Passo 3 - 
# Treinar o modelo com os dados de treino

#Passo 4 - 
# Avaliar o modelo com os dados de teste (Acurácia)

def buscar_dados():
    df = dataset_completo()
    X = df[COL_TEXTO].astype(str)
    Y = df[COL_CLASSE].astype(str)
    return X, Y


def criar_features_texto():
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ]
    )


def criar_classificador_sgd(alpha=1e-5):
    return SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        max_iter=3000,
        tol=1e-4,
        class_weight="balanced",
        random_state=42,
    )

def separar_dados():
    
    #buscando os dadps
    df = dataset_completo()
    if COL_SPLIT in df.columns:
        treino = df[df[COL_SPLIT].astype(str).str.lower() == "treino"]
        teste = df[df[COL_SPLIT].astype(str).str.lower().isin(["validacao", "validação"])]
        if teste.empty:
            teste = df[df[COL_SPLIT].astype(str).str.lower() == "teste"]
        if not treino.empty and not teste.empty:
            return (
                treino[COL_TEXTO].astype(str),
                teste[COL_TEXTO].astype(str),
                treino[COL_CLASSE].astype(str),
                teste[COL_CLASSE].astype(str),
            )

    X, Y = buscar_dados()

    # Separar os dados em treino (70%) e teste (30%)
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=42,
        stratify=Y,
    )
    return X_treino, X_teste, Y_treino, Y_teste

def treinar_modelo():
    """
    Função para treinar o modelo de Machine Learning para dor orofacial.

    Return:
    model: Pipeline
        O modelo treinado - que se chama DorOrofacialAI.
    """
    df = dataset_completo()
    if COL_GRUPO in df.columns and COL_SPLIT in df.columns:
        treino = df[df[COL_SPLIT].astype(str).str.lower() == "treino"]
        group_model = Pipeline(
            steps=[
                ("features", criar_features_texto()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        LinearSVC(
                            C=1.5,
                            class_weight="balanced",
                            dual="auto",
                            random_state=42,
                        ),
                        cv=5,
                    ),
                ),
            ]
        )
        diagnosis_model = Pipeline(
            steps=[
                ("features", criar_features_texto()),
                ("classifier", criar_classificador_sgd(alpha=1e-5)),
            ]
        )
        DorOrofacialAI = HierarchicalTextClassifier(
            group_model=group_model,
            diagnosis_model=diagnosis_model,
        )
        DorOrofacialAI.fit(
            treino[COL_TEXTO].astype(str),
            treino[COL_CLASSE].astype(str),
            treino[COL_GRUPO].astype(str),
        )
        return DorOrofacialAI

    X_treino, _, Y_treino, _ = separar_dados()
    DorOrofacialAI = Pipeline(
        steps=[
            ("features", criar_features_texto()),
            ("classifier", criar_classificador_sgd(alpha=1e-5)),
        ]
    )
    DorOrofacialAI.fit(X_treino, Y_treino)

    return DorOrofacialAI

def acuracia_modelo():

    DorOrofacialAI = treinar_modelo()
    _, X_teste, _, Y_teste = separar_dados()

   
    Y_pred = DorOrofacialAI.predict(X_teste)
    acuracia = accuracy_score(Y_teste, Y_pred)

    porcentagem = acuracia * 100

    return porcentagem 


def top3_modelo():
    DorOrofacialAI = treinar_modelo()
    _, X_teste, _, Y_teste = separar_dados()
    probs = DorOrofacialAI.predict_proba(X_teste)
    classes = np.array(DorOrofacialAI.classes_)
    indices = np.argsort(probs, axis=1)[:, -3:]
    top3 = classes[indices]
    return float(np.mean([y in linha for y, linha in zip(Y_teste.astype(str), top3)]) * 100)
