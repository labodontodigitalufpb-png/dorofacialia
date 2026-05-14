import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


ARQUIVO = "banco_Dor_Orofacial_IA_V2_hierarquico_resumo.xlsx"
ABA_DADOS = "xl/worksheets/sheet2.xml"
NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def coluna_para_indice(referencia):
    letras = "".join(ch for ch in referencia if ch.isalpha())
    indice = 0
    for letra in letras:
        indice = indice * 26 + ord(letra) - 64
    return indice - 1


def valor_celula(celula):
    inline_string = celula.find("a:is", NS)
    if inline_string is not None:
        return "".join(texto.text or "" for texto in inline_string.findall(".//a:t", NS))

    valor = celula.find("a:v", NS)
    return valor.text if valor is not None else ""


def valores_linha(linha):
    valores = []
    for celula in linha.findall("a:c", NS):
        indice = coluna_para_indice(celula.get("r"))
        while len(valores) < indice:
            valores.append("")
        valores.append(valor_celula(celula))
    return valores


def carregar_xlsx_simples(caminho=ARQUIVO):
    with zipfile.ZipFile(caminho) as arquivo:
        raiz = ET.fromstring(arquivo.read(ABA_DADOS))

    linhas = raiz.findall(".//a:sheetData/a:row", NS)
    dados = [valores_linha(linha) for linha in linhas]
    colunas = dados[0]
    registros = [linha + [""] * (len(colunas) - len(linha)) for linha in dados[1:]]
    return pd.DataFrame(registros, columns=colunas)


def modelo_texto(classificador):
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=1)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ]
    )
    return Pipeline([("features", features), ("classifier", classificador)])


def avaliar_textos(df):
    modelos = [
        ("ComplementNB", ComplementNB(alpha=0.2)),
        (
            "SGD log_loss",
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
            "LogReg",
            LogisticRegression(max_iter=2000, C=4, class_weight="balanced", solver="liblinear"),
        ),
        ("LinearSVC", LinearSVC(C=1.5, class_weight="balanced", dual="auto", random_state=42)),
    ]

    textos = [
        "texto_clinico_sem_rotulo",
        "json_entrada_app_minima",
        "json_entrada_app_com_exames",
        "perfil_semio_estruturado",
    ]

    treino = df[df["split_estratificado"] == "treino"]
    teste = df[df["split_estratificado"] == "teste"]
    validacao = df[df["split_estratificado"] == "validacao"]

    y_treino = treino["target_diagnostico"]
    y_teste = teste["target_diagnostico"]
    y_validacao = validacao["target_diagnostico"]

    for texto in textos:
        print(f"\nEntrada: {texto}")
        for nome, classificador in modelos:
            modelo = modelo_texto(classificador)
            modelo.fit(treino[texto].astype(str), y_treino)
            pred_teste = modelo.predict(teste[texto].astype(str))
            pred_validacao = modelo.predict(validacao[texto].astype(str))
            print(
                f"{nome:<14}"
                f" teste={accuracy_score(y_teste, pred_teste) * 100:5.2f}%"
                f" validacao={accuracy_score(y_validacao, pred_validacao) * 100:5.2f}%"
                f" balanced={balanced_accuracy_score(y_teste, pred_teste) * 100:5.2f}%"
                f" f1_macro={f1_score(y_teste, pred_teste, average='macro') * 100:5.2f}%"
            )


def main():
    df = carregar_xlsx_simples()

    print("Linhas/colunas:", df.shape)
    print("Splits:", df["split_estratificado"].value_counts().to_dict())
    print("Grupos:", df["target_grupo_clinico"].nunique())
    print("Diagnosticos:", df["target_diagnostico"].nunique())
    print("Distribuicao diagnosticos:")
    print(df["target_diagnostico"].value_counts().to_string())

    avaliar_textos(df)


if __name__ == "__main__":
    main()
