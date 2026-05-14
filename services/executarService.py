
import joblib
import numpy as np
from pathlib import Path
from services.config import ENCODER_PATH, MODEL_ACCURACY, MODEL_NAME, MODEL_PATH, MODEL_TOP3_ACCURACY, VECTORIZER_PATH
from services.datasetService import condutas_por_diagnostico


class DiagnosticoIA:
    def __init__(
        self,
        model_path: str = str(MODEL_PATH),
        vectorizer_path: str = str(VECTORIZER_PATH),
        encoder_path: str = str(ENCODER_PATH),
    ):
        self.model = joblib.load(model_path)
        self.vet = joblib.load(vectorizer_path) if Path(vectorizer_path).exists() else None
        self.enc = joblib.load(encoder_path) if Path(encoder_path).exists() else None
        self.condutas = condutas_por_diagnostico()

    def _scores_por_classe(self, texto):
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba([texto])[0]
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function([texto])[0]
            scores = np.asarray(scores, dtype=float)
            scores = scores - np.max(scores)
            scores = np.exp(scores)
            scores = scores / scores.sum()
        else:
            previsto = self.model.predict([texto])[0]
            return {str(previsto): 1.0}

        return {
            str(label): float(score)
            for label, score in zip(self.model.classes_, scores)
        }

    def _conduta_para_hipoteses(self, top3):
        if not top3:
            return None, []

        condutas_hipoteses = []
        for hipotese in top3:
            diagnostico = hipotese["doenca"]
            conduta = self.condutas.get(diagnostico, {})
            if conduta:
                condutas_hipoteses.append({
                    "doenca": diagnostico,
                    "prob_percent": hipotese["prob_percent"],
                    **conduta,
                })

        principal = condutas_hipoteses[0] if condutas_hipoteses else None
        return principal, condutas_hipoteses

    def _grupo_provavel(self, texto):
        if not hasattr(self.model, "predict_group_proba"):
            return None

        probs, classes = self.model.predict_group_proba([texto])
        ordenados = sorted(
            zip(classes, probs[0]),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"grupo": str(grupo), "prob_percent": round(float(prob) * 100, 2)}
            for grupo, prob in ordenados[:3]
        ]

    def predict_simples(self, sintomas_list):
        """
        Retorna Top-3 hipóteses com percentuais (probabilidades reais do modelo).
        """
        sintomas_list = [str(s).strip() for s in (sintomas_list or []) if str(s).strip()]
        if not sintomas_list:
            return {
                "sintomas_recebidos": [],
                "diagnostico_provavel": "Nenhum sintoma informado",
                "top3": [],
                "top2": [],
                "conduta_proposta": None,
                "condutas_por_hipotese": [],
                "acuracia_modelo": MODEL_ACCURACY,
                "modelo": "TF-IDF + ComplementNB",
            }

        texto = ", ".join(sintomas_list)
        scores_modelo = self._scores_por_classe(texto)
        grupos_provaveis = self._grupo_provavel(texto)
        top_modelo = sorted(scores_modelo.items(), key=lambda item: item[1], reverse=True)[:3]

        top3 = []
        for lbl, score in top_modelo:
            top3.append({
                "doenca": str(lbl),
                "prob_percent": round(float(score) * 100, 2),
                "prob_modelo_percent": round(scores_modelo.get(str(lbl), 0) * 100, 2),
            })

        diagnostico_previsto = top3[0]["doenca"] if top3 else "Classe desconhecida"
        conduta_proposta, condutas_por_hipotese = self._conduta_para_hipoteses(top3)

        return {
            "sintomas_recebidos": sintomas_list,
            "diagnostico_provavel": str(diagnostico_previsto),
            "top3": top3,
            "top2": top3[:2],
            "grupos_provaveis": grupos_provaveis,
            "conduta_proposta": conduta_proposta,
            "condutas_por_hipotese": condutas_por_hipotese,
            "explicacoes_regras": [],
            "acuracia_modelo": MODEL_ACCURACY,
            "top3_acuracia_modelo": MODEL_TOP3_ACCURACY,
            "modelo": "Hierárquico: grupo clínico calibrado + diagnóstico intra-grupo",
            "nome_modelo": MODEL_NAME,
        }
