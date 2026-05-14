from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

APP_NAME = "Dor Orofacial AI Helper"
MODEL_NAME = "DorOrofacialAI"

DATASET_PATH = BASE_DIR / "banco_Dor_Orofacial_IA_V3_hierarquico_10000_42classes.csv"
COL_TEXTO = "perfil_semio_estruturado"
COL_CLASSE = "target_diagnostico"
COL_GRUPO = "target_grupo_clinico"
COL_SPLIT = "split_estratificado"

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / f"modelo_{MODEL_NAME}.pkl"
VECTORIZER_PATH = MODEL_DIR / f"vetorizador_{MODEL_NAME}.pkl"
ENCODER_PATH = MODEL_DIR / f"encoderY_{MODEL_NAME}.pkl"

MODEL_ACCURACY = 68.25
MODEL_TOP3_ACCURACY = 88.16
