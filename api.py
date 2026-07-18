"""
API web de trainedml (FastAPI).

Expose le workflow trainedml en HTTP pour n'importe quel frontend
(HTML/JS, React, mobile...) :

- ``GET  /api/models``  : modèles disponibles (classificateurs et régresseurs)
- ``POST /api/train``   : entraîne un modèle et retourne ses scores
- ``POST /api/predict`` : prédit avec le dernier modèle entraîné
- ``POST /api/compare`` : compare tous les modèles adaptés (validation croisée)

La page de démo (static/index.html) est servie à la racine ``/``.

Lancement
---------
    pip install -r requirements.txt
    uvicorn api:app --reload
    # puis ouvrir http://localhost:8000

Le modèle entraîné est conservé en mémoire du serveur (un seul à la fois,
volontairement simple pour la démo). Pour un vrai déploiement : sauvegarder
avec Trainer.save() et charger au démarrage avec Trainer.load().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trainedml import CLASSIFIER_MAP, REGRESSOR_MAP, Trainer, compare

app = FastAPI(
    title="trainedml API",
    description="Entraîner, évaluer, prédire et comparer des modèles ML via HTTP.",
    version="0.1.0",
)

# CORS ouvert : permet à un frontend servi ailleurs (React en dev sur :5173,
# par exemple) d'appeler l'API. À restreindre en production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# État du serveur : le dernier Trainer entraîné.
STATE: Dict[str, Any] = {"trainer": None}


class TrainRequest(BaseModel):
    """Corps de la requête POST /api/train."""
    dataset: Optional[str] = Field(None, description="Dataset intégré : iris ou wine")
    url: Optional[str] = Field(None, description="URL d'un CSV distant")
    target: Optional[str] = Field(None, description="Colonne cible (si url)")
    model: str = Field("random_forest", description="Nom du modèle trainedml")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Hyperparamètres du modèle")
    test_size: float = Field(0.2, gt=0, lt=1)
    seed: int = 42


class PredictRequest(BaseModel):
    """Corps de la requête POST /api/predict."""
    features: List[List[Union[float, int, str]]] = Field(
        ..., description="Lignes à prédire, mêmes colonnes que l'entraînement"
    )


class CompareRequest(BaseModel):
    """Corps de la requête POST /api/compare."""
    dataset: Optional[str] = None
    url: Optional[str] = None
    target: Optional[str] = None
    cv: int = Field(5, ge=2, le=20)
    seed: int = 42


@app.get("/api/models")
def list_models() -> Dict[str, List[str]]:
    """Liste les modèles disponibles par type de tâche."""
    return {
        "classifiers": list(CLASSIFIER_MAP.keys()),
        "regressors": list(REGRESSOR_MAP.keys()),
    }


@app.post("/api/train")
def train(req: TrainRequest) -> Dict[str, Any]:
    """Entraîne un modèle et retourne ses scores sur le jeu de test."""
    if req.dataset is None and req.url is None:
        raise HTTPException(400, "Fournir dataset ou url + target.")
    try:
        trainer = Trainer(
            dataset=req.dataset, url=req.url, target=req.target,
            model=req.model, model_params=req.model_params,
            test_size=req.test_size, seed=req.seed,
        )
        trainer.fit()
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    STATE["trainer"] = trainer
    return {
        "model": trainer.model_name,
        "task": trainer.task,
        "scores": trainer.evaluate(),
        "feature_names": trainer.feature_names_,
        "n_train": len(trainer.X_train),
        "n_test": len(trainer.X_test),
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """Prédit avec le dernier modèle entraîné via /api/train."""
    trainer: Optional[Trainer] = STATE["trainer"]
    if trainer is None:
        raise HTTPException(409, "Aucun modèle entraîné : appeler /api/train d'abord.")
    try:
        preds = trainer.predict(req.features)
    except Exception as e:
        raise HTTPException(400, f"Prédiction impossible : {e}")
    return {"model": trainer.model_name, "predictions": [str(p) for p in preds]}


@app.post("/api/compare")
def compare_models(req: CompareRequest) -> Dict[str, Any]:
    """Compare tous les modèles adaptés au dataset (validation croisée)."""
    if req.dataset is None and req.url is None:
        raise HTTPException(400, "Fournir dataset ou url + target.")
    try:
        df = compare(
            dataset=req.dataset, url=req.url, target=req.target,
            cv=req.cv, seed=req.seed, show_progress=False,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    df = df.round(4).reset_index()
    return {"cv": req.cv, "results": df.to_dict(orient="records")}


# La page de démo est servie en dernier pour ne pas masquer les routes /api/*.
_static = Path(__file__).parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
