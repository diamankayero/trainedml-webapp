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

import matplotlib
matplotlib.use("Agg")  # rendu des figures sans écran (serveur)

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trainedml import CLASSIFIER_MAP, REGRESSOR_MAP, DataLoader, Trainer, compare
from trainedml.report import generate_report


def _resolve_data(dataset=None, url=None, target=None, data=None):
    """
    Résout la source de données d'une requête : dataset intégré, URL distante
    ou lignes envoyées par le client (upload). Retourne (X, y).
    """
    if data is not None:
        if not target:
            raise HTTPException(400, "Fournir target (nom de la colonne cible) avec data.")
        df = pd.DataFrame(data)
        if target not in df.columns:
            raise HTTPException(400, f"Colonne cible {target!r} absente des données.")
        return df.drop(columns=[target]), df[target]
    if dataset is None and url is None:
        raise HTTPException(400, "Fournir dataset, url + target, ou data + target.")
    try:
        return DataLoader().load_dataset(name=dataset, url=url, target=target)
    except Exception as e:
        raise HTTPException(400, f"Chargement impossible : {e}")

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
    data: Optional[List[Dict[str, Any]]] = Field(
        None, description="Lignes envoyées par le client (upload), colonne cible incluse")
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
    data: Optional[List[Dict[str, Any]]] = None
    models: Optional[List[str]] = Field(
        None, description="Sous-ensemble de modèles à comparer (défaut : tous ceux de la tâche)")
    cv: int = Field(5, ge=2, le=20)
    seed: int = 42


class ReportRequest(BaseModel):
    """Corps de la requête POST /api/report."""
    dataset: Optional[str] = None
    url: Optional[str] = None
    target: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    title: str = "Rapport exploratoire"


@app.get("/api/models")
def list_models() -> Dict[str, List[str]]:
    """Liste les modèles disponibles par type de tâche."""
    return {
        "classifiers": list(CLASSIFIER_MAP.keys()),
        "regressors": list(REGRESSOR_MAP.keys()),
    }


def _feature_importances(trainer) -> Optional[List[Dict[str, Any]]]:
    """
    Importance des variables du modèle entraîné, si le modèle sait la donner :
    feature_importances_ pour les forêts, |coefficients| pour les modèles
    linéaires et logistiques (moyennés entre classes), rien pour KNN.
    Les importances sont normalisées (somme = 1) et plafonnées au top 15.
    """
    import numpy as np
    try:
        est = getattr(trainer.model, "model", trainer.model)
        if trainer.preprocessor is not None:
            # Noms dans l'espace prétraité (après one-hot), sans le préfixe num__/cat__
            names = [n.split("__", 1)[-1]
                     for n in trainer.preprocessor.get_feature_names_out()]
        else:
            names = list(trainer.feature_names_ or [])
        if hasattr(est, "feature_importances_"):
            vals = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            vals = np.abs(np.atleast_2d(est.coef_)).mean(axis=0)
        else:
            return None
        if len(names) != len(vals):
            return None
        total = float(vals.sum()) or 1.0
        pairs = sorted(zip(names, vals / total), key=lambda p: -p[1])[:15]
        return [{"feature": str(n), "importance": round(float(v), 4)} for n, v in pairs]
    except Exception:
        return None


@app.post("/api/train")
def train(req: TrainRequest) -> Dict[str, Any]:
    """Entraîne un modèle et retourne ses scores sur le jeu de test."""
    X, y = _resolve_data(req.dataset, req.url, req.target, req.data)
    try:
        trainer = Trainer(
            X=X, y=y,
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
        "importances": _feature_importances(trainer),
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
    """Compare les modèles adaptés au dataset (validation croisée)."""
    X, y = _resolve_data(req.dataset, req.url, req.target, req.data)
    models = None
    if req.models:
        all_models = {**CLASSIFIER_MAP, **REGRESSOR_MAP}
        unknown = [m for m in req.models if m not in all_models]
        if unknown:
            raise HTTPException(400, f"Modèles inconnus : {unknown}")
        models = {name: all_models[name]() for name in req.models}
    try:
        df = compare(X=X, y=y, models=models,
                     cv=req.cv, seed=req.seed, show_progress=False)
    except ValueError as e:
        raise HTTPException(400, str(e))
    df = df.round(4).reset_index()
    return {"cv": req.cv, "results": df.to_dict(orient="records")}


@app.get("/api/dataset")
def get_dataset(name: Optional[str] = None, url: Optional[str] = None,
                target: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
    """
    Retourne les données d'un dataset (lignes plafonnées à ``limit``),
    ses statistiques descriptives et les moyennes des variables numériques
    (pour pré-remplir un formulaire de prédiction).
    """
    X, y = _resolve_data(name, url, target)
    df = pd.concat([X, y], axis=1)
    df = df.where(pd.notnull(df), None)
    describe = X.describe().round(3)
    describe.insert(0, "statistique", describe.index)
    return {
        "columns": list(df.columns),
        "feature_names": list(X.columns),
        "target": str(y.name),
        "n_rows": int(len(df)),
        "rows": df.head(limit).to_dict(orient="records"),
        "describe": describe.to_dict(orient="records"),
        "means": {c: round(float(v), 4) for c, v in
                  X.select_dtypes("number").mean().items()},
        "classes": sorted(str(v) for v in y.unique()) if y.nunique() <= 20 else None,
    }


@app.post("/api/analysis")
def analysis(req: ReportRequest) -> Dict[str, Any]:
    """
    Analyse exploratoire en chiffres (pas d'images) : le frontend dessine
    lui-même avec ses propres composants. Contenu :

    - missing : valeurs manquantes par colonne (nombre et pourcentage)
    - histograms : par variable numérique, bords d'intervalles et comptages
    - correlation : matrice de corrélation des variables numériques
    - outliers : nombre d'outliers par variable (méthode IQR)
    - normality : p-value de Shapiro-Wilk par variable
    - target : distribution de la cible (comptages par classe, ou histogramme)
    """
    import numpy as np
    from trainedml.viz.normality import normality_tests
    from trainedml.viz.outliers import outlier_summary

    X, y = _resolve_data(req.dataset, req.url, req.target, req.data)
    numeric = X.select_dtypes("number")

    # Valeurs manquantes par colonne
    df = pd.concat([X, y], axis=1)
    missing = [
        {"column": str(c), "count": int(n), "pct": round(float(n) / len(df) * 100, 2)}
        for c, n in df.isnull().sum().items()
    ]

    # Histogrammes des variables numériques (12 intervalles)
    histograms = []
    for col in numeric.columns:
        values = numeric[col].dropna()
        counts, edges = np.histogram(values, bins=12)
        histograms.append({
            "column": str(col),
            "edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts],
        })

    # Matrice de corrélation
    corr = numeric.corr()
    correlation = {
        "columns": [str(c) for c in corr.columns],
        "values": [[None if pd.isna(v) else round(float(v), 3) for v in row]
                   for row in corr.values],
    }

    # Outliers (IQR) et normalité (Shapiro), avec repli silencieux par colonne
    try:
        # outlier_summary retourne {colonne: valeurs aberrantes} ; on ne garde
        # que le nombre, le détail n'intéresse pas le tableau de bord.
        outliers = [{"column": str(c), "count": int(len(v))}
                    for c, v in outlier_summary(numeric).items()]
    except Exception:
        outliers = []
    normality = []
    try:
        for col, res in normality_tests(numeric, columns=list(numeric.columns)).items():
            stat, pval = res["shapiro"]
            normality.append({"column": str(col), "p_value": round(float(pval), 4),
                              "normal": bool(pval > 0.05)})
    except Exception:
        pass

    # Distribution de la cible : classes ou histogramme selon la tâche
    if y.nunique() <= 20:
        counts = y.astype(str).value_counts()
        target = {"kind": "classes",
                  "items": [{"label": str(k), "count": int(v)} for k, v in counts.items()]}
    else:
        counts, edges = np.histogram(pd.to_numeric(y, errors="coerce").dropna(), bins=12)
        target = {"kind": "histogram", "name": str(y.name),
                  "edges": [round(float(e), 4) for e in edges],
                  "counts": [int(c) for c in counts]}

    return {
        "n_rows": int(len(df)),
        "missing": missing,
        "histograms": histograms,
        "correlation": correlation,
        "outliers": outliers,
        "normality": normality,
        "target": target,
    }


@app.post("/api/report", response_class=HTMLResponse)
def report(req: ReportRequest) -> HTMLResponse:
    """
    Génère le rapport exploratoire HTML auto-contenu de trainedml
    (statistiques, corrélations, distributions, outliers, normalité, VIF).
    """
    X, y = _resolve_data(req.dataset, req.url, req.target, req.data)
    data = pd.concat([X, y], axis=1)
    try:
        html = generate_report(data, title=req.title)
    except Exception as e:
        raise HTTPException(500, f"Génération du rapport impossible : {e}")
    return HTMLResponse(html)


# La page de démo est servie en dernier pour ne pas masquer les routes /api/*.
_static = Path(__file__).parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
