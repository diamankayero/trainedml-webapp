# trainedml-webapp

Démo web du package [trainedml](https://github.com/diamankayero/trainedml) :
une API FastAPI et une page HTML/JS qui la consomme.

**En ligne : https://trainedml.onrender.com** (plan gratuit Render : la
première visite après une période d'inactivité prend ~30 s, le temps que le
service se réveille).

Une version **React** de la même interface existe :
[ModeLmL](https://github.com/diamankayero/ModeLmL), en ligne sur
https://diamankayero.github.io/ModeLmL/. La page HTML de ce dépôt reste la
version la plus lisible pour apprendre (un seul fichier, zéro build).

Ce dépôt est volontairement séparé du package : il **consomme** `trainedml`
depuis PyPI comme n'importe quel utilisateur (`requirements.txt`), ce qui en
fait aussi un test d'intégration permanent du package publié. Le pattern :
le Python tourne côté serveur, le frontend (HTML/JS ici, React demain) ne
fait que des appels HTTP ; changer de frontend ne touche pas à l'API.

## Lancement local

```bash
pip install -r requirements.txt
uvicorn api:app --reload
# page de démo : http://localhost:8000
# doc interactive de l'API : http://localhost:8000/docs
```

## Fichiers

- `api.py` : l'application FastAPI et ses routes
- `static/index.html` : page de démo autonome (HTML + CSS + JS vanilla,
  aucune dépendance ni build), mode sombre natif
- `tests/` : tests des routes avec le TestClient FastAPI
- `render.yaml`, `Dockerfile` : déploiement

## Routes

| Route | Corps | Retour |
|---|---|---|
| `GET /api/models` | - | classificateurs et régresseurs disponibles |
| `POST /api/train` | dataset ou url+target ou data+target, model, model_params, test_size, seed | scores, tâche, features |
| `POST /api/predict` | features (lignes à prédire) | prédictions du dernier modèle entraîné |
| `POST /api/compare` | dataset ou url+target ou data+target, models, cv, seed | tableau comparatif (validation croisée) |
| `GET /api/dataset` | name ou url+target (query) | lignes (plafonnées), describe, moyennes, classes |
| `POST /api/analysis` | dataset ou url+target ou data+target | analyse en chiffres : manquants, histogrammes, corrélations, outliers, normalité, cible |
| `POST /api/report` | dataset ou url+target ou data, title | rapport EDA HTML auto-contenu (export) |
| `GET /docs` | - | documentation interactive générée par FastAPI (Swagger) |

## Déploiement

### Render (config dans render.yaml)

New > Blueprint > ce repo : Render lit `render.yaml` et crée le service.
Chaque push sur main redéploie automatiquement.

### Docker (tout hébergeur)

```bash
docker build -t trainedml-webapp .
docker run -p 8000:8000 trainedml-webapp
```

L'hébergeur fournit la variable `PORT` ; l'image l'utilise automatiquement.

## Limites assumées (démo)

- Un seul modèle en mémoire à la fois ; pas de persistance entre
  redémarrages. Pour un vrai déploiement : `Trainer.save()` au train et
  `Trainer.load()` au démarrage, ou un stockage par utilisateur.
- CORS ouvert à tous les domaines ; à restreindre en production.
- Pas d'authentification.

## Licence

MIT, comme le package.
