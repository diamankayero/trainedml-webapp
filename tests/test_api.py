"""
Tests de l'API webapp (api.py) avec le TestClient FastAPI.
"""
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import STATE, app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        STATE["trainer"] = None

    def test_models(self):
        data = self.client.get("/api/models").json()
        self.assertIn("knn", data["classifiers"])
        self.assertIn("ridge", data["regressors"])

    def test_train_then_predict(self):
        res = self.client.post("/api/train", json={"dataset": "iris", "model": "knn"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["task"], "classification")
        self.assertGreater(data["scores"]["accuracy"], 0.8)

        res = self.client.post("/api/predict", json={"features": [[5.1, 3.5, 1.4, 0.2]]})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["predictions"]), 1)

    def test_predict_without_train_is_409(self):
        res = self.client.post("/api/predict", json={"features": [[1, 2, 3, 4]]})
        self.assertEqual(res.status_code, 409)

    def test_train_unknown_model_is_400(self):
        res = self.client.post("/api/train", json={"dataset": "iris", "model": "inexistant"})
        self.assertEqual(res.status_code, 400)

    def test_train_without_data_is_400(self):
        res = self.client.post("/api/train", json={"model": "knn"})
        self.assertEqual(res.status_code, 400)

    def test_compare(self):
        res = self.client.post("/api/compare", json={"dataset": "iris", "cv": 2})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["cv"], 2)
        self.assertEqual(len(data["results"]), 3)
        self.assertIn("accuracy", data["results"][0])

    def test_demo_page_served(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("trainedml", res.text)


if __name__ == "__main__":
    unittest.main()
