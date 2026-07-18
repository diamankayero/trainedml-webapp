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

    def test_dataset(self):
        res = self.client.get("/api/dataset", params={"name": "iris", "limit": 10})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["target"], "species")
        self.assertEqual(len(data["rows"]), 10)
        self.assertEqual(data["n_rows"], 150)
        self.assertIn("sepal_length", data["means"])
        self.assertIn("setosa", data["classes"])

    def test_train_with_uploaded_data(self):
        rows = [
            {"a": i, "b": i * 2, "cible": "x" if i % 2 else "y"}
            for i in range(40)
        ]
        res = self.client.post("/api/train", json={
            "data": rows, "target": "cible", "model": "knn",
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["task"], "classification")

    def test_compare_subset_of_models(self):
        res = self.client.post("/api/compare", json={
            "dataset": "iris", "cv": 2, "models": ["knn", "logistic"],
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["results"]), 2)

    def test_report(self):
        res = self.client.post("/api/report", json={"dataset": "iris", "title": "Test EDA"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Test EDA", res.text)
        self.assertIn("data:image/png;base64", res.text)

    def test_analysis(self):
        res = self.client.post("/api/analysis", json={"dataset": "iris"})
        self.assertEqual(res.status_code, 200)
        d = res.json()
        self.assertEqual(d["n_rows"], 150)
        self.assertEqual(len(d["correlation"]["columns"]), 4)
        self.assertEqual(len(d["histograms"]), 4)
        self.assertEqual(len(d["histograms"][0]["counts"]), 12)
        self.assertEqual(d["target"]["kind"], "classes")
        self.assertEqual(len(d["target"]["items"]), 3)
        self.assertTrue(all("p_value" in n for n in d["normality"]))

    def test_train_returns_importances(self):
        res = self.client.post("/api/train", json={"dataset": "iris", "model": "random_forest"})
        imp = res.json()["importances"]
        self.assertEqual(len(imp), 4)
        self.assertAlmostEqual(sum(i["importance"] for i in imp), 1.0, places=1)
        res = self.client.post("/api/train", json={"dataset": "iris", "model": "knn"})
        self.assertIsNone(res.json()["importances"])


if __name__ == "__main__":
    unittest.main()
