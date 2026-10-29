"""The service API: auth, validation, model scoring and the gateway over HTTP."""

import numpy as np
from fastapi.testclient import TestClient

from cpa.ml import features
from cpa.service.app import create_app
from tests.test_models import table


class FakeServed:
    version = "7"
    classes = ["reset", "self_resolved"]

    def __init__(self):
        from cpa.ml.models import Encoder

        self.encoder = Encoder.fit(table(), extra=["age"])

    def predict(self, df):
        return np.tile([0.7, 0.3], (df.height, 1))


class FakeRegistry:
    def alias(self, kind, alias):
        return "7" if kind == "diagnosis" else None

    def load(self, kind, v):
        return FakeServed()


def client(monkeypatch=None, token=None):
    if token and monkeypatch:
        monkeypatch.setenv("CPA_API_TOKEN", token)
    return TestClient(create_app(FakeRegistry()))


def row():
    return {c: 0.0 for c in features.NUMERIC} | {"age": 0.0, "model": "DC-150"}


def test_health_names_the_versions():
    assert client().get("/health").json()["models"] == {"diagnosis": "7"}


def test_diagnose_scores_rows_and_validates():
    c = client()
    r = c.post("/v1/diagnose", json={"rows": [row(), row()]})
    assert r.status_code == 200 and r.json()["probabilities"] == [[0.7, 0.3], [0.7, 0.3]]
    assert c.post("/v1/diagnose", json={"rows": [{"model": "DC-150"}]}).status_code == 422
    assert c.post("/v1/diagnose", json={"rows": []}).status_code == 422
    assert c.post("/v1/risk", json={"rows": [row()]}).status_code == 503  # no risk champion


def test_gateway_over_http_masks_and_refuses_unknown_use_cases():
    c = client()
    r = c.post(
        "/v1/ai/incident-reader",
        headers={"x-caller": "team-app"},
        json={
            "fields": {
                "model": "DC-150",
                "tech_code": "",
                "tech_info": "",
                "complaints": ["dead charger, call +49 171 2345678"],
                "passages": [],
            }
        },
    )
    assert r.status_code == 200 and r.json()["guard"]["masked"] == {"phone": 1}
    assert c.post("/v1/ai/free-chat", json={"fields": {}}).status_code == 404
    m = c.get("/metrics").text
    assert 'cpa_requests_total{method="POST",path="/v1/ai/{use_case}",status="200"} 1' in m


def test_bearer_token_when_configured(monkeypatch):
    c = client(monkeypatch, token="s3cret")
    assert c.get("/v1/use-cases").status_code == 401
    assert c.get("/v1/use-cases", headers={"authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/health").status_code == 200
