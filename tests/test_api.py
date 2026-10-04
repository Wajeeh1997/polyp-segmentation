import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from polypseg.api import create_app


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def client(tiny_checkpoint, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_checkpoint))
    monkeypatch.setenv("DEVICE", "cpu")
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_mask_and_overlay(client, sample_image):
    response = client.post(
        "/predict",
        files={"file": ("polyp.jpg", _jpeg_bytes(sample_image), "image/jpeg")},
        params={"threshold": 0.3},
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["width"], body["height"]) == sample_image.size
    assert 0.0 <= body["mask_area_fraction"] <= 1.0
    mask = Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"])))
    overlay = Image.open(io.BytesIO(base64.b64decode(body["overlay_png_base64"])))
    assert mask.size == overlay.size == sample_image.size


def test_predict_overlay_returns_png(client, sample_image):
    response = client.post(
        "/predict/overlay",
        files={"file": ("polyp.jpg", _jpeg_bytes(sample_image), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == sample_image.size


def test_rejects_non_image_upload(client):
    response = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_rejects_bad_threshold(client, sample_image):
    response = client.post(
        "/predict",
        files={"file": ("polyp.jpg", _jpeg_bytes(sample_image), "image/jpeg")},
        params={"threshold": 1.5},
    )
    assert response.status_code == 422


def test_service_reports_missing_model(monkeypatch, tmp_path, sample_image):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    with TestClient(create_app()) as test_client:
        assert test_client.get("/health").status_code == 503
        response = test_client.post(
            "/predict", files={"file": ("polyp.jpg", _jpeg_bytes(sample_image), "image/jpeg")}
        )
        assert response.status_code == 503
