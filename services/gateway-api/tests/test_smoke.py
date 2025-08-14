import os
import requests


TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000")


def test_triton_health_ready():
    """Checks that Triton is live and ready for inference."""
    live = requests.get(f"{TRITON_URL}/v2/health/live")
    ready = requests.get(f"{TRITON_URL}/v2/health/ready")

    assert live.status_code == 200, f"Triton live check failed: {live.text}"
    assert ready.status_code == 200, f"Triton ready check failed: {ready.text}"


def test_model_ready():
    """Checks that a specific model is loaded and ready."""
    model_name = os.getenv("TEST_MODEL_NAME", "my_model")
    resp = requests.get(f"{TRITON_URL}/v2/models/{model_name}")
    assert resp.status_code == 200, f"Model {model_name} not ready: {resp.text}"
