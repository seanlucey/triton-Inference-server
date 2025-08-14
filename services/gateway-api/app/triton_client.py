import numpy as np
import requests
import os

TRITON_URL = os.getenv("TRITON_URL", "http://triton-server:8000")
MODEL = os.getenv("TRITON_MODEL", "ensemble_resnet")

class Triton:
    def __init__(self):
        self.url = TRITON_URL.rstrip("/")
        self.model = MODEL

    def run_ensemble(self, image_bytes: bytes, topk: int = 5):
        # HTTP/JSON V2 protocol
        inputs = [
            {
                "name": "raw",
                "shape": [1],
                "datatype": "BYTES",
                "data": [image_bytes.decode('latin1')]
            },
            {
                "name": "topk",
                "shape": [1],
                "datatype": "INT32",
                "data": [topk]
            }
        ]
        outputs = [
            {"name": "indices"},
            {"name": "scores"}
        ]
        resp = requests.post(f"{self.url}/v2/models/{self.model}/infer", json={"inputs": inputs, "outputs": outputs}, timeout=20)
        resp.raise_for_status()
        out = resp.json()["outputs"]
        indices = np.array(out[0]["data"], dtype=np.int32)
        scores = np.array(out[1]["data"], dtype=np.float32)
        return indices, scores
