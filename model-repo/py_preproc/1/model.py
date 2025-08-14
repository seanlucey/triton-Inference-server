# Triton Python backend preprocessor
# Receives raw image bytes (or base64) and outputs FP32 normalized tensor [3,224,224]

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
import io, base64

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "raw")
            raw = in_tensor.as_numpy()[0]
            if raw.dtype.type is np.bytes_ or raw.dtype == np.object_:
                raw_bytes = raw.tobytes() if hasattr(raw, 'tobytes') else raw.item()
            else:
                raw_bytes = raw
            # if base64-encoded
            try:
                raw_bytes = base64.b64decode(raw_bytes)
            except Exception:
                pass
            img = Image.open(io.BytesIO(raw_bytes)).convert('RGB').resize((224, 224))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC→CHW
            out = np.expand_dims(arr, axis=0)  # add batch dim if needed elsewhere
            tensor = pb_utils.Tensor("input", out)
            responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))
        return responses
