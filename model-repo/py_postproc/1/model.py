import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            probs = pb_utils.get_input_tensor_by_name(request, "probabilities").as_numpy()
            topk = int(pb_utils.get_input_tensor_by_name(request, "topk").as_numpy()[0]) if pb_utils.get_input_tensor_by_name(request, "topk") else 5
            idx = np.argsort(-probs, axis=1)[:, :topk]
            scores = np.take_along_axis(probs, idx, axis=1)
            # Return indices + scores; map to labels in gateway
            tensors = [
                pb_utils.Tensor("indices", idx.astype(np.int32)),
                pb_utils.Tensor("scores", scores.astype(np.float32)),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=tensors))
        return responses
