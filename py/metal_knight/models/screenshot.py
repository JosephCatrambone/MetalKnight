import os

import numpy
import onnxruntime as ort
from PIL import Image

from models.base import BaseMLModel, MODEL_STORAGE_DIR, softmax


class Screenshot(BaseMLModel):
	def __init__(self):
		self.inference_session = None

	def load_model(self):
		self.inference_session = ort.InferenceSession(os.path.join(MODEL_STORAGE_DIR, "screenshot.onnx"))

	def is_model_loaded(self) -> bool:
		return self.inference_session is not None

	def unload_model(self):
		del self.inference_session
		self.inference_session = None

	def get_input_image_size(self) -> tuple[int, int]:
		return 224, 224  # 320, 320 would work, too.

	def get_class_names(self) -> list[str]:
		return ["not_screenshot", "screenshot"]

	def predict_classes(self, image: Image.Image, normalize: bool = True) -> dict[str, float]:
		"""Run inference on an input image, returning a map from class name to probability."""
		tensor = self.prepare_image_as_chw_tensor(image)
		tensor = numpy.expand_dims(tensor, axis=0)
		# session.get_inputs()[0].name and session.get_outputs()[0].name?
		out = self.inference_session.run(["output"], {"input": tensor,})[0][0]
		if normalize:
			out = softmax(out)
		class_preds = {
			"not_screenshot": float(out[0]),
			"screenshot": float(out[1]),
		}
		return class_preds
