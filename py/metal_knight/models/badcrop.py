import os

import numpy
import onnxruntime as ort
from PIL import Image

from models.base import BaseMLModel, MODEL_STORAGE_DIR, softmax


class BadCrop(BaseMLModel):
	def __init__(self):
		self.inference_session = None

	def load_model(self):
		self.inference_session = ort.InferenceSession(os.path.join(MODEL_STORAGE_DIR, "bad_crop.onnx"))

	def is_model_loaded(self) -> bool:
		return self.inference_session is not None

	def unload_model(self):
		del self.inference_session
		self.inference_session = None

	def get_input_image_size(self) -> tuple[int, int]:
		return 224, 224

	def get_class_names(self) -> list[str]:
		return ["goodcrop", "badcrop"]

	def predict_classes(self, image: Image.Image, normalize: bool = True) -> dict[str, float]:
		"""Run inference on an input image, returning a map from class name to probability."""
		tensor = self.prepare_image_as_chw_tensor(image)
		tensor = numpy.expand_dims(tensor, axis=0)
		out = self.inference_session.run(["output"], {"input": tensor,})[0][0]
		if normalize:
			out = softmax(out)
		class_preds = {
			"badcrop": float(out[0]),
			"goodcrop": float(out[1]),
		}
		return class_preds
