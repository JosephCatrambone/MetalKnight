from abc import ABC, abstractmethod

import numpy
from PIL import Image


MODEL_STORAGE_DIR = "../resources"


def softmax(x):
	return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)


class BaseMLModel(ABC):
	def prepare_image_as_chw_tensor(self, image: Image.Image, auto_resize: bool = True) -> numpy.ndarray:
		"""Given an image in RGB, assert that the W,H match our input size and that our mode is RGB, then convert to
		a channels-first, height-second, width third format and return as a float tensor with pixels in [0,1]."""
		assert image.mode == 'RGB'
		if not auto_resize:
			assert image.size == self.get_input_image_size()
		else:
			image = image.resize(self.get_input_image_size())
		tensor = numpy.asarray(image).astype(numpy.float32) / 255.0  # Generates H=0, W=1, C=2.
		tensor = numpy.transpose(tensor, (2, 0, 1))  # TODO: Double check this is C, H, W.
		return tensor

	@abstractmethod
	def load_model(self):
		"""Load and initialize the model, prepping weights."""
		...

	@abstractmethod
	def is_model_loaded(self) -> bool:
		...

	@abstractmethod
	def unload_model(self):
		...

	@abstractmethod
	def get_class_names(self) -> list[str]:
		...

	@abstractmethod
	def get_input_image_size(self) -> tuple[int, int]:
		"""Returns a tuple of the width,height of the expected image input."""
		...

	@abstractmethod
	def predict_classes(self, image, normalize: bool = True) -> dict[str, float]:
		...