# A testing script that loads and runs all the models on the provided input image.
# Tracks inference time and, as best it can, memory.

import dataclasses
import sys
import time

from PIL import Image

from models import AdultExplicit, BaseMLModel


@dataclasses.dataclass
class TestRun:
	model_name: str
	load_time_ms: float
	inference_time_ms: float
	image_path: str
	predictions: dict[str, float]


def test_model(model: BaseMLModel, image: Image.Image):
	"""Gather super basic and lazy metrics on load time and inference time."""
	start_time = time.time()
	model.load_model()
	load_finished = time.time()
	preds = model.predict_classes(image)
	inference_finished = time.time()

	return TestRun(
		model_name=str(model),
		load_time_ms=load_finished - start_time,
		inference_time_ms=inference_finished - load_finished,
		image_path="",
		predictions=preds
	)


def main():
	all_models = [AdultExplicit]

	# Load an image from args, or if one does not exist make a blank placeholder.
	if len(sys.argv) > 1:
		image_name = sys.argv[1]
		image = Image.open(image_name).convert("RGB")
	else:
		image = Image.new("RGB", (512, 512))

	# Test each model's inference time and dump to stdout.
	for m_cls in all_models:
		m = m_cls()
		resized_image = image.resize(m.get_input_image_size())
		res = test_model(m, resized_image)
		print(res)


if __name__ == '__main__':
	main()
