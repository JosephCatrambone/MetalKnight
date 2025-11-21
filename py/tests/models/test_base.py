import numpy
import pytest

from models.base import softmax


def test_softmax():
	logits = numpy.array([1, 2, 3, 4,])
	proba = softmax(logits)
	assert proba[0] < proba[1]
	assert proba[1] < proba[2]
	assert proba[2] < proba[3]
	assert abs(numpy.sum(proba) - 1.0) < 1e-6
