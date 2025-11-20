import time
from typing import Annotated, Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel

from models import AdultExplicit

app = FastAPI()
ml_models = {
	"adult_nsfw": AdultExplicit(),
}

# These will NOT be in shared memory, so all workers will have an instance.
# If/when we need a GPU instance we have two options:
# 1. Use a high-quality stem model that produces a useful embedding and a bunch of CPU-based heads for inference.
# 2. Some kind of external shared-memory worker communicating with IPC.


@app.get("/")
def heartbeat():
	return {"now": time.time()}


@app.get("/modelinfo")
def get_model_names():
	return {"model_names": [k for k in ml_models.keys()]}

# File inherits basically directly from form, doesn't spool to disk, and is lower level.
#@app.post("/files/")
#def create_file(file: Annotated[bytes, File()]):
#	return {"file_size": len(file)}

@app.post("/inference/{model_name}")
def inference_single(model_name: str, file: UploadFile):
	if model_name not in ml_models:
		raise HTTPException(status_code=404, detail=f"Model named {model_name} not found")
	image = Image.open(file.file)
	active_model = ml_models[model_name]
	if not active_model.is_model_loaded():
		active_model.load_model()
	preds = ml_models[model_name].predict_classes(image)
	return preds


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
	return {"item_id": item_id, "q": q}
