mod ml_model;

use image::ImageReader;
use salvo::oapi::extract::*;
use salvo::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use ml_model::MLModel;


// Static models:
static NSFW_MODEL: LazyLock<Arc<Mutex<MLModel>>> = LazyLock::new(|| {
	Arc::new(Mutex::new(MLModel::new_from_bytes("nsfw", vec!["safe", "nsfw"], (224, 224), include_bytes!("../resources/adult_nsfw.onnx"))))
});
static BAD_CROP_MODEL: LazyLock<Arc<Mutex<MLModel>>> = LazyLock::new(|| {
	Arc::new(Mutex::new(MLModel::new_from_bytes("badcrop", vec!["goodcrop", "badcrop"], (224, 224), include_bytes!("../resources/bad_crop.onnx"))))
});
static SCREENSHOT_MODEL: LazyLock<Arc<Mutex<MLModel>>> = LazyLock::new(|| {
	Arc::new(Mutex::new(MLModel::new_from_bytes("screenshot", vec!["not_screenshot", "screenshot"], (224, 224), include_bytes!("../resources/screenshot.onnx"))))
});
static MODEL_LIST: [&LazyLock<Arc<Mutex<MLModel>>>; 3] = [&NSFW_MODEL, &BAD_CROP_MODEL, &SCREENSHOT_MODEL];


// Handlers:
// or endpoints, since we have no #[handler].

/// Get meta-info for each model, including the preferred input image size and output classes.
#[endpoint]
async fn model_info(model_name_param: QueryParam<String, false>) -> String {
	let model_name = model_name_param.as_deref().unwrap_or("");
	let mut resp = String::new();
	if model_name.eq("") {
		// List all models.
		for m in MODEL_LIST {
			if let Ok(model) = m.lock() {
				resp += model.name;
				resp += "\n";
			}
		}
	}
	resp
}


#[endpoint]
async fn inference(file: FormFile, model_names: QueryParam<String, false>) -> String {
	// TODO: Set return type to Json and make a real response object.
	let img = ImageReader::open(file.path()).unwrap().decode().unwrap();
	let mut model_predictions: HashMap<String, Value> = HashMap::new();

	let model_subset: Vec<String> = if let Some(name_list) = model_names.into_inner() {
		name_list.split(",").map(|f|{ f.to_string() }).collect()
	} else {
		vec![]
	};

	for model_ref in MODEL_LIST {
		if let Ok(mut model) = model_ref.lock() {
			// Could we find a way to skip getting the lock?
			if model_subset.is_empty() || model_subset.contains(&model.name.to_string()) {
				let preds = model.infer_from_image(&img.to_rgb8());
				model_predictions.insert(model.name.to_string(), serde_json::to_value(preds).unwrap());
			}
		}
	}

	let json_data = serde_json::to_value(&model_predictions).expect("Failed to serialize model outputs");
	serde_json::to_string(&json_data).expect("Failed to stringify JSON value.")
}


#[handler]
async fn old_model_inference(req: &mut Request, res: &mut Response) {
	let file = req.file("file").await;
	if let Some(file) = file {
		// ImageReader::new
		let img = ImageReader::open(file.path()).unwrap().decode().unwrap();
		//let (width, height) = img.dimensions();
		//res.render(Text::Plain(format!("Image size: {}x{}", width, height)));
		if let Ok(mut model) = NSFW_MODEL.clone().lock() {
			let preds = model.infer_from_image(&img.to_rgb8());
			res.render(Text::Plain(format!("{}", preds.get("nsfw").unwrap())));
		}
	} else {
		res.status_code(StatusCode::BAD_REQUEST);
		res.render(Text::Plain("file not found in request"));
	};
}


#[tokio::main]
async fn main() {
	tracing_subscriber::fmt().init();

	// Do we need a temp dir for photos?
	//use std::fs::create_dir_all;
	//create_dir_all("temp").unwrap();

	let router = Router::new().push(
		Router::with_path("inference")
			.get(model_info)
			.post(inference)
	);

	let doc = OpenApi::new("Metal Knight API", "0.1.0").merge_router(&router);

	let router = router
		.unshift(doc.into_router("/api-doc/openapi.json"))
		.unshift(SwaggerUi::new("/api-doc/openapi.json").into_router("/swagger-ui"));

	let acceptor = TcpListener::new("0.0.0.0:8080").bind().await;
	Server::new(acceptor).serve(router).await;
}
