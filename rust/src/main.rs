mod ml_model;

use std::collections::HashMap;
use image::{ImageReader, GenericImageView};
use salvo::oapi::extract::*;
use salvo::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value, Result};
use std::fs::create_dir_all;
use std::sync::{Arc, LazyLock, Mutex};
use ml_model::MLModel;

// Staic models:
static NSFW_MODEL: LazyLock<Arc<Mutex<MLModel>>> = LazyLock::new(|| {
	Arc::new(Mutex::new(MLModel::new_from_bytes("nsfw", vec!["safe", "nsfw"], (224, 224), include_bytes!("../resources/adult_nsfw.onnx"), 4)))
});
static BAD_CROP_MODEL: LazyLock<Arc<Mutex<MLModel>>> = LazyLock::new(|| {
	Arc::new(Mutex::new(MLModel::new_from_bytes("badcrop", vec!["goodcrop", "badcrop"], (224, 224), include_bytes!("../resources/bad_crop.onnx"), 4)))
});
static SCREENSHOT_MODEL: LazyLock<Arc<Mutex<MLModel>>> = LazyLock::new(|| {
	Arc::new(Mutex::new(MLModel::new_from_bytes("screenshot", vec!["not_screenshot", "screenshot"], (224, 224), include_bytes!("../resources/screenshot.onnx"), 4)))
});


// Handlers:
#[handler]
async fn index(res: &mut Response) {
	res.render(Text::Html("<html>Ohai</html>"));
}


#[endpoint]
async fn model_info(model_name: QueryParam<String, false>) -> String {
	format!("Hello, {}!", model_name.as_deref().unwrap_or("World"))
}


#[endpoint]
async fn inference(file: FormFile, model_names: QueryParam<String, false>) -> String {
	let img = ImageReader::open(file.path()).unwrap().decode().unwrap();
	//let mut res: Value = json!({});
	let mut model_predictions: HashMap<String, Value> = HashMap::new();

	for model_ref in [&NSFW_MODEL, &BAD_CROP_MODEL, &SCREENSHOT_MODEL] {
		if let Ok(mut model) = model_ref.lock() {
			let preds = model.infer_from_image(&img.to_rgb8());
			model_predictions.insert(model.name.to_string(), serde_json::to_value(preds).unwrap());
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
		Router::with_path("model")
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
