mod ml_model;

use image::{ImageReader, GenericImageView};
use std::fs::create_dir_all;
use std::sync::LazyLock;
use salvo::oapi::extract::*;
use salvo::prelude::*;
use tract_onnx::prelude::*;

use ml_model::MLModel;

// Staic models:
static NSFW_MODEL: LazyLock<MLModel> = LazyLock::new(|| {
	MLModel::new_from_bytes("nsfw", vec!["safe", "nsfw"], (224, 224), include_bytes!("../resources/nsfw_model.onnx"))
});


#[handler]
async fn index(res: &mut Response) {
	res.render(Text::Html("<html>Ohai</html>"));
}


#[endpoint]
async fn model_info(model_name: QueryParam<String, false>) -> String {
	format!("Hello, {}!", model_name.as_deref().unwrap_or("World"))
}


#[handler]
async fn model_inference(req: &mut Request, res: &mut Response) {
	let file = req.file("file").await;
	if let Some(file) = file {
		// ImageReader::new
		let img = ImageReader::open(file.path()).unwrap().decode().unwrap();
		//let (width, height) = img.dimensions();
		//res.render(Text::Plain(format!("Image size: {}x{}", width, height)));
		let model: &MLModel = &*NSFW_MODEL;
		if let Ok(preds) = model.infer_from_image(&img.to_rgb8()) {
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
			.post(model_inference)
	);

	let doc = OpenApi::new("Metal Knight API", "0.1.0").merge_router(&router);

	let router = router
		.unshift(doc.into_router("/api-doc/openapi.json"))
		.unshift(SwaggerUi::new("/api-doc/openapi.json").into_router("/swagger-ui"));

	let acceptor = TcpListener::new("0.0.0.0:8080").bind().await;
	Server::new(acceptor).serve(router).await;
}
