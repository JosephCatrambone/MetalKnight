
use image::{DynamicImage, ImageReader, GenericImageView};
use std::fs::create_dir_all;
use std::io::Cursor;
use std::path::Path;
use salvo::prelude::*;
use tract_onnx::prelude::*;


#[handler]
async fn index(res: &mut Response) {
	res.render(Text::Html("<html>Ohai</html>"));
}


#[handler]
async fn upload(req: &mut Request, res: &mut Response) {
	let file = req.file("file").await;
	if let Some(file) = file {
		/*
		let dest = format!("temp/{}", file.name().unwrap_or("file"));
		let info = if let Err(e) = std::fs::copy(file.path(), Path::new(&dest)) {
			res.status_code(StatusCode::INTERNAL_SERVER_ERROR);
			format!("file not found in request: {e}")
		} else {
			format!("File uploaded to {dest}")
		};

		res.render(Text::Plain(info));
		*/
		// ImageReader::new
		let img = ImageReader::open(file.path()).unwrap().decode().unwrap();
		let (width, height) = img.dimensions();
		res.render(Text::Plain(format!("Image size: {}x{}", width, height)));
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
	let router = Router::new().get(index).post(upload);

	let acceptor = TcpListener::new("0.0.0.0:8080").bind().await;
	Server::new(acceptor).serve(router).await;
}


fn inference() -> TractResult<()> {
	let model = tract_onnx::onnx()
		//.model_for_read(include_bytes!())
		// load the model
		.model_for_path("mobilenetv2-7.onnx")?
		// optimize the model
		.into_optimized()?
		// make the model runnable and fix its inputs and outputs
		.into_runnable()?;

	// open image, resize it and make a Tensor out of it
	let image = image::open("grace_hopper.jpg").unwrap().to_rgb8();
	let resized =
		image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
	let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
		let mean = [0.485, 0.456, 0.406][c];
		let std = [0.229, 0.224, 0.225][c];
		(resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
	}).into();

	// run the model on the input
	let result = model.run(tvec!(image.into()))?;

	// find and display the max value with its index
	let best = result[0]
		.to_array_view::<f32>()?
		.iter()
		.cloned()
		.zip(2..)
		.max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
	println!("result: {best:?}");
	Ok(())
}
