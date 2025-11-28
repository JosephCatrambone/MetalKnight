//mod nsfw;

use image::{imageops, self, RgbImage};
use image::imageops::FilterType;
use std::collections::HashMap;
use std::io::Cursor;
use tract_onnx::prelude::*;

/*
pub trait MLModel {
	fn get_name(&self) -> &'static str;
	fn get_preferred_image_size(&self) -> (u32, u32);
	fn get_class_names(&self) -> Vec<&'static str>;
	fn infer(&self, img: &RgbImage) -> HashMap<&str, f32>;
}
*/

pub struct MLModel {
	pub name: &'static str,
	pub preferred_image_size: (u32, u32),
	pub class_names: Vec<&'static str>,
	pub runnable: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

pub fn preprocess_for_model(
	image: &RgbImage,
	target_size: (u32, u32),
) -> Tensor {
	let current_width = image.width();
	let current_height = image.height();
	let (target_width, target_height) = target_size;

	let scale_factor = if current_width > current_height { // Landscape
		target_width as f32 / current_width as f32
	} else {
		target_height as f32 / current_height as f32
	};

	let img_new = imageops::resize(image, (current_width as f32 * scale_factor) as u32, (current_height as f32 * scale_factor) as u32, FilterType::Lanczos3);

	let image_tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, target_height as usize, target_width as usize), |(_, c, y, x)| {
		//let mean = [0.485, 0.456, 0.406][c];
		//let std = [0.229, 0.224, 0.225][c];
		//(img_new[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
		img_new[(x as _, y as _)][c] as f32 / 255.0
	}).into();

	image_tensor
}

impl MLModel {
	pub fn new_from_bytes(
		name: &'static str,
		class_names: Vec<&'static str>,
		preferred_image_size: (u32, u32),
		model_bytes: &'static [u8], // Use include_bytes!()
	) -> Self {
		let mut model_buffer = Cursor::new(model_bytes.clone());

		let model = tract_onnx::onnx()
			.model_for_read(&mut model_buffer).expect("Failed to load compiled model.")
			// load the model
			//.model_for_path("mobilenetv2-7.onnx")?
			// optimize the model
			.into_optimized().expect("Failed to convert pre-packaged model.")
			// make the model runnable and fix its inputs and outputs
			.into_runnable().expect("Failed to fix model IO.");

		MLModel {
			name,
			preferred_image_size,
			class_names,
			runnable: model
		}
	}

	pub fn infer_from_image(&self, image: &RgbImage) -> TractResult<HashMap<&'static str, f32>> {
		let t = preprocess_for_model(image, self.preferred_image_size);
		self.infer_from_tensor(t)
	}

	pub fn infer_from_tensor(&self, tensor: Tensor) -> TractResult<HashMap<&'static str, f32>> {
		let preds = self.runnable.run(tvec!(tensor.into()))?;
		let pred_array = preds[0].to_array_view::<f32>()?;

		// TODO: Normalize?

		let mut out = HashMap::new();
		for (idx, classname) in self.class_names.iter().enumerate() {
			out.insert(*classname, pred_array[idx]);
		}
		Ok(out)
	}
}