use image::{imageops, self, RgbImage};
use image::imageops::FilterType;
use ort;
use ort::session::{builder::GraphOptimizationLevel, Session, InMemorySession, SessionOutputs};
//use ort::tensor::ArrayExtensions; // If we wanted softmax on the out vectors, but this appears buggy.
use ort::value::Tensor;
use std::collections::HashMap;

pub struct MLModel {
	pub name: &'static str,
	pub preferred_image_size: (u32, u32),
	pub class_names: Vec<&'static str>,
	pub pad_image: bool,
	model: InMemorySession<'static>,
}

pub fn softmax(vec: &[f32]) -> Vec<f32> {
	let vecmax = vec.iter().cloned().fold(0.0f32, f32::max);
	let div: f32 = vec.iter().map(|v| { (v - vecmax).exp() }).sum();
	if div.abs() < 1e-6 {
		panic!()
	}
	vec.iter().map(|v| { (v - vecmax).exp() / div }).collect()
}

pub fn preprocess_for_model(
	image: &RgbImage,
	target_size: (u32, u32),
	letterbox: bool,
) -> Tensor<f32> {
	// We copy an image into a pre-allocated vec of f32 size.
	// If letterbox is true, we downsample the image, keeping the aspect ratio, and adding padding.
	// If an image is smaller than the min size it's upscaled to match.
	// If letterboxing is false, we scale the image and disregard the aspect ratio.

	let current_width = image.width();
	let current_height = image.height();
	let (target_width, target_height) = target_size;

	// The x_start and y_start are zero if we have an exact fit to the frame. (Like for non-letterbox.)
	// If we letterbox then we set these to be the required offset between the edge of the frame and the image.
	let mut x_start = 0;
	let mut y_start = 0;

	let img_new = if letterbox {
		let scale_factor = if current_width > current_height { // Landscape
			target_width as f32 / current_width as f32
		} else {
			target_height as f32 / current_height as f32
		};

		let new_width = (current_width as f32 * scale_factor) as u32;
		let new_height = (current_height as f32 * scale_factor) as u32;
		x_start = (target_width / 2) - (new_width / 2);
		y_start = (target_height / 2) - (new_height / 2);
		imageops::thumbnail(image, new_width, new_height)
	} else {
		imageops::resize(image, target_width, target_height, FilterType::Lanczos3)
	};

	/*
	let image_tensor = Array4::<f32>::from_shape_fn((1, 3, target_height as usize, target_width as usize), |(_, c, y, x)| {
		img_new[(x as _, y as _)][c] as f32 / 255.0
	});
	*/
	let mut image_bchw = Vec::<f32>::with_capacity(3 * target_width as usize * target_height as usize);
	for c in 0..3 {
		for y in 0..target_height {
			if y < y_start || y >= y_start + img_new.height() {
				for _x in 0..target_width {
					image_bchw.push(0.0);
				}
			} else {
				for x in 0..target_width {
					if x < x_start || x >= x_start + img_new.width() {
						image_bchw.push(0.0);
					} else {
						image_bchw.push(img_new[((x-x_start) as _, (y-y_start) as _)][c] as f32 / 255.0);
					}
				}
			}
		}
	}
	assert_eq!(image_bchw.len(), 3 * target_width as usize * target_height as usize);

	Tensor::from_array(([1usize, 3usize, target_height as usize, target_width as usize], image_bchw)).unwrap()
}

impl MLModel {
	pub fn new_from_bytes(
		name: &'static str,
		class_names: Vec<&'static str>,
		preferred_image_size: (u32, u32),
		pad_image: bool,  // Should the image keep the aspect ratio on resize or fill the frame?
		model_bytes: &'static [u8], // Use include_bytes!()
	) -> Self {
		let model = Session::builder().expect("Failed to init ONNX session")
			.with_optimization_level(GraphOptimizationLevel::Level3).expect("Failed to optimize ONNX graph.")
			//.with_intra_threads(threads).expect("Failed to set thread count for ONNX.")
			.commit_from_memory_directly(model_bytes).expect("Failed to commit ONNX model");

		MLModel {
			name,
			preferred_image_size,
			class_names,
			pad_image,
			model
		}
	}

	pub fn infer_from_image(&mut self, image: &RgbImage) -> HashMap<&'static str, f32> {
		let t = preprocess_for_model(image, self.preferred_image_size, self.pad_image);
		//let t_ref = ort::value::TensorRef::from_array_view(&t).unwrap();
		self.infer_from_tensor(&t)
	}

	pub fn infer_from_tensor(&mut self, tensor: &Tensor<f32>) -> HashMap<&'static str, f32>	{
		let outputs: SessionOutputs = self.model.run(ort::inputs!["input" => tensor]).unwrap();
		let predictions = outputs["output"].try_extract_array::<f32>().unwrap();

		// .softmax(Axis(0)) should be a thing but it's broken, for now.
		let normalized_predictions = softmax(predictions.as_slice().unwrap());
		// Make normalization optional?

		let mut out = HashMap::new();
		for (idx, classname) in self.class_names.iter().enumerate() {
			out.insert(*classname, normalized_predictions[idx]);
		}
		out
	}
}