fn build_app(
    cc: &eframe::CreationContext<'_>,
) -> Result<Box<dyn eframe::App>, Box<dyn std::error::Error + Send + Sync>> {
    iterascope::App::new(cc)
        .map(|app| Box::new(app) as Box<dyn eframe::App>)
        .ok_or_else(|| "IteraScope requires the wgpu rendering backend".into())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([760.0, 520.0])
            .with_title("IteraScope"),
        ..Default::default()
    };

    eframe::run_native("IteraScope", options, Box::new(build_app))
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    console_error_panic_hook::set_once();
    eframe::WebLogger::init(log::LevelFilter::Info).ok();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no browser window")
            .document()
            .expect("no document");
        let canvas = document
            .get_element_by_id("iterascope_canvas")
            .expect("index.html is missing #iterascope_canvas")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("#iterascope_canvas is not a canvas");

        let result = eframe::WebRunner::new()
            .start(canvas, eframe::WebOptions::default(), Box::new(build_app))
            .await;

        if let Some(loading) = document.get_element_by_id("iterascope_loading") {
            match &result {
                Ok(()) => loading.remove(),
                Err(error) => loading.set_inner_html(&format!(
                    "<div class='brand'>ITERASCOPE</div>\
                     <div class='error'>Could not start WebGPU.<br>{error:?}</div>"
                )),
            }
        }
    });
}
