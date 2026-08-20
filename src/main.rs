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

        let runner = eframe::WebRunner::new();
        let result = runner
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

        // The `?autotest=` smoke harness must run even in a hidden window,
        // where the browser never fires an animation frame; a timer loop
        // drives the export steps directly.
        let autotest = web_sys::window()
            .and_then(|window| window.location().search().ok())
            .is_some_and(|search| search.contains("autotest"));
        if result.is_ok() && autotest {
            wasm_bindgen_futures::spawn_local(async move {
                log::info!("autotest driver running");
                let mut idle_ticks = 0u32;
                for _ in 0..600 {
                    sleep_ms(500).await;
                    let Some(mut app) = runner.app_mut::<iterascope::App>() else {
                        break;
                    };
                    if app.web_autotest_step() {
                        idle_ticks = 0;
                    } else {
                        idle_ticks += 1;
                        if idle_ticks > 6 {
                            break;
                        }
                    }
                }
                log::info!("autotest driver finished");
            });
        }
    });
}

#[cfg(target_arch = "wasm32")]
async fn sleep_ms(milliseconds: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window()
            .expect("no browser window")
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, milliseconds)
            .expect("cannot schedule a timeout");
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}
