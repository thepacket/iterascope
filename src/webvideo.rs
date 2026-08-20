//! In-browser video encoding: WebCodecs `VideoEncoder` producing H.264,
//! muxed into an MP4 in Rust.
//!
//! The bindings are hand-written `wasm_bindgen` externs rather than the
//! `web-sys` WebCodecs API, which is gated behind an unstable-APIs build
//! flag. The encoder is configured with `avc: {format: "avc"}` so every
//! chunk arrives as length-prefixed NAL units — exactly the sample format
//! MP4 wants — and the decoder configuration carries the `avcC` record from
//! which the SPS/PPS parameter sets are lifted for the track header.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsValue};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = VideoEncoder)]
    type JsVideoEncoder;
    #[wasm_bindgen(constructor, js_class = "VideoEncoder", catch)]
    fn new(init: &js_sys::Object) -> Result<JsVideoEncoder, JsValue>;
    #[wasm_bindgen(method, catch)]
    fn configure(this: &JsVideoEncoder, config: &js_sys::Object) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn encode(
        this: &JsVideoEncoder,
        frame: &JsVideoFrame,
        options: &js_sys::Object,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(method)]
    fn flush(this: &JsVideoEncoder) -> js_sys::Promise;
    #[wasm_bindgen(method)]
    fn close(this: &JsVideoEncoder);

    #[wasm_bindgen(js_name = VideoFrame)]
    type JsVideoFrame;
    #[wasm_bindgen(constructor, js_class = "VideoFrame", catch)]
    fn new(data: &js_sys::Uint8Array, init: &js_sys::Object) -> Result<JsVideoFrame, JsValue>;
    #[wasm_bindgen(method)]
    fn close(this: &JsVideoFrame);

    type JsEncodedChunk;
    #[wasm_bindgen(method, getter, js_name = byteLength)]
    fn byte_length(this: &JsEncodedChunk) -> u32;
    #[wasm_bindgen(method, getter)]
    fn timestamp(this: &JsEncodedChunk) -> f64;
    #[wasm_bindgen(method, getter, js_name = type)]
    fn chunk_type(this: &JsEncodedChunk) -> String;
    #[wasm_bindgen(method, js_name = copyTo)]
    fn copy_to(this: &JsEncodedChunk, destination: &js_sys::Uint8Array);
}

/// Whether this browser exposes WebCodecs video encoding.
pub(crate) fn supported() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("VideoEncoder"))
        .map(|value| !value.is_undefined())
        .unwrap_or(false)
}

fn set(object: &js_sys::Object, key: &str, value: impl Into<JsValue>) {
    let _ = js_sys::Reflect::set(object, &JsValue::from_str(key), &value.into());
}

struct EncodedSample {
    bytes: Vec<u8>,
    timestamp_us: f64,
    key: bool,
}

#[derive(Default)]
struct EncoderOutput {
    samples: Vec<EncodedSample>,
    /// The `avcC` decoder configuration record from the first chunk's
    /// metadata.
    avc_configuration: Option<Vec<u8>>,
    error: Option<String>,
}

/// A WebCodecs H.264 encoder collecting chunks for MP4 muxing.
pub(crate) struct Mp4Encoder {
    encoder: JsVideoEncoder,
    output: Rc<RefCell<EncoderOutput>>,
    width: u32,
    height: u32,
    fps: u32,
    // The callbacks must outlive the encoder.
    _on_output: Closure<dyn FnMut(JsValue, JsValue)>,
    _on_error: Closure<dyn FnMut(JsValue)>,
}

impl Mp4Encoder {
    pub(crate) fn new(width: u32, height: u32, fps: u32) -> Result<Self, String> {
        let output = Rc::new(RefCell::new(EncoderOutput::default()));

        let sink = Rc::clone(&output);
        let on_output = Closure::<dyn FnMut(JsValue, JsValue)>::new(
            move |chunk: JsValue, metadata: JsValue| {
                let mut sink = sink.borrow_mut();
                if sink.avc_configuration.is_none()
                    && let Ok(config) =
                        js_sys::Reflect::get(&metadata, &JsValue::from_str("decoderConfig"))
                    && let Ok(description) =
                        js_sys::Reflect::get(&config, &JsValue::from_str("description"))
                    && !description.is_undefined()
                {
                    // The description is an ArrayBuffer or a view of one.
                    let bytes = if let Some(buffer) = description.dyn_ref::<js_sys::ArrayBuffer>() {
                        js_sys::Uint8Array::new(buffer).to_vec()
                    } else {
                        js_sys::Uint8Array::new(&description).to_vec()
                    };
                    sink.avc_configuration = Some(bytes);
                }
                let chunk: JsEncodedChunk = chunk.unchecked_into();
                let destination = js_sys::Uint8Array::new_with_length(chunk.byte_length());
                chunk.copy_to(&destination);
                let mut bytes = vec![0u8; chunk.byte_length() as usize];
                destination.copy_to(&mut bytes);
                sink.samples.push(EncodedSample {
                    bytes,
                    timestamp_us: chunk.timestamp(),
                    key: chunk.chunk_type() == "key",
                });
            },
        );
        let errors = Rc::clone(&output);
        let on_error = Closure::<dyn FnMut(JsValue)>::new(move |error: JsValue| {
            errors.borrow_mut().error = Some(
                error
                    .as_string()
                    .or_else(|| {
                        js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                            .ok()
                            .and_then(|message| message.as_string())
                    })
                    .unwrap_or_else(|| "video encoder error".to_owned()),
            );
        });

        let init = js_sys::Object::new();
        set(&init, "output", on_output.as_ref().clone());
        set(&init, "error", on_error.as_ref().clone());
        let encoder =
            JsVideoEncoder::new(&init).map_err(|_| "cannot create the video encoder".to_owned())?;

        // High profile; level 4.2 covers 1080p60, 5.1 covers 4K.
        let codec = if width * height <= 1920 * 1088 {
            "avc1.64002a"
        } else {
            "avc1.640033"
        };
        let config = js_sys::Object::new();
        set(&config, "codec", codec);
        set(&config, "width", width);
        set(&config, "height", height);
        set(&config, "framerate", fps);
        // ~0.12 bits per pixel per frame.
        let bitrate = ((width as f64 * height as f64 * fps as f64 * 0.12) as u32)
            .clamp(1_000_000, 80_000_000);
        set(&config, "bitrate", bitrate);
        set(&config, "latencyMode", "quality");
        let avc = js_sys::Object::new();
        set(&avc, "format", "avc");
        set(&config, "avc", &avc);
        encoder
            .configure(&config)
            .map_err(|_| format!("this browser cannot encode {codec} at {width}×{height}"))?;

        Ok(Self {
            encoder,
            output,
            width,
            height,
            fps,
            _on_output: on_output,
            _on_error: on_error,
        })
    }

    /// Encodes one RGBA frame (rows top first).
    pub(crate) fn encode_rgba(&self, rgba: &[u8], frame_index: usize) -> Result<(), String> {
        if let Some(error) = &self.output.borrow().error {
            return Err(error.clone());
        }
        let data = js_sys::Uint8Array::from(rgba);
        let init = js_sys::Object::new();
        set(&init, "format", "RGBA");
        set(&init, "codedWidth", self.width);
        set(&init, "codedHeight", self.height);
        let timestamp = frame_index as f64 * 1_000_000.0 / self.fps as f64;
        set(&init, "timestamp", timestamp);
        set(&init, "duration", 1_000_000.0 / self.fps as f64);
        let frame =
            JsVideoFrame::new(&data, &init).map_err(|_| "cannot build a video frame".to_owned())?;
        let options = js_sys::Object::new();
        // A key frame every two seconds keeps the file seekable.
        set(
            &options,
            "keyFrame",
            frame_index % (2 * self.fps as usize).max(1) == 0,
        );
        let result = self.encoder.encode(&frame, &options);
        frame.close();
        result.map_err(|_| "the video encoder rejected a frame".to_owned())
    }

    /// Flushes the encoder and muxes the chunks into an MP4.
    pub(crate) async fn finish(self) -> Result<Vec<u8>, String> {
        wasm_bindgen_futures::JsFuture::from(self.encoder.flush())
            .await
            .map_err(|_| "the video encoder failed to flush".to_owned())?;
        self.encoder.close();
        let output = self.output.borrow();
        if let Some(error) = &output.error {
            return Err(error.clone());
        }
        if output.samples.is_empty() {
            return Err("the video encoder produced no frames".to_owned());
        }
        let configuration = output
            .avc_configuration
            .as_ref()
            .ok_or("the video encoder provided no decoder configuration")?;
        let (sps, pps) = parse_avc_configuration(configuration)?;
        mux_mp4(self.width, self.height, self.fps, sps, pps, &output.samples)
    }
}

/// Lifts the first SPS and PPS out of an `avcC` decoder configuration
/// record.
fn parse_avc_configuration(record: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let error = || "malformed avcC configuration".to_owned();
    if record.len() < 7 || record[0] != 1 {
        return Err(error());
    }
    let mut offset = 5;
    let sps_count = (record[offset] & 0x1f) as usize;
    offset += 1;
    if sps_count == 0 {
        return Err(error());
    }
    let sps_length = u16::from_be_bytes([record[offset], record[offset + 1]]) as usize;
    offset += 2;
    let sps = record
        .get(offset..offset + sps_length)
        .ok_or_else(error)?
        .to_vec();
    // Skip any further SPS entries.
    offset += sps_length;
    for _ in 1..sps_count {
        let length = u16::from_be_bytes([record[offset], record[offset + 1]]) as usize;
        offset += 2 + length;
    }
    let pps_count = *record.get(offset).ok_or_else(error)? as usize;
    offset += 1;
    if pps_count == 0 {
        return Err(error());
    }
    let pps_length = u16::from_be_bytes([
        *record.get(offset).ok_or_else(error)?,
        *record.get(offset + 1).ok_or_else(error)?,
    ]) as usize;
    offset += 2;
    let pps = record
        .get(offset..offset + pps_length)
        .ok_or_else(error)?
        .to_vec();
    Ok((sps, pps))
}

fn mux_mp4(
    width: u32,
    height: u32,
    fps: u32,
    sps: Vec<u8>,
    pps: Vec<u8>,
    samples: &[EncodedSample],
) -> Result<Vec<u8>, String> {
    const TIMESCALE: u32 = 90_000;
    let config = mp4::Mp4Config {
        major_brand: (*b"isom").into(),
        minor_version: 512,
        compatible_brands: vec![(*b"isom").into(), (*b"avc1").into()],
        timescale: 1_000,
    };
    let cursor = std::io::Cursor::new(Vec::new());
    let mut writer =
        mp4::Mp4Writer::write_start(cursor, &config).map_err(|error| error.to_string())?;
    writer
        .add_track(&mp4::TrackConfig {
            track_type: mp4::TrackType::Video,
            timescale: TIMESCALE,
            language: "und".to_owned(),
            media_conf: mp4::MediaConfig::AvcConfig(mp4::AvcConfig {
                width: width as u16,
                height: height as u16,
                seq_param_set: sps,
                pic_param_set: pps,
            }),
        })
        .map_err(|error| error.to_string())?;
    let duration = TIMESCALE / fps.max(1);
    for sample in samples {
        writer
            .write_sample(
                1,
                &mp4::Mp4Sample {
                    start_time: (sample.timestamp_us * TIMESCALE as f64 / 1_000_000.0).round()
                        as u64,
                    duration,
                    rendering_offset: 0,
                    is_sync: sample.key,
                    bytes: mp4::Bytes::from(sample.bytes.clone()),
                },
            )
            .map_err(|error| error.to_string())?;
    }
    writer.write_end().map_err(|error| error.to_string())?;
    Ok(writer.into_writer().into_inner())
}
