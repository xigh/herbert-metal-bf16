//! herbert-metal-serve MODEL_DIR [options]
//!
//! Gemma-4 (E2B, E4B) behind llama-server's HTTP API, one slot: `POST /v1/chat/completions` (streamed as SSE or not, tools,
//! `chat_template_kwargs.enable_thinking`, `timings_per_token`, `stream_options.include_usage`), `GET /props`,
//! `GET /v1/models`, `GET /health`, `POST /apply-template`, `POST /tokenize`. The chat template is the model's own
//! (`chat_template.jinja`); thinking comes back in `reasoning_content`, tool calls in `tool_calls`. Requests are answered one
//! at a time; the prefix a request shares with the previous one stays in the K/V cache.
//!
//!   --host H             listen address (default 0.0.0.0)
//!   --port P             (default 8000)
//!   --ctx N              positions of the context, the `n_ctx` of /props (default 32768)
//!   --prefill-block N    tokens of a prompt block, the projections as one Q4 product (default 16; 1 = token by token)
//!   --sliding-ring N     the sliding layers' K/V caches as rings of N positions (default 0: the whole capacity)
//!   --keep-warm SECS     after SECS without a request, one token through the model, the cache kept (default 0: never)
//!   --side-slot N        a second state kept aside, the first N positions of every K/V cache in host memory: a short
//!                        request that shares nothing with the conversation (a client's periodic agent) and the conversation
//!                        each keep their prefix (default 0: one state)
//!   --prefix-cache DIR   the K/V of the requests' stable prefix (the system message and the tools) kept in DIR, read
//!                        back instead of computed after a restart (nothing when DIR is not a directory)
//!   --prefix-cache-keep K  the K most recently used prefixes kept in DIR (default 8)
//!   --q4                 the projections and the head in Q4_0, quantized at load (default: 16 bits)
//!   --alias NAME         the model name the API reports (default: the directory's name)
//!   --log-dir DIR        per request: the request, the rendered prompt, the raw output with its special tokens, the response
//!
//! On stderr per request: the prompt, the prefix reused, the speeds, the end, where the time of the steps went, the state
//! of the process and of the machine.

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use herbert_metal::metal::gemma::{GemmaEngine, Options};
use herbert_metal::serve::prefix_cache::PrefixCache;
use herbert_metal::serve::template::Template;
use herbert_metal::serve::worker::{error_body, Job, Out, Worker};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

struct App {
    jobs: Mutex<std::sync::mpsc::Sender<Job>>,
    next: AtomicU64,
    template: Arc<Template>,
    tokenizer: Arc<Tokenizer>,
    props: Value,
    model: String,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("error: {e}\nusage: herbert-metal-serve MODEL_DIR [--host H] [--port P] [--ctx N] [--prefill-block N] [--sliding-ring N] [--keep-warm SECS] [--side-slot N] [--prefix-cache DIR] [--prefix-cache-keep K] [--q4] [--alias NAME] [--log-dir DIR]");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut it = args.iter();
    let model_dir = PathBuf::from(it.next().ok_or("MODEL_DIR is required")?);
    let (mut host, mut port, mut alias, mut log_dir) = ("0.0.0.0".to_string(), 8000u16, None, None);
    let mut keep_warm: Option<std::time::Duration> = None;
    let mut side_slot = 0usize;
    let (mut prefix_dir, mut prefix_keep): (Option<PathBuf>, usize) = (None, 8);
    let mut o = Options { capacity: 32768, matvec_rows: 2, dump_msl: None, q4: false, logits_f32: false, prefill_block: 16, sliding_ring: 0 };
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--host" => host = value()?,
            "--port" => port = value()?.parse().map_err(|e| format!("--port: {e}"))?,
            "--ctx" => o.capacity = value()?.parse().map_err(|e| format!("--ctx: {e}"))?,
            "--bf16" => o.q4 = false,
            "--q4" => o.q4 = true,
            "--prefill-block" => o.prefill_block = value()?.parse().map_err(|e| format!("--prefill-block: {e}"))?,
            "--sliding-ring" => o.sliding_ring = value()?.parse().map_err(|e| format!("--sliding-ring: {e}"))?,
            "--keep-warm" => keep_warm = Some(std::time::Duration::from_secs(value()?.parse().map_err(|e| format!("--keep-warm: {e}"))?)).filter(|d| !d.is_zero()),
            "--side-slot" => side_slot = value()?.parse().map_err(|e| format!("--side-slot: {e}"))?,
            "--prefix-cache" => prefix_dir = Some(PathBuf::from(value()?)),
            "--prefix-cache-keep" => prefix_keep = value()?.parse().map_err(|e| format!("--prefix-cache-keep: {e}"))?,
            "--alias" => alias = Some(value()?),
            "--log-dir" => log_dir = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown option {other}")),
        }
    }
    let q4 = o.q4;
    let prefill_block = o.prefill_block;
    let dir = herbert_metal::model::resolve_model_dir(&model_dir).map_err(|e| e.to_string())?;
    if let Some(d) = &log_dir {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let model = alias.unwrap_or_else(|| dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "gemma-4".into()));
    let tokenizer = Arc::new(Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?);
    let template = Arc::new(Template::load(&dir).map_err(|e| e.to_string())?);
    let prefix_cache = match &prefix_dir {
        Some(d) => PrefixCache::open(d, prefix_keep, &dir, q4, &template.source).map_err(|e| format!("--prefix-cache: {e}"))?,
        None => None,
    };
    // The engine lives on its own thread (its Metal objects are not Send): loaded there, `n_ctx` sent back.
    let (jobs_tx, jobs_rx) = std::sync::mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<usize, String>>();
    {
        let (dir, tokenizer, template, model) = (dir.clone(), tokenizer.clone(), template.clone(), model.clone());
        std::thread::Builder::new()
            .name("engine".into())
            .spawn(move || match GemmaEngine::load(&dir, &o) {
                Ok(mut engine) => {
                    eprintln!("[engine] loaded in {:.1} s, n_ctx {}", engine.load_seconds, engine.capacity());
                    // Every shape of step through the model once before the first
                    // request: a token alone, then a prompt of each length up to one
                    // prefill block (a short last block has kernels of its own). The
                    // first submission of a pipeline, and of the weight buffers, waits
                    // in the driver (on the M5 with 26B-A4B, 18–32 s for the first
                    // prompt): paid here rather than by a client.
                    let t = std::time::Instant::now();
                    let warm = (1..=prefill_block.max(1)).try_for_each(|m| engine.prefill(&vec![2; m]).and_then(|_| engine.reset()));
                    match warm {
                        Ok(()) => eprintln!("[engine] warm-up (1 to {} tokens) in {:.1} s", prefill_block.max(1), t.elapsed().as_secs_f64()),
                        Err(e) => eprintln!("[engine] warm-up failed: {e}"),
                    }
                    match engine.side_enable(side_slot) {
                        Ok(0) => {}
                        Ok(bytes) => eprintln!("[engine] side state: {side_slot} positions, {:.2} GiB of host memory", bytes as f64 / (1u64 << 30) as f64),
                        Err(e) => {
                            let _ = ready_tx.send(Err(e.to_string()));
                            return;
                        }
                    }
                    let _ = ready_tx.send(Ok(engine.capacity()));
                    let mut w = Worker::new(engine, tokenizer, template, model, log_dir);
                    w.keep_warm = keep_warm;
                    w.prefix_cache = prefix_cache;
                    w.run(jobs_rx);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                }
            })
            .map_err(|e| e.to_string())?;
    }
    let n_ctx = ready_rx.recv().map_err(|e| e.to_string())??;
    let props = json!({
        "build_info": format!("herbert-metal-bf16 {} ({})", env!("CARGO_PKG_VERSION"), if q4 { "Q4_0" } else if herbert_metal::core::bf16::F16 { "f16" } else { "bf16" }),
        "model_path": dir.display().to_string(),
        "model_alias": model,
        "total_slots": 1,
        "default_generation_settings": { "n_ctx": n_ctx, "params": { "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05 } },
        "chat_template": template.source,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "modalities": { "vision": false, "audio": false },
    });
    let app = Arc::new(App { jobs: Mutex::new(jobs_tx), next: AtomicU64::new(1), template, tokenizer, props, model });
    let router = Router::new()
        .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
        .route("/props", get(props_route))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/apply-template", post(apply_template))
        .route("/tokenize", post(tokenize))
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
        .with_state(app);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.map_err(|e| format!("bind {host}:{port}: {e}"))?;
        eprintln!("[serve] listening on http://{host}:{port}");
        axum::serve(listener, router).await.map_err(|e| e.to_string())
    })
}

async fn props_route(State(app): State<Arc<App>>) -> Json<Value> {
    Json(app.props.clone())
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({ "object": "list", "data": [{ "id": app.model, "object": "model", "owned_by": "herbert" }] }))
}

fn error(code: u16, body: Value) -> Response {
    (StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(body)).into_response()
}

async fn apply_template(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let thinking = body.pointer("/chat_template_kwargs/enable_thinking").and_then(Value::as_bool).unwrap_or(false);
    match app.template.render(&body["messages"], body.get("tools"), thinking, true) {
        Ok(prompt) => Json(json!({ "prompt": prompt })).into_response(),
        Err(e) => error(400, error_body(400, "invalid_request_error", &e.to_string())),
    }
}

async fn tokenize(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let content = body["content"].as_str().unwrap_or("");
    let special = body["add_special"].as_bool().unwrap_or(false);
    match app.tokenizer.encode(content, special) {
        Ok(enc) => Json(json!({ "tokens": enc.get_ids() })).into_response(),
        Err(e) => error(400, error_body(400, "invalid_request_error", &e.to_string())),
    }
}

async fn chat(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let stream = body["stream"].as_bool().unwrap_or(false);
    let id = app.next.fetch_add(1, Ordering::Relaxed);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Out>(256);
    if app.jobs.lock().map(|j| j.send(Job { id, body, tx })).is_err() {
        return error(500, error_body(500, "server_error", "the engine thread is gone"));
    }
    if !stream {
        return match rx.recv().await {
            Some(Out::Message(v)) => Json(v).into_response(),
            Some(Out::Error(code, body)) => error(code, body),
            _ => error(500, error_body(500, "server_error", "no answer from the engine")),
        };
    }
    // A request refused before the first chunk is an HTTP error, as llama-server does.
    let first = match rx.recv().await {
        Some(Out::Error(code, body)) => return error(code, body),
        Some(o) => o,
        None => return error(500, error_body(500, "server_error", "no answer from the engine")),
    };
    let events = tokio_stream::once(first).chain(ReceiverStream::new(rx)).map(|o| {
        Ok::<Event, Infallible>(match o {
            Out::Chunk(v) | Out::Message(v) => Event::default().data(v.to_string()),
            Out::Done => Event::default().data("[DONE]"),
            Out::Error(_, body) => Event::default().data(body.to_string()),
        })
    });
    let mut r = Sse::new(events).into_response();
    r.headers_mut().insert(header::CACHE_CONTROL, "no-cache".parse().unwrap());
    r
}
