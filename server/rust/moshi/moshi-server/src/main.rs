// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use candle::Device;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod asr;
mod auth;
mod banner;
mod batched_asr;
mod bench;
mod lm;
mod logging;
mod metrics;
mod mimi;
mod protocol;

mod tts;
mod tts_preprocess;
mod utils;

const ROOM_ID_HEADER: &str = "room_id";

#[derive(clap::Parser, Debug)]
struct WorkerArgs {
    #[clap(short = 'l', long = "log", default_value = "info")]
    log_level: String,

    #[clap(short = 'a', long = "addr", default_value = "0.0.0.0")]
    addr: String,

    #[clap(short = 'p', long = "port", default_value = "8080")]
    port: u16,

    #[clap(long)]
    cpu: bool,

    #[clap(long)]
    config: String,

    #[clap(long)]
    silent: bool,

    /// Maximum size of each log file in MB before rotation (default: 100)
    #[clap(long, default_value = "100")]
    log_max_size_mb: u64,

    /// Maximum number of rotated log files to keep (default: 10)
    #[clap(long, default_value = "10")]
    log_max_files: usize,

    /// Use JSON structured logging
    #[clap(long)]
    json: bool,

    /// Console log style: compact, pretty, or verbose (default: pretty)
    #[clap(long, default_value = "pretty")]
    log_style: String,

    /// Disable CUDA event tracking to reduce overhead (default: false, set --disable-cuda-events to disable)
    #[clap(long)]
    disable_cuda_events: bool,

    /// Enable TF32 for CUDA to speed up matmuls on Ampere+ GPUs (default: true)
    #[clap(long, default_value = "true", action = clap::ArgAction::Set)]
    enable_tf32: bool,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    Validate { configs: Vec<String> },
    Configs { which: String },
    Worker(WorkerArgs),
}

#[derive(clap::Parser, Debug)]
#[clap(name = "server", about = "Kyutai moshi server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TtsConfig {
    pub lm_model_file: String,
    pub text_tokenizer_file: String,
    pub speaker_tokenizer_file: String,
    pub audio_tokenizer_file: String,
    pub voices: std::collections::HashMap<String, String>,
    pub voice_dir: String,
    pub model: moshi::lm::Config,
    pub generation: moshi::tts_streaming::Config,
    #[serde(default)]
    pub log_tokens: bool,
    #[serde(default)]
    pub dtype_override: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AsrConfig {
    pub lm_model_file: String,
    pub text_tokenizer_file: String,
    pub audio_tokenizer_file: String,
    pub model: moshi::lm::Config,
    pub asr_delay_in_tokens: usize,
    #[serde(default)]
    pub log_frequency_s: Option<f64>,
    #[serde(default)]
    pub conditioning_delay: Option<f32>,
    // The default for bools in rust is false.
    #[serde(default)]
    pub conditioning_learnt_padding: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub dtype_override: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MimiConfig {
    pub audio_tokenizer_file: String,
    pub auth_recv: bool,
    pub rooms: Vec<String>,
    pub default_room: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LmConfig {
    pub lm_model_file: String,
    pub text_tokenizer_file: String,
    pub audio_tokenizer_file: String,
    pub model: moshi::lm::Config,
    pub gen: moshi::lm_generate_multistream::Config,
    #[serde(default)]
    pub dtype_override: Option<String>,
}

fn default_warmup_enabled() -> bool {
    true
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WarmupConfig {
    /// Enable or disable eager warmup for supported modules.
    #[serde(default = "default_warmup_enabled")]
    pub enabled: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self { enabled: default_warmup_enabled() }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ModuleConfig {
    Tts {
        path: String,
        #[serde(flatten)]
        config: TtsConfig,
    },
    Asr {
        path: String,
        #[serde(flatten)]
        config: AsrConfig,
    },
    BatchedAsr {
        path: String,
        #[serde(flatten)]
        config: AsrConfig,
        batch_size: usize,
    },
    Mimi {
        send_path: String,
        recv_path: String,
        #[serde(flatten)]
        config: MimiConfig,
    },
    Lm {
        path: String,
        #[serde(flatten)]
        config: LmConfig,
    },
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub static_dir: String,
    pub log_dir: String,
    pub instance_name: String,
    #[serde(default)]
    pub warmup: WarmupConfig,
    #[serde(default)]
    pub modules: std::collections::HashMap<String, ModuleConfig>,
    /// Authentication configuration derived from environment.
    #[serde(skip)]
    #[serde(default)]
    pub auth: auth::AuthConfig,
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        use utils::resolve_or_download as rod;
        let config_str = std::fs::read_to_string(p)?;
        let mut config: Self = toml::from_str(&config_str)?;

        // Derive auth config from environment.
        config.auth = auth::AuthConfig::from_env();

        // Collect all paths that need to be resolved.
        let mut paths = Vec::new();

        // Helper to add a path to our collection list.
        fn add_path(paths: &mut Vec<String>, path: &str) {
            paths.push(path.to_string());
        }

        for (_, c) in config.modules.iter() {
            match c {
                ModuleConfig::Mimi { config: c, .. } => {
                    add_path(&mut paths, &c.audio_tokenizer_file);
                }
                ModuleConfig::Tts { config: c, .. } => {
                    add_path(&mut paths, &c.lm_model_file);
                    add_path(&mut paths, &c.text_tokenizer_file);
                    add_path(&mut paths, &c.speaker_tokenizer_file);
                    add_path(&mut paths, &c.audio_tokenizer_file);
                    for (_, v) in c.voices.iter() {
                        add_path(&mut paths, v);
                    }
                    add_path(&mut paths, &c.voice_dir);
                }
                ModuleConfig::BatchedAsr { config: c, .. } => {
                    add_path(&mut paths, &c.lm_model_file);
                    add_path(&mut paths, &c.text_tokenizer_file);
                    add_path(&mut paths, &c.audio_tokenizer_file);
                }
                ModuleConfig::Asr { config: c, .. } => {
                    add_path(&mut paths, &c.lm_model_file);
                    add_path(&mut paths, &c.text_tokenizer_file);
                    add_path(&mut paths, &c.audio_tokenizer_file);
                }
                ModuleConfig::Lm { config: c, .. } => {
                    add_path(&mut paths, &c.audio_tokenizer_file);
                    add_path(&mut paths, &c.text_tokenizer_file);
                    add_path(&mut paths, &c.lm_model_file);
                }
            }
        }
        add_path(&mut paths, &config.static_dir);
        add_path(&mut paths, &config.log_dir);
        add_path(&mut paths, &config.instance_name);

        // Resolve all paths in parallel.
        use rayon::prelude::*;
        let resolved_paths: Result<std::collections::HashMap<String, String>> = paths
            .into_par_iter()
            .map(|p| {
                let resolved = rod(&p)?;
                Ok((p, resolved))
            })
            .collect();
        let resolved_paths = resolved_paths?;

        // Update the config with resolved paths.
        for (_, c) in config.modules.iter_mut() {
            match c {
                ModuleConfig::Mimi { config: c, .. } => {
                    c.audio_tokenizer_file = resolved_paths[&c.audio_tokenizer_file].clone();
                }
                ModuleConfig::Tts { config: c, .. } => {
                    c.lm_model_file = resolved_paths[&c.lm_model_file].clone();
                    c.text_tokenizer_file = resolved_paths[&c.text_tokenizer_file].clone();
                    c.speaker_tokenizer_file = resolved_paths[&c.speaker_tokenizer_file].clone();
                    c.audio_tokenizer_file = resolved_paths[&c.audio_tokenizer_file].clone();
                    for (_, v) in c.voices.iter_mut() {
                        *v = resolved_paths[v].clone();
                    }
                    c.voice_dir = resolved_paths[&c.voice_dir].clone();
                }
                ModuleConfig::BatchedAsr { config: c, .. } => {
                    c.lm_model_file = resolved_paths[&c.lm_model_file].clone();
                    c.text_tokenizer_file = resolved_paths[&c.text_tokenizer_file].clone();
                    c.audio_tokenizer_file = resolved_paths[&c.audio_tokenizer_file].clone();
                }
                ModuleConfig::Asr { config: c, .. } => {
                    c.lm_model_file = resolved_paths[&c.lm_model_file].clone();
                    c.text_tokenizer_file = resolved_paths[&c.text_tokenizer_file].clone();
                    c.audio_tokenizer_file = resolved_paths[&c.audio_tokenizer_file].clone();
                }
                ModuleConfig::Lm { config: c, .. } => {
                    c.audio_tokenizer_file = resolved_paths[&c.audio_tokenizer_file].clone();
                    c.text_tokenizer_file = resolved_paths[&c.text_tokenizer_file].clone();
                    c.lm_model_file = resolved_paths[&c.lm_model_file].clone();
                }
            }
        }
        config.static_dir = resolved_paths[&config.static_dir].clone();
        config.log_dir = resolved_paths[&config.log_dir].clone();
        config.instance_name = resolved_paths[&config.instance_name].clone();
        Ok(config)
    }
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[allow(unused)]
enum Module {
    Tts { path: String, m: Arc<tts::Model> },
    Asr { path: String, m: Arc<asr::Asr> },
    BatchedAsr { path: String, m: Arc<batched_asr::BatchedAsr> },
    Mimi { send_path: String, recv_path: String, m: Arc<mimi::Mimi> },
    Lm { path: String, m: Arc<lm::Lm> },
}

struct SharedStateInner {
    config: Config,
}

type SharedState = Arc<SharedStateInner>;

fn lm_router(s: Arc<lm::Lm>, path: &str) -> axum::Router<()> {
    async fn lm_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<lm::Lm>,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket).await {
            tracing::error!(?err, "lm")
        }
    }

    #[tracing::instrument(skip(ws, headers, state), fields(client_ip))]
    async fn lm_streaming(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<Arc<lm::Lm>>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        if let Some(ip) = &addr {
            tracing::Span::current().record("client_ip", ip);
        }
        tracing::info!("handling lm-streaming query");
        let state = state.0.clone();
        let upg = ws
            .write_buffer_size(0)
            .protocols(["permessage-deflate"])
            .on_upgrade(move |v| lm_websocket(v, state, addr));
        Ok(upg)
    }

    axum::Router::new().route(path, axum::routing::get(lm_streaming)).with_state(s)
}

impl Module {
    fn run_warmup<F>(
        module: &str,
        path: &str,
        warmup_cfg: &WarmupConfig,
        warmup_fn: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        use crate::metrics::warmup as warmup_metrics;

        if !warmup_cfg.enabled {
            tracing::info!(module, path, "skipping warmup (disabled)");
            warmup_metrics::SKIPPED.inc();
            return Ok(());
        }

        let start = Instant::now();
        tracing::info!(module, path, "starting warmup");
        let res = warmup_fn();
        let elapsed = start.elapsed().as_secs_f64();

        match &res {
            Ok(_) => {
                warmup_metrics::DURATION.observe(elapsed);
                warmup_metrics::SUCCESS.inc();
                tracing::info!(module, path, duration_ms = (elapsed * 1000.0), "warmup completed");
            }
            Err(err) => {
                warmup_metrics::DURATION.observe(elapsed);
                warmup_metrics::FAILURE.inc();
                tracing::error!(
                    module,
                    path,
                    duration_ms = (elapsed * 1000.0),
                    ?err,
                    "warmup failed"
                );
            }
        }

        res
    }

    fn new(
        module_cfg: &ModuleConfig,
        full_cfg: &Config,
        dev: &Device,
        warmup_cfg: &WarmupConfig,
    ) -> Result<Self> {
        let m = match module_cfg {
            ModuleConfig::Lm { path, config } => {
                let m = lm::Lm::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                Self::Lm { m, path: path.to_string() }
            }
            ModuleConfig::Asr { path, config } => {
                let m = asr::Asr::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                Self::run_warmup("asr", path, warmup_cfg, || m.warmup())?;
                Self::Asr { m, path: path.to_string() }
            }
            ModuleConfig::BatchedAsr { path, config, batch_size } => {
                let m = batched_asr::BatchedAsr::new(
                    *batch_size,
                    config,
                    full_cfg,
                    dev,
                    warmup_cfg.enabled,
                )?;
                let m = Arc::new(m);
                Self::BatchedAsr { m, path: path.to_string() }
            }
            ModuleConfig::Tts { path, config } => {
                let voice = config.voices.keys().next();
                let m = tts::Model::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                if let Some(voice) = voice {
                    let voice = voice.clone();
                    Self::run_warmup("tts", path, warmup_cfg, || {
                        m.run(&TtsQuery {
                            text: vec!["hello".to_string()],
                            seed: 42,
                            temperature: 0.8,
                            top_k: 250,
                            voice: Some(voice.clone()),
                            voices: None,
                            max_seq_len: None,
                            return_timestamps: None,
                            cfg_alpha: None,
                        })
                        .map(|_| ())
                        .map_err(Into::into)
                    })?;
                } else {
                    tracing::info!(path, "skipping tts warmup (no voices configured)");
                }
                Self::Tts { m, path: path.to_string() }
            }
            ModuleConfig::Mimi { send_path, recv_path, config } => {
                let m = mimi::Mimi::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                Self::Mimi { m, send_path: send_path.to_string(), recv_path: recv_path.to_string() }
            }
        };
        Ok(m)
    }

    fn router(&self, shared_state: &SharedState) -> Result<axum::Router<()>> {
        let router = match self {
            Self::Lm { path, m } => lm_router(m.clone(), path),
            Self::Asr { path, m } => asr_router(m.clone(), path, shared_state),
            Self::BatchedAsr { path, m } => batched_asr_router(m.clone(), path, shared_state),
            Self::Tts { path, m } => tts_router(m.clone(), path, shared_state),
            Self::Mimi { send_path, recv_path, m } => {
                mimi_router(m.clone(), send_path, recv_path, shared_state)
            }
        };
        Ok(router)
    }
}

struct AppStateInner {
    modules: Vec<Module>,
}

type AppState = Arc<AppStateInner>;

impl AppStateInner {
    async fn new(args: &WorkerArgs, config: Config) -> Result<Self> {
        let device = device(args.cpu)?;

        #[cfg(feature = "cuda")]
        if let candle::Device::Cuda(d) = &device {
            unsafe {
                if args.disable_cuda_events {
                    tracing::info!("disabling CUDA event tracking");
                    d.disable_event_tracking();
                }
                /*
                if args.enable_tf32 {
                    tracing::info!("enabling TF32");
                    d.set_tf32(true);
                }
                */
            }
        };

        let mut modules_f = Vec::with_capacity(config.modules.len());
        for (_, module_cfg) in config.modules.iter() {
            let config = config.clone();
            let device = device.clone();
            let module_cfg = module_cfg.clone();
            modules_f.push(tokio::task::spawn_blocking(move || {
                Module::new(&module_cfg, &config, &device, &config.warmup)
            }));
        }
        let mut modules = Vec::with_capacity(modules_f.len());
        for m in modules_f {
            modules.push(m.await??);
        }
        Ok(Self { modules })
    }
}

/// Configuration for log rotation
struct LogConfig {
    log_dir: String,
    instance_name: String,
    log_level: String,
    silent: bool,
    max_size_mb: u64,
    max_files: usize,
    json: bool,
    log_style: logging::LogStyle,
}

fn tracing_init(config: LogConfig) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use std::io::IsTerminal;
    use tracing_rolling_file::{RollingConditionBase, RollingFileAppenderBase};
    use tracing_subscriber::fmt::time::ChronoLocal;
    use tracing_subscriber::prelude::*;

    let build_info = utils::BuildInfo::new();

    // Create log directory if it doesn't exist
    std::fs::create_dir_all(&config.log_dir)?;

    // Build rolling file appender with size-based rotation and max file count
    // Uses Debian-style naming: log.instance, log.instance.1, log.instance.2, etc.
    let log_path =
        std::path::Path::new(&config.log_dir).join(format!("log.{}", config.instance_name));

    // Create rolling condition: rotate daily OR when file exceeds max_size_mb
    let condition = RollingConditionBase::new().daily().max_size(config.max_size_mb * 1024 * 1024); // Convert MB to bytes

    let file_appender = RollingFileAppenderBase::new(log_path, condition, config.max_files)?;

    // Get non-blocking writer for async file writes
    let (non_blocking_file, guard) = tracing_appender::non_blocking(file_appender);

    let filter = tracing_subscriber::filter::LevelFilter::from_str(&config.log_level)?;

    // Custom timestamp format: "2025-12-02 01:36:42.113" (more readable than ISO 8601)
    let timer = ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".to_string());

    // File layer: NO ANSI colors, clean timestamps (or JSON)
    let file_layer = if config.json {
        let json_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_timer(timer.clone())
            .with_file(true)
            .with_line_number(true)
            .with_target(true)
            .with_writer(non_blocking_file)
            .with_filter(filter.clone());

        json_layer.boxed()
    } else {
        let text_layer = tracing_subscriber::fmt::layer()
            .event_format(
                tracing_subscriber::fmt::format()
                    .with_timer(timer.clone())
                    .with_file(true)
                    .with_line_number(true)
                    .with_target(true)
                    .with_ansi(false), // No ANSI escape codes in log files
            )
            .with_writer(non_blocking_file)
            .with_filter(filter.clone());

        text_layer.boxed()
    };

    if config.silent {
        // File-only logging
        tracing_subscriber::registry().with(file_layer).init();
    } else {
        // Console layer: WITH custom formatter for terminal (or JSON)
        let console_layer = if config.json {
            let json_layer = tracing_subscriber::fmt::layer()
                .json()
                .with_timer(timer)
                .with_file(true)
                .with_line_number(true)
                .with_target(true)
                .with_writer(std::io::stdout)
                .with_filter(filter);

            json_layer.boxed()
        } else {
            let use_ansi = std::io::stdout().is_terminal();
            let show_file = config.log_style == logging::LogStyle::Verbose;

            // Use custom pretty formatter with level icons
            let pretty_formatter = logging::PrettyFormatter::new(timer)
                .with_ansi(use_ansi)
                .with_file(show_file)
                .with_target(true)
                .with_style(config.log_style);

            let text_layer = tracing_subscriber::fmt::layer()
                .event_format(pretty_formatter)
                .with_writer(std::io::stdout)
                .with_filter(filter);

            text_layer.boxed()
        };

        tracing_subscriber::registry().with(file_layer).with(console_layer).init();
    }

    tracing::info!(?build_info);
    tracing::info!(
        log_dir = %config.log_dir,
        max_size_mb = config.max_size_mb,
        max_files = config.max_files,
        json = config.json,
        "Logging initialized with rotation"
    );

    Ok(guard)
}

async fn metrics(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    _state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    use prometheus::Encoder;

    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    if let Err(err) = encoder.encode(&metric_families, &mut buffer) {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response();
    };
    axum::response::Response::builder()
        .status(200)
        .header(axum::http::header::CONTENT_TYPE, encoder.format_type())
        .body(axum::body::Body::from(buffer))
        .unwrap()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // When an error bubbles up in the tokio main function, the whole program does not
    // seem to crash if some background tasks are still running.
    // This can lead to errors such as "port already in use" not being reported so we
    // exit the process explicitely here.
    if let Err(err) = main_().await {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

async fn main_() -> Result<()> {
    // Load .env file if present (before parsing args so env vars are available)
    dotenvy::dotenv().ok();

    let args = <Args as clap::Parser>::parse();
    match args.command {
        Command::Configs { which } => {
            eprintln!(
                "The 'configs' command has been removed. Python scripts are no longer embedded."
            );
            eprintln!("Unknown config: {which}");
            std::process::exit(1);
        }
        Command::Validate { configs } => {
            tracing_subscriber::fmt().init();
            for config in configs.iter() {
                let _ = Config::load(config)?;
                tracing::info!(?config, "loaded succesfully")
            }
        }
        Command::Worker(args) => {
            use axum::routing::get;

            let mut config = Config::load(&args.config)?;

            // Initialize logging first so GPU detection logs are visible
            if std::env::var("RUST_LOG").is_err() {
                std::env::set_var("RUST_LOG", format!("{},hyper=info,mio=info", args.log_level))
            }
            // Parse log style (default to pretty if invalid)
            let log_style = args.log_style.parse().unwrap_or_else(|e: String| {
                eprintln!("Warning: {e}, using 'pretty'");
                logging::LogStyle::Pretty
            });

            let log_config = LogConfig {
                log_dir: config.log_dir.clone(),
                instance_name: config.instance_name.clone(),
                log_level: args.log_level.clone(),
                silent: args.silent,
                max_size_mb: args.log_max_size_mb,
                max_files: args.log_max_files,
                json: args.json,
                log_style,
            };
            let _guard = tracing_init(log_config)?;

            // Print startup banner (before tracing span so it appears first)
            let banner = banner::ServerBanner::new();
            if !args.silent {
                banner.print_logo();
                let version = utils::BuildInfo::new().git_describe();
                if banner::supports_color() {
                    println!(
                        "  {} v{}\n",
                        owo_colors::OwoColorize::bold(&"moshi-server"),
                        owo_colors::OwoColorize::bright_white(&version)
                    );
                } else {
                    println!("  moshi-server v{}\n", version);
                }
            }

            // Create a span for the startup sequence
            let startup_span = tracing::info_span!("startup");
            let _enter = startup_span.enter();

            // Log Better Auth status (after tracing is initialized)
            let auth_enabled = std::env::var("BETTER_AUTH_SECRET").is_ok();
            if auth_enabled {
                tracing::info!("Better Auth JWT validation enabled (BETTER_AUTH_SECRET is set)");
            }

            // Variables to collect for banner
            let mut gpu_name: Option<String> = None;
            let mut gpu_vram_mb: Option<u64> = None;
            let mut effective_batch_size: Option<usize> = None;

            // Auto-detect GPU capabilities and adjust configuration
            if let Ok(gpu_info) = utils::get_gpu_info() {
                gpu_name = Some(gpu_info.name.clone());
                gpu_vram_mb = Some(gpu_info.total_vram_mb());
                // Extract model info from the first LM-bearing module for logging
                let model_info = config.modules.values().find_map(|m| match m {
                    ModuleConfig::Lm { config: c, .. } => Some(utils::ModelInfo::from_lm_config(c)),
                    ModuleConfig::Asr { config: c, .. } => {
                        Some(utils::ModelInfo::from_asr_config(c))
                    }
                    ModuleConfig::BatchedAsr { config: c, .. } => {
                        Some(utils::ModelInfo::from_asr_config(c))
                    }
                    ModuleConfig::Tts { config: c, .. } => {
                        Some(utils::ModelInfo::from_tts_config(c))
                    }
                    _ => None,
                });

                // Log combined GPU and model summary
                gpu_info.log_combined_summary(model_info.as_ref());

                // Get recommended dtype based on GPU compute capability
                let auto_dtype = gpu_info.recommended_dtype();

                // Read configuration from environment or use defaults
                let model_params_billions: f64 = std::env::var("MOSHI_MODEL_PARAMS_BILLIONS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(utils::DEFAULT_MODEL_PARAMS_BILLIONS);

                let per_batch_item_mb: u64 = std::env::var("MOSHI_PER_BATCH_ITEM_MB")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| {
                        let mut estimated: Option<u64> = None;
                        for (module_name, module_cfg) in config.modules.iter() {
                            let (model_cfg, dtype_override) = match module_cfg {
                                ModuleConfig::BatchedAsr { config: c, .. } => {
                                    (Some(&c.model), c.dtype_override.as_deref())
                                }
                                ModuleConfig::Asr { config: c, .. } => {
                                    (Some(&c.model), c.dtype_override.as_deref())
                                }
                                ModuleConfig::Tts { config: c, .. } => {
                                    (Some(&c.model), c.dtype_override.as_deref())
                                }
                                ModuleConfig::Lm { config: c, .. } => {
                                    (Some(&c.model), c.dtype_override.as_deref())
                                }
                                _ => (None, None),
                            };
                            let model_cfg = match model_cfg {
                                Some(model_cfg) => model_cfg,
                                None => continue,
                            };
                            let dtype_label = dtype_override.unwrap_or(auto_dtype);
                            let dtype_bytes = match dtype_label {
                                "bf16" | "f16" => 2,
                                _ => 4,
                            };
                            let estimate = utils::estimate_per_batch_item_mb(
                                model_cfg,
                                dtype_bytes,
                                Some(&gpu_info),
                            );
                            tracing::debug!(
                                module = module_name,
                                dtype = dtype_label,
                                raw_kv_mb = estimate.raw_kv_mb,
                                overhead_multiplier = estimate.overhead_multiplier,
                                per_batch_item_mb = estimate.estimated_mb,
                                "Estimated per-batch memory"
                            );
                            estimated = Some(
                                estimated
                                    .map_or(estimate.estimated_mb, |v| v.max(estimate.estimated_mb)),
                            );
                        }
                        if let Some(estimated) = estimated {
                            tracing::info!(
                                per_batch_item_mb = estimated,
                                "Using estimated per-batch memory; override with MOSHI_PER_BATCH_ITEM_MB"
                            );
                            estimated
                        } else {
                            utils::DEFAULT_PER_BATCH_ITEM_MB
                        }
                    });

                // Calculate batch size with detailed breakdown
                let batch_calc =
                    gpu_info.calculate_batch_size(model_params_billions, per_batch_item_mb);
                batch_calc.log_breakdown();

                if batch_calc.available_for_batching_mb == 0 {
                    tracing::error!(
                        "CRITICAL: VRAM insufficient for model weights + reserved memory. \
                         Startup will likely fail with CUDA_ERROR_OUT_OF_MEMORY. \
                         Try using a lower precision model, reducing VRAM_RESERVED_MB, or switching to the Low-RAM config."
                    );
                }

                let max_safe_batch_size = batch_calc.recommended_batch_size;

                for (name, module_cfg) in config.modules.iter_mut() {
                    match module_cfg {
                        ModuleConfig::BatchedAsr { batch_size, config: asr_config, .. } => {
                            // Auto-set dtype_override if not specified
                            if asr_config.dtype_override.is_none() {
                                tracing::info!(
                                    module = name,
                                    dtype = auto_dtype,
                                    "Auto-setting dtype_override for BatchedAsr"
                                );
                                asr_config.dtype_override = Some(auto_dtype.to_string());
                            }

                            // Adjust batch size if too large
                            if *batch_size > max_safe_batch_size {
                                tracing::warn!(
                                    module = name,
                                    configured = *batch_size,
                                    adjusted = max_safe_batch_size,
                                    "Reducing batch size due to VRAM constraints"
                                );
                                *batch_size = max_safe_batch_size.max(1);
                            }
                            effective_batch_size = Some(*batch_size);
                        }
                        ModuleConfig::Asr { config: asr_config, .. } => {
                            if asr_config.dtype_override.is_none() {
                                tracing::info!(
                                    module = name,
                                    dtype = auto_dtype,
                                    "Auto-setting dtype_override for Asr"
                                );
                                asr_config.dtype_override = Some(auto_dtype.to_string());
                            }
                        }
                        ModuleConfig::Tts { config: tts_config, .. } => {
                            if tts_config.dtype_override.is_none() {
                                tracing::info!(
                                    module = name,
                                    dtype = auto_dtype,
                                    "Auto-setting dtype_override for Tts"
                                );
                                tts_config.dtype_override = Some(auto_dtype.to_string());
                            }
                        }
                        ModuleConfig::Lm { config: lm_config, .. } => {
                            if lm_config.dtype_override.is_none() {
                                tracing::info!(
                                    module = name,
                                    dtype = auto_dtype,
                                    "Auto-setting dtype_override for Lm"
                                );
                                lm_config.dtype_override = Some(auto_dtype.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            } else {
                tracing::warn!("Could not detect GPU capabilities. Using configured values.");
            }

            let num_workers = tokio::runtime::Handle::current().metrics().num_workers();
            tracing::info!(num_workers, "starting worker");

            let static_dir = utils::resolve_or_download(&config.static_dir)?;
            let shared_state = Arc::new(SharedStateInner { config: config.clone() });
            let state = Arc::new(AppStateInner::new(&args, config).await?);
            // Initialize server start time for uptime tracking
            init_server_start_time();

            // Start background metrics updater
            spawn_metrics_updater();

            // Print configuration summary box (if not silent)
            if !args.silent {
                // Collect module info for the banner
                let module_infos: Vec<banner::ModuleInfo> = shared_state
                    .config
                    .modules
                    .iter()
                    .map(|(name, cfg)| {
                        let (module_type, path) = match cfg {
                            ModuleConfig::Tts { path, .. } => ("TTS", path.clone()),
                            ModuleConfig::Asr { path, .. } => ("ASR", path.clone()),
                            ModuleConfig::BatchedAsr { path, .. } => ("BatchedASR", path.clone()),
                            ModuleConfig::Mimi { send_path, .. } => ("Mimi", send_path.clone()),
                            ModuleConfig::Lm { path, .. } => ("LM", path.clone()),
                        };
                        banner::ModuleInfo {
                            name: name.clone(),
                            module_type: module_type.to_string(),
                            path,
                        }
                    })
                    .collect();

                let banner_config = banner::BannerConfig {
                    version: utils::BuildInfo::new().git_describe(),
                    addr: args.addr.clone(),
                    port: args.port,
                    modules: module_infos,
                    auth_enabled,
                    gpu_name,
                    gpu_vram_mb,
                    batch_size: effective_batch_size,
                    instance_name: shared_state.config.instance_name.clone(),
                };

                banner.print_banner(&banner_config);
            }

            // End startup span before starting the server
            drop(_enter);

            let mut app = axum::Router::new()
                .route("/api/status", get(server_status))
                .route("/api/health", get(health_check))
                .route("/api/build_info", get(build_info))
                .route("/api/modules_info", get(modules_info))
                .route("/metrics", axum::routing::get(metrics))
                .fallback_service(
                    tower_http::services::ServeDir::new(&static_dir)
                        .append_index_html_on_directories(true),
                )
                .layer(
                    tower::ServiceBuilder::new()
                        .layer(tower_http::request_id::SetRequestIdLayer::x_request_id(
                            tower_http::request_id::MakeRequestUuid,
                        ))
                        .layer(tower_http::trace::TraceLayer::new_for_http()),
                )
                .with_state(state.clone());
            for module in state.modules.iter() {
                app = app.merge(module.router(&shared_state)?)
            }

            let sock_addr = std::net::SocketAddr::from((
                std::net::IpAddr::from_str(args.addr.as_str())
                    .unwrap_or(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
                args.port,
            ));
            tracing::info!("listening on {}", sock_addr);
            let listener = tokio::net::TcpListener::bind(sock_addr).await?;
            axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>())
                .await?
        }
    }
    Ok(())
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingOutput {
    Pcm,
    PcmMessagePack,
    OggOpus,
    OggOpusMessagePack,
}
fn default_seed() -> u64 {
    42
}
fn default_temperature() -> f64 {
    0.8
}
fn default_top_k() -> usize {
    250
}
fn default_format() -> StreamingOutput {
    StreamingOutput::OggOpus
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct TtsStreamingQuery {
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_format")]
    format: StreamingOutput,
    voice: Option<String>,
    voices: Option<Vec<String>>,
    max_seq_len: Option<usize>,
    cfg_alpha: Option<f64>,
    /// JWT token for authentication (alternative to Authorization header)
    token: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct TtsQuery {
    text: Vec<String>,
    seed: u64,
    temperature: f64,
    top_k: usize,
    voice: Option<String>,
    voices: Option<Vec<String>>,
    max_seq_len: Option<usize>,
    return_timestamps: Option<bool>,
    cfg_alpha: Option<f64>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct TtsResponse {
    wav: String,
    transcript: Vec<crate::tts::WordWithTimestamps>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::warmup as warmup_metrics;
    use std::sync::{Mutex, OnceLock};

    fn metric_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn warmup_success_increments_success_counter() {
        let _guard = metric_lock();
        let before = warmup_metrics::SUCCESS.get();
        Module::run_warmup(
            "asr",
            "/asr",
            &WarmupConfig { enabled: true },
            || -> anyhow::Result<()> { Ok(()) },
        )
        .unwrap();
        let after = warmup_metrics::SUCCESS.get();
        assert!(
            (after - before - 1.0).abs() < f64::EPSILON,
            "expected success counter to increment by 1 (before {before}, after {after})"
        );
    }

    #[test]
    fn warmup_failure_increments_failure_counter() {
        let _guard = metric_lock();
        let before = warmup_metrics::FAILURE.get();
        let res = Module::run_warmup(
            "asr",
            "/asr",
            &WarmupConfig { enabled: true },
            || -> anyhow::Result<()> { anyhow::bail!("boom") },
        );
        assert!(res.is_err(), "expected warmup to fail");
        let after = warmup_metrics::FAILURE.get();
        assert!(
            (after - before - 1.0).abs() < f64::EPSILON,
            "expected failure counter to increment by 1 (before {before}, after {after})"
        );
    }

    #[test]
    fn warmup_skipped_increments_skipped_counter() {
        let _guard = metric_lock();
        let before = warmup_metrics::SKIPPED.get();
        Module::run_warmup(
            "tts",
            "/tts",
            &WarmupConfig { enabled: false },
            || -> anyhow::Result<()> { Ok(()) },
        )
        .unwrap();
        let after = warmup_metrics::SKIPPED.get();
        assert!(
            (after - before - 1.0).abs() < f64::EPSILON,
            "expected skipped counter to increment by 1 (before {before}, after {after})"
        );
    }
}

fn tts_router(s: Arc<tts::Model>, path: &str, ss: &SharedState) -> axum::Router<()> {
    use base64::Engine;

    async fn t(
        state: axum::extract::State<(Arc<tts::Model>, SharedState)>,
        headers: axum::http::HeaderMap,
        req: axum::Json<TtsQuery>,
    ) -> utils::AxumResult<Response> {
        tracing::debug!("handling tts query {req:?}");
        match auth::check_with_user(&headers, None) {
            Ok(claims) => {
                tracing::debug!(user_id = %claims.user.id, session_id = %claims.session.id, "authenticated via JWT");
            }
            Err(err) => return Ok(err.into_response()),
        }
        let (wav, transcript) = {
            let _guard = state.0 .0.mutex.lock().await;
            state.0 .0.run(&req)?
        };
        tracing::debug!("ok {}", wav.len());
        if req.return_timestamps.unwrap_or(false) {
            let data =
                TtsResponse { wav: base64::prelude::BASE64_STANDARD.encode(wav), transcript };
            Ok((
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                axum::Json(data),
            )
                .into_response())
        } else {
            Ok((StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "audio/wav")], wav)
                .into_response())
        }
    }

    #[tracing::instrument(skip(ws, headers, state), fields(client_ip))]
    async fn streaming_t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<tts::Model>, SharedState)>,
        req: axum::extract::Query<TtsStreamingQuery>,
    ) -> utils::AxumResult<Response> {
        tracing::debug!("handling tts streaming query {req:?}");
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        if let Some(ip) = &addr {
            tracing::Span::current().record("client_ip", ip);
        }
        let auth_result = auth::check_with_user(&headers, req.token.as_deref());

        let tts_query = req.0.clone();
        let tts = state.0 .0.clone();
        let upg =
            ws.write_buffer_size(0).protocols(["permessage-deflate"]).on_upgrade(move |mut socket| async move {
                match &auth_result {
                    Err(err) => {
                        tracing::warn!(?err, "WebSocket auth failed, closing with 4001");
                        let _ = crate::utils::close_with_reason(
                            &mut socket,
                            crate::protocol::CloseCode::AuthenticationFailed,
                            Some("Authentication failed"),
                        ).await;
                        return;
                    }
                    Ok(claims) => {
                        tracing::debug!(user_id = %claims.user.id, session_id = %claims.session.id, "authenticated via JWT");
                    }
                }
                if let Err(err) = tts.handle_socket(socket, tts_query).await {
                    tracing::error!(?err, "tts socket handler failed");
                }
            });
        Ok(upg)
    }

    axum::Router::new()
        .route(path, axum::routing::post(t))
        .route(&format!("{path}_streaming"), axum::routing::get(streaming_t))
        .with_state((s, ss.clone()))
}

async fn build_info(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    _state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    let build_info = utils::BuildInfo::new();
    utils::WrapJson(Ok(build_info)).into_response()
}

// ============================================================================
// Server Status Endpoint
// ============================================================================

/// Response structure for /api/status endpoint
#[derive(serde::Serialize, Debug)]
struct StatusResponse {
    /// Server status: "healthy", "degraded", or "unhealthy"
    status: &'static str,
    /// Server uptime in seconds
    uptime_seconds: u64,
    /// ISO 8601 timestamp when server started
    started_at: String,
    /// Build information
    build: utils::BuildInfo,
    /// Module capacity information
    capacity: CapacityInfo,
    /// Authentication configuration (without secrets)
    auth: AuthInfo,
}

/// Capacity information for all modules
#[derive(serde::Serialize, Debug)]
struct CapacityInfo {
    /// Total slots across all batched modules
    total_slots: usize,
    /// Used slots across all batched modules
    used_slots: usize,
    /// Available slots (total - used)
    available_slots: usize,
    /// Per-module breakdown
    modules: Vec<ModuleCapacity>,
}

/// Capacity information for a single module
#[derive(serde::Serialize, Debug)]
struct ModuleCapacity {
    /// Module name/path
    name: String,
    /// Module type (batched_asr, py_batched_asr, py)
    module_type: &'static str,
    /// Total slots for this module
    total_slots: usize,
    /// Used slots for this module
    used_slots: usize,
    /// Available slots for this module
    available_slots: usize,
}

/// Authentication configuration (without secrets)
#[derive(serde::Serialize, Debug)]
struct AuthInfo {
    /// Whether API key auth is configured
    api_key_configured: bool,
    /// Whether Better Auth JWT validation is enabled
    better_auth_enabled: bool,
}

/// Global server start time (set once at startup)
static SERVER_START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
static SERVER_START_TIMESTAMP: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Initialize server start time (call once at startup)
fn init_server_start_time() {
    SERVER_START_TIME.get_or_init(std::time::Instant::now);
    SERVER_START_TIMESTAMP
        .get_or_init(|| chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
}

/// Get server uptime in seconds
fn get_uptime_seconds() -> u64 {
    SERVER_START_TIME.get().map(|start| start.elapsed().as_secs()).unwrap_or(0)
}

fn spawn_metrics_updater() {
    utils::spawn("metrics_updater", async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            interval.tick().await;

            if let Ok(info) = utils::get_gpu_info() {
                use crate::metrics::system;
                system::FREE_VRAM.set(info.free_vram as f64);
                system::TOTAL_VRAM.set(info.total_vram as f64);
                system::USED_VRAM.set((info.total_vram.saturating_sub(info.free_vram)) as f64);
                system::GPU_UTILIZATION.set(info.utilization as f64);
            }
        }
    });
}

async fn server_status(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    // Collect capacity info from all modules
    let mut total_slots = 0usize;
    let mut used_slots = 0usize;
    let mut modules = Vec::new();

    for module in state.modules.iter() {
        match module {
            Module::BatchedAsr { path, m } => {
                let t = m.total_slots();
                let u = m.used_slots();
                total_slots += t;
                used_slots += u;
                modules.push(ModuleCapacity {
                    name: path.clone(),
                    module_type: "batched_asr",
                    total_slots: t,
                    used_slots: u,
                    available_slots: t.saturating_sub(u),
                });
            }
            _ => {}
        }
    }

    let available_slots = total_slots.saturating_sub(used_slots);

    // Determine overall status
    let status = if available_slots == 0 && total_slots > 0 {
        "degraded" // At capacity
    } else {
        "healthy"
    };

    let response = StatusResponse {
        status,
        uptime_seconds: get_uptime_seconds(),
        started_at: SERVER_START_TIMESTAMP.get().cloned().unwrap_or_else(|| "unknown".to_string()),
        build: utils::BuildInfo::new(),
        capacity: CapacityInfo { total_slots, used_slots, available_slots, modules },
        auth: AuthInfo {
            api_key_configured: std::env::var("MOSHI_API_KEY").is_ok(),
            better_auth_enabled: std::env::var("BETTER_AUTH_SECRET").is_ok(),
        },
    };

    utils::WrapJson(Ok(response)).into_response()
}

/// Simple health check endpoint returning JSON
async fn health_check() -> impl IntoResponse {
    #[derive(serde::Serialize)]
    struct HealthResponse {
        status: &'static str,
        uptime_seconds: u64,
    }

    axum::Json(HealthResponse { status: "ok", uptime_seconds: get_uptime_seconds() })
}

async fn modules_info(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    let modules: Vec<_> = state
        .modules
        .iter()
        .filter_map(|m| match m {
            Module::BatchedAsr { path, m } => {
                let config = m.config();
                let mut info = std::collections::HashMap::new();
                info.insert("type", "batched_asr".to_string());
                info.insert("path", path.to_string());
                info.insert("lm", config.lm_model_file.clone());
                info.insert("audio_tokenizer", config.audio_tokenizer_file.clone());
                info.insert("used_slots", m.used_slots().to_string());
                info.insert("total_slots", m.total_slots().to_string());
                Some(info)
            }
            _ => None,
        })
        .collect();
    utils::WrapJson(Ok(modules)).into_response()
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct AsrStreamingQuery {
    /// JWT token for authentication (alternative to Authorization header)
    token: Option<String>,
}

fn asr_router(s: Arc<asr::Asr>, path: &str, ss: &SharedState) -> axum::Router<()> {
    async fn asr_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<asr::Asr>,
        query: AsrStreamingQuery,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket, query).await {
            tracing::error!(?err, "asr")
        }
    }

    async fn health() -> impl IntoResponse {
        StatusCode::OK
    }

    #[tracing::instrument(skip(ws, headers, state), fields(client_ip))]
    async fn t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<asr::Asr>, SharedState)>,
        req: axum::extract::Query<AsrStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        if let Some(ip) = &addr {
            tracing::Span::current().record("client_ip", ip);
        }
        tracing::info!("handling asr-streaming query");
        let auth_result = auth::check(&headers, req.token.as_deref());

        let asr_query = req.0.clone();
        let asr = state.0 .0.clone();
        let upg = ws.write_buffer_size(0).protocols(["permessage-deflate"]).on_upgrade(
            move |mut socket| async move {
                if let Err(err) = auth_result {
                    tracing::warn!(?err, "WebSocket auth failed, closing with 4001");
                    let _ = crate::utils::close_with_reason(
                        &mut socket,
                        crate::protocol::CloseCode::AuthenticationFailed,
                        Some("Authentication failed"),
                    )
                    .await;
                    return;
                }
                asr_websocket(socket, asr, asr_query, addr).await
            },
        );
        Ok(upg)
    }
    axum::Router::new()
        .route(path, axum::routing::get(t))
        .route(&format!("{path}/health"), axum::routing::get(health))
        .with_state((s, ss.clone()))
}

fn batched_asr_router(
    s: Arc<batched_asr::BatchedAsr>,
    path: &str,
    ss: &SharedState,
) -> axum::Router<()> {
    async fn asr_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<batched_asr::BatchedAsr>,
        query: AsrStreamingQuery,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket, query).await {
            tracing::error!(?err, "asr")
        }
    }

    async fn health() -> impl IntoResponse {
        StatusCode::OK
    }

    // TODO: add a batch mode.
    async fn t(
        state: axum::extract::State<(Arc<batched_asr::BatchedAsr>, SharedState)>,
        headers: axum::http::HeaderMap,
        req: axum::body::Bytes,
    ) -> utils::AxumResult<Response> {
        tracing::info!(len = req.len(), "handling asr post query");
        if let Err(err) = auth::check(&headers, None) {
            return Ok(err.into_response());
        }
        let transcript = state.0 .0.handle_query(req).await?;
        Ok((
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            axum::Json(transcript),
        )
            .into_response())
    }

    #[tracing::instrument(skip(ws, headers, state), fields(client_ip))]
    async fn streaming_t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<batched_asr::BatchedAsr>, SharedState)>,
        req: axum::extract::Query<AsrStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        if let Some(ip) = &addr {
            tracing::Span::current().record("client_ip", ip);
        }
        tracing::info!("handling batched asr-streaming query");
        let auth_result = auth::check(&headers, req.token.as_deref());

        let asr_query = req.0.clone();
        let asr = state.0 .0.clone();
        let upg = ws.write_buffer_size(0).protocols(["permessage-deflate"]).on_upgrade(
            move |mut socket| async move {
                if let Err(err) = auth_result {
                    tracing::warn!(?err, "WebSocket auth failed, closing with 4001");
                    let _ = crate::utils::close_with_reason(
                        &mut socket,
                        crate::protocol::CloseCode::AuthenticationFailed,
                        Some("Authentication failed"),
                    )
                    .await;
                    return;
                }
                asr_websocket(socket, asr, asr_query, addr).await
            },
        );
        Ok(upg)
    }
    axum::Router::new()
        .route(path, axum::routing::post(t))
        .route(path, axum::routing::get(streaming_t))
        .route(&format!("{path}/health"), axum::routing::get(health))
        .with_state((s, ss.clone()))
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct MimiStreamingQuery {
    /// JWT token for authentication (alternative to Authorization header)
    token: Option<String>,
    room_id: Option<String>,
}

fn mimi_router(
    s: Arc<mimi::Mimi>,
    send_path: &str,
    recv_path: &str,
    ss: &SharedState,
) -> axum::Router<()> {
    async fn mimi_recv_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<mimi::Mimi>,
        room_id: Option<String>,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.recv_socket(socket, room_id).await {
            tracing::error!(?err, "mimi")
        }
    }

    async fn recv(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<mimi::Mimi>, SharedState)>,
        req: axum::extract::Query<MimiStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling mimi-streaming query");
        // It's tricky to set the headers of a websocket in javascript so we pass the token via the
        // query too.
        let auth_result = if state.0 .0.auth_recv() {
            auth::check(&headers, req.token.as_deref())
        } else {
            Ok(())
        };

        let room_id = match headers.get(ROOM_ID_HEADER) {
            Some(v) => v.to_str().ok().map(|v| v.to_string()),
            None => req.room_id.clone(),
        };
        let state = state.0 .0.clone();
        let upg = ws.write_buffer_size(0).protocols(["permessage-deflate"]).on_upgrade(
            move |mut socket| async move {
                if let Err(err) = auth_result {
                    tracing::warn!(?err, "WebSocket auth failed, closing with 4001");
                    let _ = crate::utils::close_with_reason(
                        &mut socket,
                        crate::protocol::CloseCode::AuthenticationFailed,
                        Some("Authentication failed"),
                    )
                    .await;
                    return;
                }
                mimi_recv_websocket(socket, state, room_id, addr).await
            },
        );
        Ok(upg)
    }

    async fn mimi_send_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<mimi::Mimi>,
        room_id: String,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.send_socket(socket, room_id).await {
            tracing::error!(?err, "mimi")
        }
    }

    async fn send(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<mimi::Mimi>, SharedState)>,
        req: axum::extract::Query<MimiStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling mimi-streaming send query");
        let auth_result = auth::check(&headers, req.token.as_deref());

        let room_id = match headers.get(ROOM_ID_HEADER) {
            Some(v) => v.to_str().ok().map(|v| v.to_string()),
            None => req.room_id.clone(),
        };
        let room_id = match room_id {
            None => Err(anyhow::format_err!("no room_id")),
            Some(room_id) => Ok(room_id),
        };

        let state = state.0 .0;
        let upg = ws.write_buffer_size(0).protocols(["permessage-deflate"]).on_upgrade(
            move |mut socket| async move {
                if let Err(err) = auth_result {
                    tracing::warn!(?err, "WebSocket auth failed, closing with 4001");
                    let _ = crate::utils::close_with_reason(
                        &mut socket,
                        crate::protocol::CloseCode::AuthenticationFailed,
                        Some("Authentication failed"),
                    )
                    .await;
                    return;
                }

                let room_id = match room_id {
                    Ok(id) => id,
                    Err(err) => {
                        tracing::error!(?err, "no room_id, closing socket");
                        // We could send a CloseCode::InvalidMessage here
                        let _ = crate::utils::close_with_reason(
                            &mut socket,
                            crate::protocol::CloseCode::InvalidMessage,
                            Some("Missing room_id"),
                        )
                        .await;
                        return;
                    }
                };

                mimi_send_websocket(socket, state, room_id, addr).await
            },
        );
        Ok(upg)
    }
    axum::Router::new()
        .route(send_path, axum::routing::get(send))
        .route(recv_path, axum::routing::get(recv))
        .with_state((s, ss.clone()))
}
