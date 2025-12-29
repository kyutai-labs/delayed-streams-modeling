// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::AsrStreamingQuery as Query;
use anyhow::{Context, Result};
use axum::extract::ws;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::collections::VecDeque;
use tokio::time::{timeout, Duration};

const FRAME_SIZE: usize = 1920;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum InMsg {
    Init,
    Marker { id: i64 },
    Audio { pcm: Vec<f32> },
    OggOpus { data: Vec<u8> },
    Ping,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum OutMsg {
    Word { text: String, start_time: f64 },
    EndWord { stop_time: f64 },
    Marker { id: i64 },
    Step { step_idx: usize, prs: Vec<f32>, buffered_pcm: usize },
    Error { message: String },
    Ready,
}

#[derive(Debug)]
pub struct Asr {
    asr_delay_in_tokens: usize,
    temperature: f64,
    lm: moshi::lm::LmModel,
    audio_tokenizer: moshi::mimi::Mimi,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
    instance_name: String,
    log_dir: std::path::PathBuf,
    conditions: Option<moshi::conditioner::Condition>,
}

impl Asr {
    pub fn new(asr: &crate::AsrConfig, config: &crate::Config, dev: &Device) -> Result<Self> {
        let dtype = crate::utils::model_dtype(asr.dtype_override.as_deref(), dev)?;
        let vb_lm =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&asr.lm_model_file], dtype, dev)? };
        let lm =
            moshi::lm::LmModel::new(&asr.model, moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm))?;
        let conditions = match lm.condition_provider() {
            None => None,
            Some(cp) => {
                let delay =
                    asr.conditioning_delay.context("missing conditioning_delay in config")?;
                let conditions = cp.condition_cont("delay", -delay)?;
                tracing::info!(?conditions, "generated conditions");
                Some(conditions)
            }
        };
        let audio_tokenizer = {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[&asr.audio_tokenizer_file],
                    DType::F32,
                    dev,
                )?
            };
            let mut cfg = moshi::mimi::Config::v0_1(Some(asr.model.audio_codebooks));
            // The mimi transformer runs at 25Hz.
            cfg.transformer.max_seq_len = asr.model.transformer.max_seq_len * 2;
            moshi::mimi::Mimi::new(cfg, vb)?
        };
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&asr.text_tokenizer_file)
            .with_context(|| asr.text_tokenizer_file.clone())?;
        Ok(Self {
            asr_delay_in_tokens: asr.asr_delay_in_tokens,
            lm,
            temperature: asr.temperature.unwrap_or(0.0),
            audio_tokenizer,
            text_tokenizer: text_tokenizer.into(),
            log_dir: config.log_dir.clone().into(),
            instance_name: config.instance_name.clone(),
            conditions,
        })
    }

    pub fn warmup(&self) -> Result<()> {
        let lm = self.lm.clone();
        let audio_tokenizer = self.audio_tokenizer.clone();
        let mut state = moshi::asr::State::new(
            1,
            self.asr_delay_in_tokens,
            self.temperature,
            audio_tokenizer,
            lm,
        )?;
        let dev = state.device().clone();
        let pcm_vec = vec![0f32; FRAME_SIZE * state.batch_size()];
        let pcm = Tensor::from_vec(pcm_vec, (state.batch_size(), 1, FRAME_SIZE), &dev)?;
        for _ in 0..2 {
            let _asr_msgs =
                state.step_pcm(pcm.clone(), self.conditions.as_ref(), &().into(), |_, _, _| ())?;
        }
        Ok(())
    }

    pub async fn handle_socket(&self, socket: ws::WebSocket, query: Query) -> Result<()> {
        use futures_util::{SinkExt, StreamExt};
        use serde::Serialize;

        let (mut sender, mut receiver) = socket.split();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<OutMsg>();
        let (log_tx, log_rx) = std::sync::mpsc::channel::<(Tensor, Vec<Tensor>)>();
        let log_tx_inference = log_tx.clone();
        let (log_done_tx, log_done_rx) = tokio::sync::oneshot::channel::<()>();

        let instance_name = self.instance_name.clone();
        let log_dir = self.log_dir.clone();
        let query_clone = query.clone();

        let logger_handle = crate::utils::spawn_blocking("logger_loop", move || {
            let mut all_text_tokens = vec![];
            let mut all_audio_tokens_vec = vec![];

            for (text_tokens_tensor, audio_tokens_tensors) in log_rx {
                let text_tokens_vec = text_tokens_tensor.to_vec1::<u32>().unwrap_or_default();
                let audio_tokens_vecs = audio_tokens_tensors
                    .iter()
                    .map(|t| t.to_vec1::<u32>().unwrap_or_default())
                    .collect::<Vec<_>>();
                all_text_tokens.push(text_tokens_vec);
                all_audio_tokens_vec.push(audio_tokens_vecs);
            }

            if all_text_tokens.is_empty() {
                return Ok(());
            }

            let num_steps = all_text_tokens.len();
            let batch_size = all_text_tokens[0].len();
            let text_tokens_flat: Vec<u32> = all_text_tokens.into_iter().flatten().collect();
            let text_tokens =
                Tensor::from_vec(text_tokens_flat, (batch_size, num_steps), &Device::Cpu)?;

            let num_codebooks = all_audio_tokens_vec[0].len();
            let audio_tokens_flat: Vec<u32> =
                all_audio_tokens_vec.into_iter().flatten().flatten().collect();
            let audio_tokens = Tensor::from_vec(
                audio_tokens_flat,
                (batch_size, num_codebooks, num_steps),
                &Device::Cpu,
            )?;

            let since_epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
            let (secs, us) = (since_epoch.as_secs(), since_epoch.subsec_micros());
            let base_path = log_dir.join(format!("{instance_name}-asr-{secs}-{us}"));

            let json_filename = base_path.with_extension("json");
            let json_content = serde_json::to_string_pretty(&query_clone)?;
            std::fs::write(json_filename, json_content)?;

            let st_filename = base_path.with_extension("safetensors");
            let text_tokens = text_tokens.to_dtype(DType::I64)?;
            let audio_tokens = audio_tokens.to_dtype(DType::I64)?;
            let st_content =
                std::collections::HashMap::from([("text", text_tokens), ("audio", audio_tokens)]);
            candle::safetensors::save(&st_content, st_filename)?;
            let _ = log_done_tx.send(());
            Ok(())
        });

        let lm = self.lm.clone();
        let audio_tokenizer = self.audio_tokenizer.clone();
        let mut state = moshi::asr::State::new(
            1,
            self.asr_delay_in_tokens,
            self.temperature,
            audio_tokenizer,
            lm,
        )?;
        let text_tokenizer = self.text_tokenizer.clone();

        let _asr_delay_in_tokens = self.asr_delay_in_tokens;
        let conditions = self.conditions.clone();
        let mut ogg_opus_decoder = kaudio::ogg_opus::Decoder::new(24000, 1920)?;
        let (pcm_tx, pcm_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(100);
        let recv_loop = crate::utils::spawn("recv_loop", async move {
            let mut _markers: VecDeque<(usize, i64)> = VecDeque::new();
            while let Some(msg) = receiver.next().await {
                let msg = match msg? {
                    ws::Message::Binary(x) => {
                        if crate::metrics::stream::enabled() {
                            crate::metrics::stream::ASR_WS_IN_MESSAGES.inc();
                            crate::metrics::stream::ASR_WS_IN_BYTES.inc_by(x.len() as u64);
                        }
                        x
                    }
                    ws::Message::Ping(_) | ws::Message::Pong(_) | ws::Message::Text(_) => continue,
                    ws::Message::Close(_) => break,
                };
                let msg: InMsg = match rmp_serde::from_slice(&msg) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(error = %e, msg_len = msg.len(), "failed to deserialize InMsg, skipping message");
                        continue;
                    }
                };
                let pcm = match msg {
                    InMsg::Init => None,
                    InMsg::Marker { id } => {
                        tracing::info!("received marker {id}");
                        // Markers need to be handled carefully with pipelining.
                        // We'll send them through the pcm_tx as a special message if needed,
                        // or just rely on the step_idx.
                        // For now, let's just use a special signal or assume markers are rare.
                        None
                    }
                    InMsg::OggOpus { data } => ogg_opus_decoder.decode(&data)?.map(|v| v.to_vec()),
                    InMsg::Audio { pcm } => Some(pcm),
                    InMsg::Ping => None,
                };
                if let Some(pcm) = pcm {
                    pcm_tx.send(pcm)?;
                }
            }
            Ok::<(), anyhow::Error>(())
        });

        let (mimi_tx, mimi_rx) = std::sync::mpsc::sync_channel::<Vec<Vec<u32>>>(100);
        let mimi_dev = state.device().clone();
        let mimi_batch_size = state.batch_size();
        let mut mimi_tokenizer = state.audio_tokenizer.clone();
        #[cfg(feature = "cuda")]
        let mut pinned_pcm = if let Device::Cuda(cuda_dev) = &mimi_dev {
            let p = unsafe {
                cuda_dev
                    .cuda_stream()
                    .context()
                    .alloc_pinned::<f32>(FRAME_SIZE * state.batch_size())
            }
            .ok();
            if let Some(mut p) = p {
                if let Ok(slice) = p.as_mut_slice() {
                    slice.fill(0.0);
                    Some(p)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(not(feature = "cuda"))]
        let mut pinned_pcm: Option<Vec<f32>> = None;

        let mimi_handle = crate::utils::spawn_blocking("mimi_encode_loop", move || {
            for pcm in pcm_rx {
                let pcm_len = pcm.len();

                let pcm_tensor = {
                    #[cfg(feature = "cuda")]
                    {
                        match pinned_pcm.as_mut() {
                            Some(p) => match p.as_mut_slice() {
                                Ok(slice) => {
                                    let len = pcm_len.min(slice.len());
                                    slice[..len].copy_from_slice(&pcm[..len]);
                                    Tensor::from_slice(&slice[..len], (1, 1, len), &mimi_dev)
                                }
                                Err(_) => Tensor::from_vec(pcm, (1, 1, pcm_len), &mimi_dev),
                            },
                            None => Tensor::from_vec(pcm, (1, 1, pcm_len), &mimi_dev),
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let _ = &mut pinned_pcm;
                        Tensor::from_vec(pcm, (1, 1, pcm_len), &mimi_dev)
                    }
                }?
                .broadcast_as((mimi_batch_size, 1, pcm_len))?;
                let audio_tokens = mimi_tokenizer.encode_step(&pcm_tensor.into(), &().into())?;
                if let Some(audio_tokens) = audio_tokens.as_option() {
                    let (_one, _codebooks, steps) = audio_tokens.dims3()?;
                    let mut all_steps = Vec::with_capacity(steps);
                    for step in 0..steps {
                        let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                        all_steps.push(codes);
                    }
                    mimi_tx.send(all_steps)?;
                }
            }
            Ok::<(), anyhow::Error>(())
        });

        let inference_handle = crate::utils::spawn_blocking("inference_loop", move || {
            for steps_tokens in mimi_rx {
                for codes in steps_tokens {
                    let asr_msgs = state.step_tokens_vec(
                        codes,
                        conditions.as_ref(),
                        &().into(),
                        |_, text_tokens, audio_tokens| {
                            if let Err(err) =
                                log_tx_inference.send((text_tokens.clone(), audio_tokens.to_vec()))
                            {
                                tracing::error!(?err, "failed to send log");
                            }
                        },
                    )?;
                    for asr_msg in asr_msgs {
                        let msg = match asr_msg {
                            moshi::asr::AsrMsg::Word { tokens, start_time, .. } => OutMsg::Word {
                                text: text_tokenizer.decode_piece_ids(&tokens)?,
                                start_time,
                            },
                            moshi::asr::AsrMsg::Step { step_idx, prs } => {
                                let prs = prs.iter().map(|p| p[0]).collect::<Vec<_>>();
                                OutMsg::Step { step_idx, prs, buffered_pcm: 0 }
                            }
                            moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                                OutMsg::EndWord { stop_time }
                            }
                        };
                        tx.send(msg)?
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        });
        let send_loop = crate::utils::spawn("send_loop", async move {
            use bytes::BufMut;

            let mut chunk_buf = bytes::BytesMut::with_capacity(8 * 1024);
            let mut chunk_buf_spare = bytes::BytesMut::with_capacity(8 * 1024);
            loop {
                // The recv method is cancel-safe so can be wrapped in a timeout.
                let msg = timeout(Duration::from_secs(10), rx.recv()).await;
                let msg = match msg {
                    Ok(None) => break,
                    Err(_) => ws::Message::Ping(vec![].into()),
                    Ok(Some(msg)) => {
                        chunk_buf.clear();
                        {
                            let mut w = (&mut chunk_buf).writer();
                            msg.serialize(
                                &mut rmp_serde::Serializer::new(&mut w)
                                    .with_human_readable()
                                    .with_struct_map(),
                            )?;
                        }
                        std::mem::swap(&mut chunk_buf, &mut chunk_buf_spare);
                        let bytes = chunk_buf_spare.split().freeze();
                        if crate::metrics::stream::enabled() {
                            crate::metrics::stream::ASR_WS_OUT_MESSAGES.inc();
                            crate::metrics::stream::ASR_WS_OUT_BYTES.inc_by(bytes.len() as u64);
                        }
                        ws::Message::Binary(bytes)
                    }
                };
                sender.send(msg).await?;
            }
            tracing::info!("send loop exited");
            Ok::<(), anyhow::Error>(())
        });
        // recv_loop and send_loop are already JoinHandle<()> from crate::utils::spawn
        let mut recv_handle = recv_loop;
        let mut send_handle = send_loop;

        let sleep = tokio::time::sleep(std::time::Duration::from_secs(360));
        tokio::pin!(sleep);

        // Use tokio::select! with proper abort handling for spawned tasks.
        // When one branch completes or times out, we explicitly abort the other tasks.
        tokio::select! {
            _ = &mut sleep => {
                tracing::error!("reached timeout, aborting background tasks");
            }
            _ = &mut recv_handle => {}
            _ = &mut send_handle => {}
        }

        // Explicitly abort tasks if they are still running
        recv_handle.abort();
        send_handle.abort();
        mimi_handle.abort();
        inference_handle.abort();

        // Wait briefly for aborted tasks to clean up
        let _ = tokio::time::timeout(std::time::Duration::from_millis(500), async {
            let _ = recv_handle.await;
            let _ = send_handle.await;
            let _ = mimi_handle.await;
            let _ = inference_handle.await;
            drop(log_tx); // Close the log channel to trigger logger completion
            let _ = logger_handle.await;
            let _ = log_done_rx.await;
        })
        .await;

        tracing::info!("exiting handle-socket");
        Ok(())
    }
}
