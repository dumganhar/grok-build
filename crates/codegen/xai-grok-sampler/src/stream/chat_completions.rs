//! Layer-2 stream transform for the Chat Completions API.
//!
//! Consumes a raw `ChatCompletionChunk` stream and produces
//! [`SamplingEvent`]s. Pure: no I/O, no shell coupling.

use std::collections::BTreeMap;
use std::ops::Range;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use futures_util::stream::{BoxStream, Stream};

use xai_grok_sampling_types::{
    AssistantItem, ChatCompletionChunk, ConversationItem, ConversationResponse,
    ResponseModelMetadata, SamplingError, StopReason, TokenUsage, ToolCall,
};

use crate::events::{SamplingChannel, SamplingErrorInfo, SamplingEvent};
use crate::metrics::InferenceLatencyStats;
use crate::types::RequestId;

const TAGGED_TOOL_CALL_OPEN: &str = "<tool_call>";
const TAGGED_TOOL_CALL_CLOSE: &str = "</tool_call>";

#[derive(Debug)]
struct TaggedToolCall {
    name: String,
    arguments: String,
    range: Range<usize>,
}

fn parse_tagged_tool_call(body: &str) -> Option<(String, String)> {
    const FUNCTION_OPEN: &str = "<function=";
    const FUNCTION_CLOSE: &str = "</function>";
    const PARAMETER_OPEN: &str = "<parameter=";
    const PARAMETER_CLOSE: &str = "</parameter>";

    let name_start = body.find(FUNCTION_OPEN)? + FUNCTION_OPEN.len();
    let name_end = name_start + body[name_start..].find('>')?;
    let name = body[name_start..name_end].trim();
    if name.is_empty() {
        return None;
    }

    let parameters_start = name_end + 1;
    let parameters_end = parameters_start + body[parameters_start..].find(FUNCTION_CLOSE)?;
    let parameters = &body[parameters_start..parameters_end];
    let mut arguments = serde_json::Map::new();
    let mut cursor = 0;

    while let Some(open_offset) = parameters[cursor..].find(PARAMETER_OPEN) {
        let key_start = cursor + open_offset + PARAMETER_OPEN.len();
        let key_end = key_start + parameters[key_start..].find('>')?;
        let value_start = key_end + 1;
        let value_end = value_start + parameters[value_start..].find(PARAMETER_CLOSE)?;
        let key = parameters[key_start..key_end].trim();
        if key.is_empty() {
            return None;
        }
        let raw_value = parameters[value_start..value_end].trim();
        let value = serde_json::from_str(raw_value)
            .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_owned()));
        arguments.insert(key.to_owned(), value);
        cursor = value_end + PARAMETER_CLOSE.len();
    }

    Some((
        name.to_owned(),
        serde_json::Value::Object(arguments).to_string(),
    ))
}

fn parse_tagged_tool_calls(content: &str) -> Vec<TaggedToolCall> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(open_offset) = content[cursor..].find(TAGGED_TOOL_CALL_OPEN) {
        let start = cursor + open_offset;
        let body_start = start + TAGGED_TOOL_CALL_OPEN.len();
        let Some(close_offset) = content[body_start..].find(TAGGED_TOOL_CALL_CLOSE) else {
            break;
        };
        let body_end = body_start + close_offset;
        let end = body_end + TAGGED_TOOL_CALL_CLOSE.len();
        if let Some((name, arguments)) = parse_tagged_tool_call(&content[body_start..body_end]) {
            calls.push(TaggedToolCall {
                name,
                arguments,
                range: start..end,
            });
        }
        cursor = end;
    }
    calls
}

fn repaired_native_arguments(native: &str, tagged: &str) -> Option<String> {
    let tagged = serde_json::from_str::<serde_json::Value>(tagged).ok()?;
    let Ok(mut native) = serde_json::from_str::<serde_json::Value>(native) else {
        return Some(tagged.to_string());
    };
    repair_argument_value(&mut native, &tagged).then(|| native.to_string())
}

fn argument_value_is_placeholder(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::String(value) => value.is_empty(),
        serde_json::Value::Array(value) => value.is_empty(),
        serde_json::Value::Object(value) => value.is_empty(),
        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => false,
    }
}

fn repair_argument_value(native: &mut serde_json::Value, tagged: &serde_json::Value) -> bool {
    if native == tagged {
        return false;
    }
    if argument_value_is_placeholder(tagged) {
        return false;
    }
    if argument_value_is_placeholder(native) {
        *native = tagged.clone();
        return true;
    }

    match (native, tagged) {
        (serde_json::Value::Object(native), serde_json::Value::Object(tagged)) => {
            let mut changed = false;
            for (key, native_value) in native {
                if let Some(tagged_value) = tagged.get(key) {
                    changed |= repair_argument_value(native_value, tagged_value);
                }
            }
            changed
        }
        (serde_json::Value::Array(native), serde_json::Value::Array(tagged))
            if native.len() == tagged.len() =>
        {
            native
                .iter_mut()
                .zip(tagged)
                .fold(false, |changed, (native, tagged)| {
                    repair_argument_value(native, tagged) || changed
                })
        }
        (native, tagged) if std::mem::discriminant(native) != std::mem::discriminant(tagged) => {
            *native = tagged.clone();
            true
        }
        _ => false,
    }
}

/// Return the text prefix that is safe to stream before a possible tagged tool
/// call. A partial opening marker is held so chunk boundaries cannot leak it.
fn streaming_safe_prefix_len(text: &str) -> (usize, bool) {
    if let Some(index) = text.find(TAGGED_TOOL_CALL_OPEN) {
        return (index, true);
    }

    let text = text.as_bytes();
    let marker = TAGGED_TOOL_CALL_OPEN.as_bytes();
    let held_suffix_len = (1..marker.len())
        .rev()
        .find(|&len| text.ends_with(&marker[..len]))
        .unwrap_or(0);
    (text.len() - held_suffix_len, false)
}

/// Some OpenAI-compatible model servers duplicate native tool calls as tagged
/// assistant text. The native call owns execution; recover its arguments from
/// the tagged twin only when malformed, and hide every matched text duplicate.
fn recover_tagged_tool_calls(content: &str, tool_calls: &mut [ToolCall]) -> String {
    if !content.contains(TAGGED_TOOL_CALL_OPEN) {
        return content.to_owned();
    }

    let tagged_calls = parse_tagged_tool_calls(content);
    let mut used = vec![false; tagged_calls.len()];
    for call in tool_calls {
        let Some((index, tagged)) = tagged_calls
            .iter()
            .enumerate()
            .find(|(index, tagged)| !used[*index] && tagged.name == call.name)
        else {
            continue;
        };
        used[index] = true;
        if let Some(arguments) = repaired_native_arguments(&call.arguments, &tagged.arguments) {
            call.arguments = std::sync::Arc::<str>::from(arguments);
            tracing::warn!(
                tool_call_id = %call.id,
                tool_name = %call.name,
                "recovered malformed or incomplete Chat Completions tool arguments from tagged assistant text"
            );
        }
    }

    if !used.iter().any(|value| *value) {
        return content.to_owned();
    }
    let mut visible = String::with_capacity(content.len());
    let mut cursor = 0;
    for (tagged, used) in tagged_calls.iter().zip(used) {
        if used {
            visible.push_str(&content[cursor..tagged.range.start]);
            cursor = tagged.range.end;
        }
    }
    visible.push_str(&content[cursor..]);
    visible
}

/// Transform a raw Chat Completions chunk stream into a stream of
/// [`SamplingEvent`]s.
///
/// The output stream emits exactly one terminal event per request:
/// [`SamplingEvent::Completed`] on normal stream end, or
/// [`SamplingEvent::Failed`] on error / idle timeout. Callers must not
/// consume past the terminal event (the implementation `return`s after
/// yielding it).
///
/// `idle_timeout` covers two cases:
/// 1. The transport stops yielding chunks at all (`tokio::time::timeout`).
/// 2. The transport keeps yielding empty / keepalive chunks but no
///    meaningful content (separate `last_content_chunk_at` timer).
///
/// Both produce `SamplingEvent::Failed { kind: IdleTimeout }`.
pub fn stream_chat_completions<'a>(
    raw_stream: BoxStream<'a, Result<ChatCompletionChunk, SamplingError>>,
    model_metadata: Option<ResponseModelMetadata>,
    request_id: RequestId,
    idle_timeout: Duration,
) -> impl Stream<Item = SamplingEvent> + Send + 'a {
    async_stream::stream! {
        let stream_start = Instant::now();
        let mut chunk_timestamps: Vec<Instant> = Vec::new();

        // Emit StreamStarted before reading any chunks so subscribers
        // can record TTFB / TTLB baselines.
        yield SamplingEvent::StreamStarted {
            request_id: request_id.clone(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        };

        if let Some(metadata) = model_metadata {
            yield SamplingEvent::ModelMetadata {
                request_id: request_id.clone(),
                metadata,
            };
        }

        // Per-response accumulators
        let mut first_chunk_seen = false;
        let mut first_choice_seen = false;
        let mut first_token_emitted = false;
        let mut model: String = String::new();
        let mut model_fingerprint: Option<String> = None;
        let mut usage: Option<TokenUsage> = None;
        let mut cost_usd_ticks: Option<i64> = None;
        let mut finish_reason: Option<StopReason> = None;

        let mut content_acc = String::new();
        let mut streamed_content_len = 0;
        let mut defer_tagged_text = false;
        let mut reasoning_acc = String::new();
        // Tool call deltas keyed by positional index. Each entry is
        // (id, name, arguments_buffer); the first chunk for an index
        // carries id+name and starts the arguments buffer, subsequent
        // chunks append to arguments only.
        let mut tool_call_acc: BTreeMap<u32, (String, String, String)> = BTreeMap::new();

        // Index counter spanning text + reasoning chunks (matches the
        // shell's chunk_index used for notification correlation).
        let mut chunk_index: u64 = 0;
        // Separate counter for AgentMessageChunk (text-only) emissions;
        // mirrored onto ConversationResponse.message_chunks_emitted so
        // downstream can detect lost-streaming-events scenarios.
        let mut message_chunk_count: u64 = 0;

        // Content-aware idle timer: the outer
        // `tokio::time::timeout(idle_timeout, stream.next())` already
        // catches "transport stops yielding chunks". This second timer
        // catches the more subtle case where the model keeps emitting
        // keepalive / empty-delta SSE events that satisfy the outer
        // timer but make no real progress -- some inference engines
        // do exactly that.
        let mut last_content_chunk_at = Instant::now();

        let mut stream = raw_stream;
        loop {
            let next = match tokio::time::timeout(idle_timeout, stream.next()).await {
                Ok(Some(next)) => next,
                Ok(None) => break, // stream ended normally
                Err(_elapsed) => {
                    let err = SamplingError::IdleTimeout {
                        elapsed_secs: idle_timeout.as_secs(),
                    };
                    yield SamplingEvent::Failed {
                        request_id: request_id.clone(),
                        error: SamplingErrorInfo::from(&err),
                    };
                    return;
                }
            };
            let chunk = match next {
                Ok(chunk) => chunk,
                Err(err) => {
                    yield SamplingEvent::Failed {
                        request_id: request_id.clone(),
                        error: SamplingErrorInfo::from(&err),
                    };
                    return;
                }
            };

            if !first_chunk_seen {
                model = chunk.model.clone();
                model_fingerprint = chunk
                    .system_fingerprint
                    .clone()
                    .filter(|s| !s.is_empty());
                first_chunk_seen = true;
            }

            if let Some(u) = chunk.usage.clone() {
                // Wire cost is cumulative for the response, so last-write-wins.
                // Never clobber a known cost with missing/unreported.
                let chunk_cost = xai_grok_sampling_types::reported_cost_ticks(u.cost_in_usd_ticks);
                cost_usd_ticks = match (cost_usd_ticks, chunk_cost) {
                    (_, Some(n)) => Some(n),
                    (prev, None) => prev,
                };
                usage = Some(u.into());
            }

            // Track whether this chunk carried meaningful content.
            // Set inside the choices loop and checked at the end.
            let mut chunk_has_content = false;

            for choice in chunk.choices.into_iter() {
                first_choice_seen = true;
                if let Some(fr) = choice.finish_reason {
                    finish_reason = Some(fr.into());
                    chunk_has_content = true;
                }

                let delta = choice.delta;

                if let Some(text) = delta.content
                    && !text.is_empty()
                {
                    if !first_token_emitted {
                        first_token_emitted = true;
                        yield SamplingEvent::FirstToken {
                            request_id: request_id.clone(),
                        };
                    }
                    chunk_has_content = true;
                    chunk_timestamps.push(Instant::now());
                    content_acc.push_str(&text);
                    if !defer_tagged_text {
                        let pending = &content_acc[streamed_content_len..];
                        let (safe_len, found_tag) = streaming_safe_prefix_len(pending);
                        let safe_text = pending[..safe_len].to_owned();
                        streamed_content_len += safe_len;
                        defer_tagged_text = found_tag;
                        if !safe_text.is_empty() {
                            chunk_index += 1;
                            message_chunk_count += 1;
                            yield SamplingEvent::ChannelToken {
                                request_id: request_id.clone(),
                                channel: SamplingChannel::Text,
                                text: safe_text,
                                chunk_index,
                            };
                        }
                    }
                }

                if let Some(thought) = delta.reasoning_content
                    && !thought.is_empty()
                {
                    if !first_token_emitted {
                        first_token_emitted = true;
                        yield SamplingEvent::FirstToken {
                            request_id: request_id.clone(),
                        };
                    }
                    chunk_has_content = true;
                    chunk_index += 1;
                    reasoning_acc.push_str(&thought);
                    yield SamplingEvent::ChannelToken {
                        request_id: request_id.clone(),
                        channel: SamplingChannel::Reasoning,
                        text: thought,
                        chunk_index,
                    };
                }

                for tc_delta in delta.tool_calls.into_iter() {
                    chunk_has_content = true;

                    let entry = tool_call_acc
                        .entry(tc_delta.index)
                        .or_insert_with(|| (String::new(), String::new(), String::new()));

                    let mut id_for_event: Option<String> = None;
                    let mut name_for_event: Option<String> = None;
                    let mut args_for_event: Option<String> = None;

                    if let Some(id) = tc_delta.id {
                        entry.0 = id.clone();
                        id_for_event = Some(id);
                    }
                    if let Some(func) = tc_delta.function {
                        if let Some(name) = func.name {
                            entry.1 = name.clone();
                            name_for_event = Some(name);
                        }
                        if let Some(args) = func.arguments {
                            entry.2.push_str(&args);
                            args_for_event = Some(args);
                        }
                    }

                    yield SamplingEvent::ToolCallDelta {
                        request_id: request_id.clone(),
                        tool_index: tc_delta.index,
                        id: id_for_event,
                        name: name_for_event,
                        arguments_delta: args_for_event,
                    };
                }
            }

            if chunk_has_content {
                last_content_chunk_at = Instant::now();
            } else if last_content_chunk_at.elapsed() > idle_timeout {
                let err = SamplingError::IdleTimeout {
                    elapsed_secs: idle_timeout.as_secs(),
                };
                yield SamplingEvent::Failed {
                    request_id: request_id.clone(),
                    error: SamplingErrorInfo::from(&err),
                };
                return;
            }
        }

        // ── Build the final response ─────────────────────────────────
        let mut tool_calls: Vec<ToolCall> = tool_call_acc
            .into_values()
            .map(|(id, name, arguments)| ToolCall {
                id: std::sync::Arc::<str>::from(id),
                name,
                arguments: std::sync::Arc::<str>::from(arguments),
            })
            .collect();
        let visible_content = recover_tagged_tool_calls(&content_acc, &mut tool_calls);
        let streamed_prefix = &content_acc[..streamed_content_len];
        let deferred_visible_text = match visible_content.strip_prefix(streamed_prefix) {
            Some(text) => text.to_owned(),
            None => {
                tracing::warn!(
                    "reconciled Chat Completions text did not retain its streamed prefix"
                );
                visible_content.clone()
            }
        };
        if !deferred_visible_text.is_empty() {
            chunk_index += 1;
            message_chunk_count += 1;
            yield SamplingEvent::ChannelToken {
                request_id: request_id.clone(),
                channel: SamplingChannel::Text,
                text: deferred_visible_text,
                chunk_index,
            };
        }

        // Honor tool calls by overriding the stop reason if the model
        // forgot to set it (mirrors the shell's behavior).
        if !tool_calls.is_empty() {
            finish_reason = Some(StopReason::ToolCalls);
        }

        // Build the trailing Assistant + any reasoning sibling.
        let mut items: Vec<ConversationItem> = Vec::new();
        if first_choice_seen {
            if !reasoning_acc.is_empty() {
                items.push(ConversationItem::Reasoning(
                    xai_grok_sampling_types::synthesized_reasoning_item(reasoning_acc),
                ));
            }
            items.push(ConversationItem::Assistant(AssistantItem {
                content: std::sync::Arc::<str>::from(visible_content),
                tool_calls,
                model_id: Some(model),
                model_fingerprint,
                // Chat Completions does not echo the applied reasoning effort.
                reasoning_effort: None,
            }));
        } else {
            items.push(ConversationItem::assistant(""));
        }

        let stream_end = Instant::now();
        let metrics =
            InferenceLatencyStats::from_timestamps(stream_start, &chunk_timestamps, stream_end);

        let response = ConversationResponse {
            items,
            stop_reason: finish_reason,
            usage,
            cost_usd_ticks,
            message_chunks_emitted: message_chunk_count,
            doom_loop_signals: Vec::new(),
            stop_message: None,
            message_id: None,
            raw_stop_reason: None,
            stop_sequence: None,
        };

        yield SamplingEvent::Completed {
            request_id: request_id.clone(),
            response: Box::new(response),
            metrics,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use std::pin::pin;
    use xai_grok_sampling_types::{
        ChatChunkChoice, ChatChunkDelta, FinishReason, Role, ToolCallDelta as ChunkToolCallDelta,
        ToolCallFunctionDelta, Usage, rs,
    };

    fn rid() -> RequestId {
        RequestId::from("test-req")
    }

    fn make_chunk(deltas: Vec<ChatChunkDelta>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chunk-1".into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: "test-model".into(),
            choices: deltas
                .into_iter()
                .enumerate()
                .map(|(i, delta)| ChatChunkChoice {
                    index: i as u32,
                    delta,
                    finish_reason: None,
                })
                .collect(),
            usage: None,
            system_fingerprint: None,
        }
    }

    fn text_chunk(text: &str) -> ChatCompletionChunk {
        make_chunk(vec![ChatChunkDelta {
            role: Some(Role::Assistant),
            content: Some(text.to_string()),
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
        }])
    }

    fn final_chunk(reason: FinishReason) -> ChatCompletionChunk {
        let mut chunk = make_chunk(vec![ChatChunkDelta::default()]);
        chunk.choices[0].finish_reason = Some(reason);
        chunk
    }

    async fn collect(s: impl Stream<Item = SamplingEvent>) -> Vec<SamplingEvent> {
        let mut out = Vec::new();
        let mut s = pin!(s);
        while let Some(ev) = s.next().await {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn empty_stream_yields_started_then_completed() {
        let raw = stream::iter(Vec::<Result<ChatCompletionChunk, SamplingError>>::new()).boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], SamplingEvent::StreamStarted { .. }));
        match &events[1] {
            SamplingEvent::Completed { response, .. } => {
                assert!(response.is_empty());
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn text_only_stream_emits_first_token_then_channel_tokens_then_completed() {
        let chunks: Vec<Result<ChatCompletionChunk, SamplingError>> = vec![
            Ok(text_chunk("Hello, ")),
            Ok(text_chunk("world!")),
            Ok(final_chunk(FinishReason::Stop)),
        ];
        let raw = stream::iter(chunks).boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        // Expected sequence: StreamStarted, FirstToken, ChannelToken(Text)
        // x 2, Completed.
        assert!(matches!(events[0], SamplingEvent::StreamStarted { .. }));
        assert!(matches!(events[1], SamplingEvent::FirstToken { .. }));

        let text_tokens: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                SamplingEvent::ChannelToken {
                    channel: SamplingChannel::Text,
                    text,
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text_tokens, vec!["Hello, ", "world!"]);

        match events.last().unwrap() {
            SamplingEvent::Completed { response, .. } => {
                let a = response.assistant().expect("assistant item present");
                assert_eq!(a.content.as_ref(), "Hello, world!");
                assert_eq!(response.stop_reason, Some(StopReason::Stop));
                assert_eq!(response.message_chunks_emitted, 2);
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reasoning_chunk_emits_reasoning_channel_and_first_token_once() {
        let mut reasoning_chunk = make_chunk(vec![ChatChunkDelta {
            role: Some(Role::Assistant),
            content: None,
            reasoning_content: Some("thinking...".into()),
            tool_calls: vec![],
            tool_call_id: None,
        }]);
        reasoning_chunk.choices[0].finish_reason = None;

        let chunks: Vec<Result<ChatCompletionChunk, SamplingError>> = vec![
            Ok(reasoning_chunk),
            Ok(text_chunk("done")),
            Ok(final_chunk(FinishReason::Stop)),
        ];
        let raw = stream::iter(chunks).boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        // FirstToken should appear exactly once.
        let first_token_count = events
            .iter()
            .filter(|e| matches!(e, SamplingEvent::FirstToken { .. }))
            .count();
        assert_eq!(first_token_count, 1);

        let mut saw_reasoning = false;
        let mut saw_text = false;
        for e in &events {
            if let SamplingEvent::ChannelToken { channel, text, .. } = e {
                match channel {
                    SamplingChannel::Reasoning => {
                        assert_eq!(text, "thinking...");
                        saw_reasoning = true;
                    }
                    SamplingChannel::Text => {
                        assert_eq!(text, "done");
                        saw_text = true;
                    }
                }
            }
        }
        assert!(saw_reasoning && saw_text);

        match events.last().unwrap() {
            SamplingEvent::Completed { response, .. } => {
                let r = response
                    .reasoning_items()
                    .next()
                    .expect("reasoning sibling preserved");
                let rs::SummaryPart::SummaryText(t) = &r.summary[0];
                assert_eq!(t.text, "thinking...");
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_call_stream_emits_deltas_and_assembles_final_call() {
        // First chunk has id + name + part of arguments.
        let chunk1 = make_chunk(vec![ChatChunkDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: vec![ChunkToolCallDelta {
                index: 0,
                id: Some("call_abc".into()),
                kind: Some("function".into()),
                function: Some(ToolCallFunctionDelta {
                    name: Some("do_thing".into()),
                    arguments: Some("{\"x\":".into()),
                }),
            }],
            tool_call_id: None,
        }]);
        // Second chunk has only argument fragment.
        let chunk2 = make_chunk(vec![ChatChunkDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: vec![ChunkToolCallDelta {
                index: 0,
                id: None,
                kind: None,
                function: Some(ToolCallFunctionDelta {
                    name: None,
                    arguments: Some("1}".into()),
                }),
            }],
            tool_call_id: None,
        }]);

        let raw = stream::iter::<Vec<Result<ChatCompletionChunk, SamplingError>>>(vec![
            Ok(chunk1),
            Ok(chunk2),
        ])
        .boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        let deltas: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                SamplingEvent::ToolCallDelta {
                    tool_index,
                    id,
                    name,
                    arguments_delta,
                    ..
                } => Some((
                    *tool_index,
                    id.clone(),
                    name.clone(),
                    arguments_delta.clone(),
                )),
                _ => None,
            })
            .collect();

        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].0, 0);
        assert_eq!(deltas[0].1.as_deref(), Some("call_abc"));
        assert_eq!(deltas[0].2.as_deref(), Some("do_thing"));
        assert_eq!(deltas[0].3.as_deref(), Some("{\"x\":"));
        assert_eq!(deltas[1].1, None);
        assert_eq!(deltas[1].2, None);
        assert_eq!(deltas[1].3.as_deref(), Some("1}"));

        match events.last().unwrap() {
            SamplingEvent::Completed { response, .. } => {
                let calls = response.tool_calls();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id.as_ref(), "call_abc");
                assert_eq!(calls[0].name, "do_thing");
                assert_eq!(calls[0].arguments.as_ref(), "{\"x\":1}");
                // Tool calls force ToolCalls stop reason.
                assert_eq!(response.stop_reason, Some(StopReason::ToolCalls));
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_tool_arguments_recover_from_tagged_assistant_text() {
        let tagged = concat!(
            "Waiting...<tool_call>\n",
            "<function=get_command_or_subagent_output>\n",
            "<parameter=task_ids>[\"task-1\"]</parameter>\n",
            "<parameter=timeout_ms>60000</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let tool_chunk = make_chunk(vec![ChatChunkDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: vec![ChunkToolCallDelta {
                index: 0,
                id: Some("call_wait".into()),
                kind: Some("function".into()),
                function: Some(ToolCallFunctionDelta {
                    name: Some("get_command_or_subagent_output".into()),
                    arguments: Some("{\"task_ids\":[\"task-1\"],\"timeout_ms\":".into()),
                }),
            }],
            tool_call_id: None,
        }]);
        let split = tagged.find(TAGGED_TOOL_CALL_OPEN).unwrap() + "<tool_".len();
        let raw = stream::iter(vec![
            Ok(text_chunk(&tagged[..split])),
            Ok(text_chunk(&tagged[split..])),
            Ok(tool_chunk),
        ])
        .boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        let streamed_text: String = events
            .iter()
            .filter_map(|event| match event {
                SamplingEvent::ChannelToken {
                    channel: SamplingChannel::Text,
                    text,
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(streamed_text, "Waiting...");

        match events.last().unwrap() {
            SamplingEvent::Completed { response, .. } => {
                let assistant = response.assistant().expect("assistant item present");
                assert_eq!(assistant.content.as_ref(), "Waiting...");
                let arguments: serde_json::Value =
                    serde_json::from_str(&assistant.tool_calls[0].arguments).unwrap();
                assert_eq!(arguments["task_ids"], serde_json::json!(["task-1"]));
                assert_eq!(arguments["timeout_ms"], 60_000);
                assert_eq!(response.message_chunks_emitted, 1);
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn placeholder_tool_argument_recovers_from_tagged_assistant_text() {
        let tagged = concat!(
            "<tool_call>\n",
            "<function=get_command_or_subagent_output>\n",
            "<parameter=task_ids>[\"task-1\"]</parameter>\n",
            "<parameter=timeout_ms>60000</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let mut tool_calls = vec![ToolCall {
            id: std::sync::Arc::<str>::from("call_wait"),
            name: "get_command_or_subagent_output".into(),
            arguments: std::sync::Arc::<str>::from(
                r#"{"task_ids":["native-task"],"timeout_ms":{}}"#,
            ),
        }];

        let visible = recover_tagged_tool_calls(tagged, &mut tool_calls);

        assert_eq!(visible, "");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&tool_calls[0].arguments).unwrap(),
            serde_json::json!({"task_ids": ["native-task"], "timeout_ms": 60_000})
        );
    }

    #[test]
    fn valid_tool_arguments_remain_authoritative_while_duplicate_text_is_hidden() {
        let tagged = concat!(
            "<tool_call>\n",
            "<function=get_command_or_subagent_output>\n",
            "<parameter=task_ids>[\"tagged-task\"]</parameter>\n",
            "<parameter=timeout_ms>60000</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let native_arguments = r#"{"task_ids":["native-task"],"timeout_ms":120000}"#;
        let mut tool_calls = vec![ToolCall {
            id: std::sync::Arc::<str>::from("call_wait"),
            name: "get_command_or_subagent_output".into(),
            arguments: std::sync::Arc::<str>::from(native_arguments),
        }];

        let visible = recover_tagged_tool_calls(tagged, &mut tool_calls);

        assert_eq!(visible, "");
        assert_eq!(tool_calls[0].arguments.as_ref(), native_arguments);
    }

    #[tokio::test]
    async fn unmatched_tagged_tool_text_is_preserved() {
        let tagged = concat!(
            "<tool_call>\n",
            "<function=example>\n",
            "<parameter=value>1</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let raw = stream::iter(vec![
            Ok(text_chunk("<tool_")),
            Ok(text_chunk(&tagged["<tool_".len()..])),
        ])
        .boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        let streamed_text: String = events
            .iter()
            .filter_map(|event| match event {
                SamplingEvent::ChannelToken {
                    channel: SamplingChannel::Text,
                    text,
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(streamed_text, tagged);
        match events.last().unwrap() {
            SamplingEvent::Completed { response, .. } => {
                assert_eq!(response.assistant().unwrap().content.as_ref(), tagged);
                assert_eq!(response.message_chunks_emitted, 1);
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mid_stream_error_yields_failed_no_completed() {
        let chunks: Vec<Result<ChatCompletionChunk, SamplingError>> = vec![
            Ok(text_chunk("hi")),
            Err(SamplingError::EventStreamError("conn reset".into())),
        ];
        let raw = stream::iter(chunks).boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        assert!(
            events
                .iter()
                .any(|e| matches!(e, SamplingEvent::Failed { .. }))
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, SamplingEvent::Completed { .. }))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn idle_timeout_when_stream_stalls() {
        // A stream that yields one chunk then hangs forever.
        let raw = stream::iter(vec![Ok(text_chunk("hello"))])
            .chain(stream::pending())
            .boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_millis(100),
        ))
        .await;

        // Stream should emit StreamStarted, FirstToken, ChannelToken
        // then Failed(IdleTimeout) when the stall hits the deadline.
        match events.last().unwrap() {
            SamplingEvent::Failed { error, .. } => {
                assert_eq!(error.kind, crate::events::SamplingErrorKind::IdleTimeout);
            }
            other => panic!("expected Failed(IdleTimeout), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn model_metadata_yielded_after_stream_started() {
        let raw = stream::iter(Vec::<Result<ChatCompletionChunk, SamplingError>>::new()).boxed();
        let metadata = ResponseModelMetadata {
            context_window: Some(8192),
            max_completion_tokens: Some(4096),
            models_etag: None,
        };
        let events = collect(stream_chat_completions(
            raw,
            Some(metadata.clone()),
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        assert!(matches!(events[0], SamplingEvent::StreamStarted { .. }));
        match &events[1] {
            SamplingEvent::ModelMetadata { metadata: m, .. } => {
                assert_eq!(m.context_window, Some(8192));
                assert_eq!(m.max_completion_tokens, Some(4096));
            }
            other => panic!("expected ModelMetadata second, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn usage_is_extracted_from_chunk() {
        let mut chunk_with_usage = make_chunk(vec![ChatChunkDelta::default()]);
        chunk_with_usage.usage = Some(Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            cost_in_usd_ticks: None,
        });

        let chunks: Vec<Result<ChatCompletionChunk, SamplingError>> = vec![
            Ok(text_chunk("ok")),
            Ok(chunk_with_usage),
            Ok(final_chunk(FinishReason::Stop)),
        ];
        let raw = stream::iter(chunks).boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;

        match events.last().unwrap() {
            SamplingEvent::Completed { response, .. } => {
                let u = response.usage.as_ref().expect("usage extracted");
                assert_eq!(u.prompt_tokens, 100);
                assert_eq!(u.completion_tokens, 50);
                assert_eq!(u.total_tokens, 150);
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    /// Server-reported cost lands on the response; the REST mapper's `0`
    /// backfill means "unreported" and must yield `None`.
    #[tokio::test]
    async fn cost_is_extracted_and_zero_is_unreported() {
        for (wire, expected) in [(Some(78), Some(78)), (Some(0), None), (None, None)] {
            let mut chunk_with_usage = make_chunk(vec![ChatChunkDelta::default()]);
            chunk_with_usage.usage = Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: None,
                completion_tokens_details: None,
                cost_in_usd_ticks: wire,
            });
            let chunks: Vec<Result<ChatCompletionChunk, SamplingError>> = vec![
                Ok(text_chunk("ok")),
                Ok(chunk_with_usage),
                Ok(final_chunk(FinishReason::Stop)),
            ];
            let raw = stream::iter(chunks).boxed();
            let events = collect(stream_chat_completions(
                raw,
                None,
                rid(),
                Duration::from_secs(60),
            ))
            .await;
            match events.last().unwrap() {
                SamplingEvent::Completed { response, .. } => {
                    assert_eq!(response.cost_usd_ticks, expected, "wire {wire:?}");
                }
                other => panic!("expected Completed, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn later_missing_cost_does_not_clobber_earlier_ticks() {
        let mut first = make_chunk(vec![ChatChunkDelta::default()]);
        first.usage = Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            cost_in_usd_ticks: Some(99),
        });
        let mut second = make_chunk(vec![ChatChunkDelta::default()]);
        second.usage = Some(Usage {
            prompt_tokens: 12,
            completion_tokens: 6,
            total_tokens: 18,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            cost_in_usd_ticks: Some(0),
        });
        let chunks: Vec<Result<ChatCompletionChunk, SamplingError>> = vec![
            Ok(text_chunk("ok")),
            Ok(first),
            Ok(second),
            Ok(final_chunk(FinishReason::Stop)),
        ];
        let raw = stream::iter(chunks).boxed();
        let events = collect(stream_chat_completions(
            raw,
            None,
            rid(),
            Duration::from_secs(60),
        ))
        .await;
        match events.last().unwrap() {
            SamplingEvent::Completed { response, .. } => {
                assert_eq!(response.cost_usd_ticks, Some(99));
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }
}
