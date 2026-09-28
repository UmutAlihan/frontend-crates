// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fastokens backend using the `fastokens` crate for high-performance BPE encoding.
//!
//! Two entry points share the encoder:
//!
//! - [`FastTokenizer`] loads a HuggingFace `tokenizer.json` and keeps the existing hybrid
//!   behavior: `fastokens` encodes, `HuggingFaceTokenizer` decodes, both from the same file.
//! - [`FastTikTokenTokenizer`] loads a bare `tiktoken.model` (the Kimi family ships no
//!   `tokenizer.json`) with the same pattern and special-token resolution as
//!   [`TikTokenTokenizer`](crate::TikTokenTokenizer), and encodes *and* decodes through
//!   `fastokens`, so no HuggingFace file is needed.

use std::collections::HashSet;
use std::path::Path;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::{
    EncodeSegment, Encoding, Error, Result, TokenIdType,
    hf::HuggingFaceTokenizer,
    tiktoken,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

fn fast_encode(encoder: &fastokens::Tokenizer, input: &str) -> Result<Encoding> {
    let ids = encoder
        .encode(input)
        .map_err(|e| Error::msg(format!("Fastokens encode error: {e}")))?;
    Ok(Encoding::Sp(ids))
}

fn fast_encode_segments(
    encoder: &fastokens::Tokenizer,
    segments: &[EncodeSegment<'_>],
) -> Result<Encoding> {
    let segments: Vec<fastokens::EncodeSegment<'_>> = segments
        .iter()
        .map(|segment| fastokens::EncodeSegment {
            text: segment.text,
            allow_special: segment.allow_special,
        })
        .collect();
    let ids = encoder
        .encode_segments(&segments)
        .map_err(|e| Error::msg(format!("Fastokens segmented encode error: {e}")))?;
    Ok(Encoding::Sp(ids))
}

/// Hybrid tokenizer: fast BPE encoding via `fastokens`, decoding via HuggingFace.
///
/// Both backends are loaded from the same `tokenizer.json` file.
pub struct FastTokenizer {
    fast_encoder: fastokens::Tokenizer,
    hf_decoder: HuggingFaceTokenizer,
}

impl FastTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let fast_encoder = fastokens::Tokenizer::from_file(Path::new(path))
            .map_err(|e| Error::msg(format!("Error loading fastokens tokenizer: {e}")))?;
        let hf_decoder = HuggingFaceTokenizer::from_file(path)?;
        Ok(Self {
            fast_encoder,
            hf_decoder,
        })
    }
}

impl Encoder for FastTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        fast_encode(&self.fast_encoder, input)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.par_iter().map(|input| self.encode(input)).collect()
    }

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        fast_encode_segments(&self.fast_encoder, segments)
    }
}

impl Decoder for FastTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.hf_decoder.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for FastTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        Ok(())
    }

    // `fast_encoder` and `hf_decoder` are loaded from the same tokenizer.json,
    // so the HF side's vocabulary introspection applies to both.
    fn vocab_size(&self) -> Option<usize> {
        self.hf_decoder.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        self.hf_decoder.token_to_id(token)
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        self.hf_decoder.special_token_ids()
    }

    fn num_special_tokens_added(&self) -> Result<usize> {
        Ok(0)
    }
}

/// `fastokens` over a bare `tiktoken.model`, for the Kimi family and other models that
/// ship tiktoken ranks instead of a `tokenizer.json`.
///
/// The vocabulary, pre-tokenization regex, and special-token table are resolved from the
/// same files by the same helpers as [`TikTokenTokenizer`](crate::TikTokenTokenizer), so
/// the two produce identical ids; only the BPE engine differs (`fastokens`' parallel
/// byte-level BPE instead of `tiktoken_rs`). Decoding joins the raw token bytes from the
/// same ranks, exactly like the tiktoken backend, so unlike [`FastTokenizer`] no
/// HuggingFace tokenizer is loaded.
pub struct FastTikTokenTokenizer {
    inner: fastokens::Tokenizer,
    /// Raw bytes per id for every rank and special token. `decode` joins these itself so
    /// UTF-8 completeness is judged on bytes: `fastokens`' decoder returns a lossy `String`,
    /// which cannot tell a vocabulary token that legitimately ends in `EF BF BD` from a
    /// truncated multi-byte sequence.
    id_to_bytes: FxHashMap<u32, Vec<u8>>,
    special_token_ids: HashSet<u32>,
    special_tokens: Vec<String>,
}

impl FastTikTokenTokenizer {
    /// Build from a tiktoken model file, auto-detecting the BPE pattern from `config.json`
    /// and the special tokens from `tokenizer_config.json` in the same directory — the same
    /// resolution as [`TikTokenTokenizer::from_file_auto`](crate::TikTokenTokenizer::from_file_auto).
    pub fn from_file_auto(path: &str) -> Result<Self> {
        let file_path = Path::new(path);
        let directory = file_path
            .parent()
            .ok_or_else(|| Error::msg("Cannot determine parent directory of tiktoken file"))?;

        let pattern = tiktoken::detect_bpe_pattern(directory)?;
        let encoder = tiktoken::parse_tiktoken_file(path)?;
        // Use max rank + 1 (not len) to avoid ID collisions with sparse/non-contiguous ranks
        let num_base_tokens = encoder.values().max().map_or(0, |&m| m + 1) as usize;
        let special_tokens = tiktoken::load_special_tokens(directory, num_base_tokens)?;
        Self::from_ranks(encoder, pattern, special_tokens, path)
    }

    /// Build from a tiktoken model file with an explicit BPE regex pattern and special-token
    /// map, mirroring [`TikTokenTokenizer::from_file`](crate::TikTokenTokenizer::from_file).
    pub fn from_file(
        path: &str,
        pattern: &str,
        special_tokens: FxHashMap<String, u32>,
    ) -> Result<Self> {
        let encoder = tiktoken::parse_tiktoken_file(path)?;
        Self::from_ranks(encoder, pattern, special_tokens, path)
    }

    fn from_ranks(
        encoder: FxHashMap<Vec<u8>, u32>,
        pattern: &str,
        special_tokens: FxHashMap<String, u32>,
        path: &str,
    ) -> Result<Self> {
        let mut ranks: Vec<(Vec<u8>, u32)> = encoder.into_iter().collect();
        ranks.sort_unstable_by_key(|(_, rank)| *rank);

        let mut id_to_bytes: FxHashMap<u32, Vec<u8>> = ranks
            .iter()
            .map(|(bytes, rank)| (*rank, bytes.clone()))
            .collect();
        for (content, id) in &special_tokens {
            id_to_bytes.insert(*id, content.as_bytes().to_vec());
        }

        let special_token_ids: HashSet<u32> = special_tokens.values().copied().collect();
        let special_token_strings = tiktoken::sorted_special_token_strings(&special_tokens);
        let config =
            fastokens::tiktoken::TiktokenConfig::new(pattern, special_tokens.into_iter().collect());
        let inner = fastokens::Tokenizer::from_tiktoken_ranks(&ranks, config).map_err(|e| {
            Error::msg(format!(
                "Error loading fastokens tiktoken tokenizer from {path}: {e}"
            ))
        })?;
        Ok(Self {
            inner,
            id_to_bytes,
            special_token_ids,
            special_tokens: special_token_strings,
        })
    }

    /// Atomic special-token strings registered with the encoder, sorted; the boundary set
    /// for [`CachedTokenizer`](crate::CachedTokenizer).
    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }
}

impl Encoder for FastTikTokenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        fast_encode(&self.inner, input)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.par_iter().map(|input| self.encode(input)).collect()
    }

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        fast_encode_segments(&self.inner, segments)
    }
}

impl Decoder for FastTikTokenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        // Same procedure as `TikTokenTokenizer::decode`: join raw bytes (unknown ids are
        // skipped, like fastokens and HuggingFace do), then try strict UTF-8 first so a
        // vocabulary token whose bytes end in EF BF BD is `Complete`; only genuinely
        // invalid bytes fall back to lossy conversion and the trailing-U+FFFD rule.
        let mut bytes = Vec::new();
        for &id in token_ids {
            if skip_special_tokens && self.special_token_ids.contains(&id) {
                continue;
            }
            if let Some(token) = self.id_to_bytes.get(&id) {
                bytes.extend_from_slice(token);
            }
        }
        match String::from_utf8(bytes) {
            Ok(text) => Ok(DecodeResult::Complete(text)),
            Err(e) => Ok(DecodeResult::from_decoded(
                String::from_utf8_lossy(e.as_bytes()).into_owned(),
            )),
        }
    }
}

impl Tokenizer for FastTikTokenTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        Ok(())
    }

    // Same per-segment, no-post-processor composition as `FastTokenizer`.

    fn vocab_size(&self) -> Option<usize> {
        Some(self.inner.vocab_size())
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        Ok(self.inner.token_to_id(token))
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        let mut ids: Vec<TokenIdType> = self.special_token_ids.iter().copied().collect();
        ids.sort_unstable();
        Ok(ids)
    }

    // No post-processor: nothing is added beyond the caller's text.
    fn num_special_tokens_added(&self) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HuggingFaceTokenizer, TokenizerOptions};

    // Minimal synthetic BPE tokenizer with no normalizer or post-processor --
    // compatible with fastokens. Vocab covers: H,T,a,d,e,h,i,l,o,r,s,t,w + punctuation.
    const TOKENIZER_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/minimal-bpe/tokenizer.json"
    );
    const SEGMENTED_TOKENIZER_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
    );

    #[test]
    fn test_fast_encode_decode_roundtrip() {
        let tokenizer = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        // Encode then decode: verifies both paths execute without error.
        // With a null decoder, HF inserts spaces between tokens so exact equality
        // is not expected here -- we just verify the operations succeed and produce
        // non-empty results.
        let text = "Hello, world!";
        let encoding = tokenizer.encode(text).unwrap();
        assert!(!encoding.token_ids().is_empty());
        let decoded: String = tokenizer.decode(encoding.token_ids(), true).unwrap().into();
        assert!(!decoded.is_empty());
        // The decoded text should contain the same non-space characters
        let enc_chars: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let dec_chars: String = decoded.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(
            enc_chars, dec_chars,
            "non-space characters must be preserved"
        );
    }

    #[test]
    fn test_fast_matches_hf_encoding() {
        let fast = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let hf = HuggingFaceTokenizer::from_file(TOKENIZER_PATH).unwrap();

        for text in &["Hello, world!", "Hello", " world", "He llo"] {
            let fast_ids = fast.encode(text).unwrap();
            let hf_ids = hf.encode(text).unwrap();
            assert_eq!(
                fast_ids.token_ids(),
                hf_ids.token_ids(),
                "fastokens and HuggingFace must produce identical token IDs for '{text}'"
            );
        }
    }

    #[test]
    fn test_fast_batch_encode() {
        let tokenizer = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let inputs = &["Hello", " world", "Hello, world!"];
        let encodings = tokenizer.encode_batch(inputs).unwrap();
        assert_eq!(encodings.len(), inputs.len());
        for (enc, input) in encodings.iter().zip(inputs.iter()) {
            assert!(
                !enc.token_ids().is_empty(),
                "encoding for '{input}' must be non-empty"
            );
        }
    }

    #[test]
    fn test_fast_segmented_encoding_preserves_trust_boundaries() {
        let tokenizer = FastTokenizer::from_file(SEGMENTED_TOKENIZER_PATH).unwrap();
        let upstream =
            fastokens::Tokenizer::from_file(std::path::Path::new(SEGMENTED_TOKENIZER_PATH))
                .unwrap();
        let marker = "<s>";

        let trusted = tokenizer
            .encode_segments(&[EncodeSegment::control(marker)])
            .unwrap();
        assert_eq!(
            trusted.token_ids(),
            &[upstream.token_to_id(marker).unwrap()],
            "trusted renderer output must recognize the control token"
        );

        let ordinary = tokenizer
            .encode_segments(&[EncodeSegment::ordinary(marker)])
            .unwrap();
        assert_ne!(
            ordinary.token_ids(),
            trusted.token_ids(),
            "untrusted content must encode the control-token spelling as ordinary text"
        );

        let segments = [
            EncodeSegment::ordinary("hello "),
            EncodeSegment::control(marker),
            EncodeSegment::ordinary(marker),
        ];
        let upstream_segments = [
            fastokens::EncodeSegment::ordinary("hello "),
            fastokens::EncodeSegment::special(marker),
            fastokens::EncodeSegment::ordinary(marker),
        ];
        let actual = tokenizer.encode_segments(&segments).unwrap();
        let expected = upstream.encode_segments(&upstream_segments).unwrap();
        assert_eq!(actual.token_ids(), expected);

        assert!(
            tokenizer
                .encode_segments(&[])
                .unwrap()
                .token_ids()
                .is_empty()
        );
    }

    #[test]
    fn test_fast_with_decode_stream() {
        use crate::Tokenizer as TokenizerWrapper;
        use std::sync::Arc;

        let tokenizer = Arc::new(FastTokenizer::from_file(TOKENIZER_PATH).unwrap());
        let wrapper = TokenizerWrapper::from(tokenizer);

        // Encode a prompt and a continuation, then step through the decode stream
        let prompt_ids = wrapper.encode("Hello").unwrap().token_ids().to_vec();
        let continuation = ", world!";
        let cont_ids = wrapper.encode(continuation).unwrap().token_ids().to_vec();

        let mut stream = wrapper.decode_stream(&prompt_ids, true);
        // Accumulate incremental chunks from decode_stream
        let mut accumulated = String::new();
        for id in &cont_ids {
            if let Some(chunk) = stream.step(*id).unwrap() {
                accumulated.push_str(&chunk);
            }
        }

        // DecodeStream uses prompt tokens as context, so the expected text is
        // decode(prompt + continuation) minus decode(prompt) -- not a bare
        // decode(continuation) which lacks the surrounding context.
        let mut all_ids = prompt_ids.clone();
        all_ids.extend_from_slice(&cont_ids);
        let full_text: String = wrapper.decode(&all_ids, true).unwrap().into();
        let prompt_text: String = wrapper.decode(&prompt_ids, true).unwrap().into();
        let expected = &full_text[prompt_text.len()..];
        assert_eq!(
            accumulated, expected,
            "streamed chunks must equal context-aware decoded continuation"
        );
    }

    #[test]
    fn vocabulary_metadata_forwards_to_hf_decoder() {
        let fast = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let hf = HuggingFaceTokenizer::from_file(TOKENIZER_PATH).unwrap();
        assert_eq!(fast.vocab_size(), hf.vocab_size());
        assert_eq!(
            fast.token_to_id("Hello").unwrap(),
            hf.token_to_id("Hello").unwrap()
        );
        assert_eq!(
            fast.special_token_ids().unwrap(),
            hf.special_token_ids().unwrap()
        );
    }

    #[test]
    fn special_token_accounting_matches_fast_encoder() {
        let fast = FastTokenizer::from_file(SEGMENTED_TOKENIZER_PATH).unwrap();
        let upstream =
            fastokens::Tokenizer::from_file(std::path::Path::new(SEGMENTED_TOKENIZER_PATH))
                .unwrap();
        let hf = HuggingFaceTokenizer::from_file(SEGMENTED_TOKENIZER_PATH).unwrap();

        assert_eq!(hf.num_special_tokens_added().unwrap(), 1);
        assert_eq!(fast.num_special_tokens_added().unwrap(), 0);
        let hf_with_special_tokens = hf.with_options(TokenizerOptions {
            add_special_tokens: true,
        });

        for text in ["hello", "hello there"] {
            let fast_ids = fast.encode(text).unwrap();
            assert_eq!(
                fast_ids.token_ids(),
                upstream.encode(text).unwrap(),
                "FastTokenizer must match the encoder that omits the HF post-processor"
            );
            assert_eq!(
                hf_with_special_tokens
                    .encode(text)
                    .unwrap()
                    .token_ids()
                    .len(),
                fast_ids.token_ids().len() + 1,
                "the HF post-processor must add the BOS token FastTokenizer omits"
            );
        }
    }
}

#[cfg(test)]
mod tiktoken_parity_tests {
    use super::*;
    use crate::TikTokenTokenizer;

    const TIKTOKEN_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/mock-tiktoken-bpe/tiktoken.model"
    );

    fn pair() -> (TikTokenTokenizer, FastTikTokenTokenizer) {
        let reference = TikTokenTokenizer::from_file_auto(TIKTOKEN_PATH)
            .expect("load mock Kimi tiktoken fixture through tiktoken-rs");
        let fast = FastTikTokenTokenizer::from_file_auto(TIKTOKEN_PATH)
            .expect("load mock Kimi tiktoken fixture through fastokens");
        (reference, fast)
    }

    fn corpus() -> Vec<String> {
        vec![
            String::new(),
            "hello world".into(),
            "Hello, World! 123 4567 89".into(),
            "  leading and trailing  ".into(),
            "tabs\tand\nnewlines\r\nmixed   spacing".into(),
            "<|im_start|>user\nhi there<|im_end|><|im_start|>assistant\n".into(),
            "a literal <|im_start|> inside plain text".into(),
            "emoji 😀🚀 and café naïve Zürich".into(),
            "北京 東京 mixed 中英文 text ソフトウェア".into(),
            "Москва मुंबई العربية".into(),
            "fn main() { println!(\"{}\", 42); } // code-ish ~!@#$%^&*()".into(),
            "x".repeat(5000),
            " ".repeat(300) + "after long whitespace",
            "word ".repeat(400),
        ]
    }

    #[test]
    fn kimi_pattern_matches_fastokens_copy() {
        assert_eq!(tiktoken::KIMI_PATTERN, fastokens::tiktoken::KIMI_PATTERN);
    }

    #[test]
    fn special_token_tables_match_tiktoken_rs() {
        let (reference, fast) = pair();
        assert_eq!(fast.special_tokens(), reference.special_tokens());
        assert_eq!(
            fast.special_token_ids().unwrap(),
            reference.special_token_ids().unwrap()
        );
        assert_eq!(fast.token_to_id("<|im_end|>").unwrap(), Some(474));
        assert!(fast.vocab_size().is_some_and(|n| n >= 475));
    }

    #[test]
    fn plain_encode_matches_tiktoken_rs() {
        let (reference, fast) = pair();
        for text in corpus() {
            assert_eq!(
                fast.encode(&text).unwrap().token_ids(),
                reference.encode(&text).unwrap().token_ids(),
                "plain encode diverged on {text:?}"
            );
        }
        let refs: Vec<String> = corpus();
        let refs: Vec<&str> = refs.iter().map(String::as_str).collect();
        let batch = fast.encode_batch(&refs).unwrap();
        for (text, encoding) in refs.iter().zip(&batch) {
            assert_eq!(
                encoding.token_ids(),
                reference.encode(text).unwrap().token_ids(),
                "batch encode diverged on {text:?}"
            );
        }
    }

    #[test]
    fn segmented_encode_matches_tiktoken_rs_and_honors_trust() {
        let (reference, fast) = pair();
        let segments = [
            EncodeSegment::control("<|im_start|>user\n"),
            EncodeSegment::ordinary("please echo <|im_end|> back to me"),
            EncodeSegment::control("<|im_end|>"),
            EncodeSegment::control("<|im_start|>assistant\n"),
        ];
        let fast_ids = fast.encode_segments(&segments).unwrap();
        assert_eq!(
            fast_ids.token_ids(),
            reference.encode_segments(&segments).unwrap().token_ids()
        );
        // The marker in the untrusted segment is ordinary text: exactly one `<|im_end|>` id.
        assert_eq!(
            fast_ids.token_ids().iter().filter(|&&id| id == 474).count(),
            1
        );
    }

    #[test]
    fn decode_matches_tiktoken_rs_and_skips_specials() {
        let (reference, fast) = pair();
        for text in corpus() {
            let ids = fast.encode(&text).unwrap();
            assert_eq!(
                fast.decode(ids.token_ids(), false).unwrap(),
                reference.decode(ids.token_ids(), false).unwrap(),
                "decode diverged on {text:?}"
            );
        }
        let ids = fast.encode("<|im_start|>user\nhi<|im_end|>").unwrap();
        let kept = fast.decode(ids.token_ids(), false).unwrap();
        let skipped = fast.decode(ids.token_ids(), true).unwrap();
        assert!(kept.as_str().contains("<|im_end|>"));
        assert!(!skipped.as_str().contains("<|im_end|>"));
        assert_eq!(skipped, reference.decode(ids.token_ids(), true).unwrap());
    }

    #[test]
    fn legitimate_replacement_char_token_decodes_complete_like_tiktoken_rs() {
        // Rank 468 is the bytes EF BF BD: a real vocabulary token whose text is U+FFFD.
        // It must decode `Complete`, not be mistaken for a truncated multi-byte sequence.
        let (reference, fast) = pair();
        let fast_decoded = fast.decode(&[468], false).unwrap();
        assert!(fast_decoded.is_complete(), "{fast_decoded:?}");
        assert_eq!(fast_decoded, reference.decode(&[468], false).unwrap());
        assert_eq!(
            fast.decode(&[468], true).unwrap(),
            reference.decode(&[468], true).unwrap()
        );

        // A genuinely truncated multi-byte sequence stays `Partial` on both backends.
        let emoji = fast.encode("😀").unwrap();
        let cut = &emoji.token_ids()[..emoji.token_ids().len() - 1];
        assert!(!cut.is_empty());
        let fast_cut = fast.decode(cut, false).unwrap();
        assert!(fast_cut.is_partial(), "{fast_cut:?}");
        assert_eq!(fast_cut, reference.decode(cut, false).unwrap());

        // Unknown ids are skipped rather than failing the whole decode.
        assert_eq!(
            fast.decode(&[468, 9_999_999], false).unwrap(),
            fast.decode(&[468], false).unwrap()
        );
    }

    #[test]
    fn missing_config_is_an_error_not_a_guess() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::copy(TIKTOKEN_PATH, dir.path().join("tiktoken.model")).unwrap();
        let path = dir.path().join("tiktoken.model");
        assert!(FastTikTokenTokenizer::from_file_auto(path.to_str().unwrap()).is_err());
    }

    /// Parity against the real Kimi K3 vocabulary. Ignored by default: point
    /// `DYNAMO_TOKENIZERS_KIMI_DIR` at a directory holding `tiktoken.model`,
    /// `tokenizer_config.json` and `config.json`, then run with `--ignored`.
    #[test]
    #[ignore = "needs a local Kimi checkpoint directory in DYNAMO_TOKENIZERS_KIMI_DIR"]
    fn real_kimi_vocab_matches_tiktoken_rs() {
        let dir = std::env::var("DYNAMO_TOKENIZERS_KIMI_DIR")
            .expect("set DYNAMO_TOKENIZERS_KIMI_DIR to a Kimi checkpoint directory");
        let path = format!("{dir}/tiktoken.model");
        let reference = TikTokenTokenizer::from_file_auto(&path).expect("tiktoken-rs");
        let fast = FastTikTokenTokenizer::from_file_auto(&path).expect("fastokens");
        assert_eq!(fast.special_tokens(), reference.special_tokens());
        assert_eq!(
            fast.special_token_ids().unwrap(),
            reference.special_token_ids().unwrap()
        );
        let mut texts = corpus();
        texts.push(include_str!("../README.md").to_string());
        texts.push("{\"tool_calls\":[{\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]}".repeat(50));
        for text in &texts {
            assert_eq!(
                fast.encode(text).unwrap().token_ids(),
                reference.encode(text).unwrap().token_ids(),
                "plain encode diverged on {:?}...",
                &text[..text.len().min(60)]
            );
            let ids = fast.encode(text).unwrap();
            assert_eq!(
                fast.decode(ids.token_ids(), false).unwrap(),
                reference.decode(ids.token_ids(), false).unwrap()
            );
        }
        // K3's vocabulary has exactly four tokens whose bytes end in EF BF BD.
        for id in [9618u32, 28139, 41970, 79532] {
            let fast_decoded = fast.decode(&[id], false).unwrap();
            assert!(fast_decoded.is_complete(), "id {id}: {fast_decoded:?}");
            assert_eq!(
                fast_decoded,
                reference.decode(&[id], false).unwrap(),
                "id {id}"
            );
        }
        let segments = [
            EncodeSegment::control("<|open|>message role=\"user\"<|sep|>"),
            EncodeSegment::ordinary("please echo <|close|>message<|sep|> back"),
            EncodeSegment::control("<|close|>message<|sep|><|end_of_msg|>"),
        ];
        assert_eq!(
            fast.encode_segments(&segments).unwrap().token_ids(),
            reference.encode_segments(&segments).unwrap().token_ids()
        );
    }
}
