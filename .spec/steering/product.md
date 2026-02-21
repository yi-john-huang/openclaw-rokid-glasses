# Product Overview

## Product Description
**Project**: openclaw-rokid-glasses (Rokid Bridge)
**Version**: 0.1.0
**Type**: Standalone Bridge Service / Protocol Adapter

A lightweight SSE bridge that connects Rokid AR Glasses (via the Lingzhu/Rokid AI App platform) to the `openclaw-secure-stack` backend. It acts as a protocol adapter and security relay, translating Rokid's proprietary request format into OpenAI-compatible chat completions and streaming responses back to the glasses.

## Core Features
- **HMAC Authentication**: Verifies every Rokid request using HMAC-SHA256 with replay protection
- **SSE Streaming Relay**: Passes streaming responses from Secure Stack directly to Rokid glasses
- **Camera Image Processing**: Converts Rokid camera frames to OpenAI vision format (base64 → data URL)
- **Per-Device Conversation History**: In-memory session history keyed by device ID with TTL eviction
- **AR-Optimized Response Formatting**: Prepends a system prompt tuned for small transparent displays
- **Rate Limiting**: Sliding-window per-device rate limiting

## Target Use Case
Three primary use cases for Rokid AR Glasses wearers:
1. **Voice Conversation** — Hands-free chat via speech-to-text input routed to an AI agent
2. **Camera Photo Analysis** — User photographs something; the glasses explain what they see
3. **Voice Command Control** — Spoken commands trigger device or agent actions

## Key Value Proposition
- **Zero modification to openclaw-secure-stack**: All governance, sanitisation, and audit remain in the existing stack
- **Lightweight adapter**: Rokid Bridge only handles protocol translation — HMAC auth, image conversion, history, SSE relay
- **AR-native UX**: Automatic concise formatting for the small AR display overlay
- **Co-deployable**: Ships as a separate Docker service that joins the existing stack's Docker network

## Target Users
- Developers building Rokid AR Glasses applications on the openclaw-secure-stack platform
- End users wearing Rokid glasses who interact with AI via voice and camera

## Success Metrics
- Successful end-to-end SSE streaming from Rokid glasses through Bridge → Secure Stack → OpenClaw → back to glasses
- HMAC authentication rejecting forged or replayed requests 100% of the time
- Conversation history persisting correctly across turns per device
- Latency overhead of bridge < 20ms (excluding upstream processing)

## Technical Advantages
- **No vendor lock-in**: The SSE relay passthrough is the extension point for future Rokid format changes
- **Security by design**: Constant-time HMAC comparison, replay window, rate limiting, non-root Docker user
- **Separate deployment**: Bridge failure does not affect the secure stack or other clients
