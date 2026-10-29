# Person A: Voice Pipeline and Phone Simulator

See the [architecture plan](ASHA-Vani-architecture-and-plan%20(1).md) for full details. Person A owns speech transport and the simulated caller experience.

## Call flow

```text
Phone simulator → audio ingest → Prisma streaming transcript
       → Person B's graph (one completed utterance at a time)
       → reply text → Timbre audio → phone simulator
```

The graph owns conversation decisions. Audio streaming, buffering, playback, and call lifecycle stay in the call session. Start half-duplex: pause/mute incoming audio while the system speaks.

## Build

1. **Ingest:** simulator/replay audio adapter; normalize to the format Prisma expects (planned: 8 kHz PCM16, 1024-byte frames).
2. **Prisma:** one streaming connection per call; handle partial/final transcripts and end-of-utterance detection. Verify current API events and headers before coding against them.
3. **Call session:** keep turns ordered; connect audio-in, transcript reader, graph handoff, and speaker queues; handle disconnects, timeouts, concurrency caps, and latency logging.
4. **Timbre:** stream graph replies back as audio. Use fixed audio/text for critical safety prompts where possible.
5. **Simulator and replay:** mic/noise controls, preset synthetic scenarios, visible SIM label, and repeatable replay runs.

## Agree with Person B

- Graph input: session ID, utterance sequence, final transcript, and available language metadata.
- Graph output: reply text plus the action/turn status the voice layer needs.
- Event names/payloads, error behavior, and who owns each API boundary.
- Whether Prisma can signal a configured end-word such as “OVER,” or only transcribe it. Treat end-word handling as a proposal until verified.

Initial demo is simulated telephony; real PSTN/Twilio is stretch scope. Use synthetic calls and label simulation clearly.
