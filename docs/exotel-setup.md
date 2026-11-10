# Exotel AgentStream setup

## Server environment

Set these in the backend `.env` (never in browser code or chat):

```dotenv
EXOTEL_ACCOUNT_SID=
EXOTEL_API_KEY=
EXOTEL_API_TOKEN=
EXOTEL_API_BASE_URL=https://api.in.exotel.com
EXOTEL_EXOPHONE=
PUBLIC_BASE_URL=https://your-public-host.example
```

`PUBLIC_BASE_URL` is the public HTTPS origin that tunnels/forwards to this FastAPI server (port 8000 locally). `/health` returns the generated `exotel_stream_url`; for example, `https://demo.example` becomes `wss://demo.example/ws/exotel`. Do not use `/ws/call`, which is the browser simulator protocol.

Run the app with `uv run asha-vani` and expose port 8000 using the chosen HTTPS tunnel. The tunnel URL changes when a temporary tunnel restarts; update Exotel then. Keep the API key/token in Exotel's secure endpoint-auth fields if offered. If its AgentStream form accepts URL user-info, its URL shape is `wss://<API_KEY>:<API_TOKEN>@<public-host>/ws/exotel`; do not paste a credential-bearing URL into source control or logs.

## Exotel call flow

1. Configure the AgentStream/Voicebot applet for 8 kHz, mono, signed linear PCM (16-bit little-endian) and point it at `/ws/exotel`.
2. Enable Basic Auth with the API Key as username and API Token as password. The server checks the `Authorization` header before accepting the socket.
3. Assign the app to the ASHA-VANI ExoPhone.
4. Set the inbound flow to **Voicebot → Hangup**. A goodbye is sent only after the graph receives “no”; Exotel playback is acknowledged with a mark before the server closes the socket, allowing the flow to reach Hangup.

The current route handles AgentStream `connected`, `start`, `media`, `mark`, `dtmf`, and `stop` events. Audio is decoded to the same 8 kHz PCM16 path used by the browser session and outbound Timbre audio is packetized for AgentStream. The demo still needs a real inbound call to validate account-specific applet settings, network behavior, and audio timing.

## Database prerequisite

Run `supabase/migrations/202611100002_visit_reports_and_history.sql` in the Supabase SQL Editor after migration 001. This adds separate patient profiles, per-call visit numbers, and confirmed report snapshots while preserving the current data. The server uses the service-role key; it must remain server-side.
