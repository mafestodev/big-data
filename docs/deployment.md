# Demo Deployment

## Architecture

```text
Public HTTPS URL
        |
Cloudflare Quick Tunnel
        |
Gunicorn on 127.0.0.1:7860
        |
Flask frontend and API
        |
Swin model and embedding gallery
```

The model, gallery, and dataset run on the demonstration computer. Cloudflare
provides a temporary encrypted public address and forwards requests to the
local Gunicorn server. The gallery stores paths relative to `data/archive`, so
the application can return the matched reference image with the predicted
identity, species, similarity score, and processing time. The complete dataset
is not uploaded to Cloudflare; only requested images pass through the tunnel.

## Requirements

- `models/swin_embedding_model.pt`
- `gallery/embeddings.npz`
- `data/archive/images/`
- The project virtual environment and dependencies
- `cloudflared`
- An active internet connection

## Start the deployment

In the first terminal:

```bash
./scripts/run_demo_server.sh
```

Wait for the model and gallery to load, then verify:

```bash
curl http://127.0.0.1:7860/health
```

The response must contain `"model_ready": true`.

In a second terminal:

```bash
./scripts/start_quick_tunnel.sh
```

Open the generated `trycloudflare.com` address from a separate device or
network and perform an identification request.

## Deployment evidence

Capture the following for assessment:

1. The terminal showing Gunicorn running.
2. The terminal showing the public Cloudflare URL.
3. The public application open on another device.
4. A successful identification result.
5. The public `/health` response showing that the model is ready.

## Limitations

- The public hostname changes whenever the Quick Tunnel restarts.
- The demonstration computer and both processes must remain running.
- Quick Tunnels provide no uptime guarantee and are not intended for permanent
  production hosting.
- Reference images remain available only while the local dataset is present.
