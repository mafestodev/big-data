from __future__ import annotations

import time
from typing import Protocol

from flask import Flask, jsonify, render_template, request

from wildlife_reid.api.preprocessing import InvalidImageError, decode_rgb_image
from wildlife_reid.common.config import Settings


class Identifier(Protocol):
    def identify(self, image): ...


def create_app(settings: Settings | None = None, identifier: Identifier | None = None) -> Flask:
    settings = settings or Settings.from_environment()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.extensions["wildlife_identifier"] = identifier
    app.extensions["wildlife_settings"] = settings

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        ready = app.extensions["wildlife_identifier"] is not None
        return jsonify({"status": "ok", "model_ready": ready})

    @app.get("/model-info")
    def model_info():
        return jsonify(
            {
                "model_name": settings.model_name,
                "embedding_dimension": settings.embedding_dimension,
                "image_size": settings.image_size,
                "model_ready": app.extensions["wildlife_identifier"] is not None,
            }
        )

    @app.post("/api/v1/identify")
    def identify():
        service = app.extensions["wildlife_identifier"]
        if service is None:
            return jsonify({"error": "The trained model and gallery are not loaded"}), 503
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify({"error": "Provide an image using the 'image' form field"}), 400
        try:
            image = decode_rgb_image(uploaded.read())
        except InvalidImageError as exc:
            return jsonify({"error": str(exc)}), 400

        started = time.perf_counter()
        result = service.identify(image)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return jsonify(
            {
                "predicted_identity": result.identity,
                "similarity_score": round(result.similarity, 6),
                "processing_time_ms": elapsed_ms,
            }
        )

    return app


def main() -> None:
    settings = Settings.from_environment()
    identifier = None
    try:
        from wildlife_reid.api.inference import InferenceService

        identifier = InferenceService.load(settings)
    except FileNotFoundError:
        # The scaffold can run before training; /health exposes model_ready=false.
        pass
    create_app(settings, identifier).run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()

