from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from wildlife_reid.api.app import create_app


@dataclass
class FakeResult:
    identity: str = "animal-00482"
    similarity: float = 0.87


class FakeIdentifier:
    def identify(self, image):
        assert image.mode == "RGB"
        return FakeResult()


def image_payload() -> BytesIO:
    stream = BytesIO()
    Image.new("RGB", (16, 16), "green").save(stream, format="JPEG")
    stream.seek(0)
    return stream


def test_health_reports_model_ready():
    client = create_app(identifier=FakeIdentifier()).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"model_ready": True, "status": "ok"}


def test_identify_returns_prediction():
    client = create_app(identifier=FakeIdentifier()).test_client()

    response = client.post(
        "/api/v1/identify",
        data={"image": (image_payload(), "animal.jpg")},
        content_type="multipart/form-data",
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["predicted_identity"] == "animal-00482"
    assert body["similarity_score"] == 0.87
    assert body["processing_time_ms"] >= 0


def test_identify_requires_image():
    client = create_app(identifier=FakeIdentifier()).test_client()

    response = client.post("/api/v1/identify")

    assert response.status_code == 400

