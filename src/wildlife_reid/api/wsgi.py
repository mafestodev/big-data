"""Production WSGI entry point."""

from wildlife_reid.api.app import create_app
from wildlife_reid.api.inference import InferenceService
from wildlife_reid.common.config import Settings

settings = Settings.from_environment()
identifier = InferenceService.load(settings)
app = create_app(settings=settings, identifier=identifier)
