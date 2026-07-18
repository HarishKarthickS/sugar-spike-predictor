"""Flask application factory for Sugar Spike Predictor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask

from app.config import Config


def create_app(config: type[Config] = Config):
    from flask import Flask

    from app.routes import bp

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    )

    app.register_blueprint(bp)
    return app
