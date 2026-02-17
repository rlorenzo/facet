from flask import Blueprint

thumbnails_bp = Blueprint('thumbnails', __name__)

from viewer.thumbnails import routes  # noqa: E402,F401
