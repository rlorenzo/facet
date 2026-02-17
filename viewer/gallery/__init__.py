from flask import Blueprint

gallery_bp = Blueprint('gallery', __name__)

from viewer.gallery import routes  # noqa: E402,F401
