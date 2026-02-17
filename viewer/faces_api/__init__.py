from flask import Blueprint

faces_api_bp = Blueprint('faces_api', __name__)

from viewer.faces_api import routes  # noqa: E402,F401
