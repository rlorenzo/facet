from flask import Blueprint

comparison_bp = Blueprint('comparison', __name__)

from viewer.comparison import routes  # noqa: E402,F401
