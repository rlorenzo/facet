from flask import Blueprint

stats_bp = Blueprint('stats', __name__)

from viewer.stats import routes  # noqa: E402,F401
