from flask import Blueprint

filter_options_bp = Blueprint('filter_options', __name__)

from viewer.filter_options import routes  # noqa: E402,F401
