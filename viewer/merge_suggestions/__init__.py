from flask import Blueprint

merge_suggestions_bp = Blueprint('merge_suggestions', __name__)

from viewer.merge_suggestions import routes  # noqa: E402,F401
