from flask import Blueprint

persons_bp = Blueprint('persons', __name__)

from viewer.persons import routes  # noqa: E402,F401
