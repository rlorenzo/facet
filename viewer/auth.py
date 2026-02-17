import re
import hmac
from functools import wraps
from flask import request, redirect, session, jsonify, abort
from viewer.config import _share_secret, VIEWER_CONFIG


def generate_person_share_token(person_id):
    """Generate an HMAC token for sharing a person page."""
    return hmac.new(_share_secret.encode(), str(person_id).encode(), 'sha256').hexdigest()


def verify_person_share_token(person_id, token):
    """Verify an HMAC share token for a person page."""
    expected = generate_person_share_token(person_id)
    return hmac.compare_digest(token, expected)


# --- PASSWORD AUTHENTICATION ---
def _get_viewer_password():
    """Get password from viewer config, returns empty string if not set."""
    return VIEWER_CONFIG.get('password', '')


def _is_authenticated():
    """Check if current session is authenticated."""
    password = _get_viewer_password()
    if not password:
        return True  # No password required
    return session.get('authenticated', False)


def _get_edition_password():
    """Get edition password from config."""
    return VIEWER_CONFIG.get('edition_password', '')


def is_edition_enabled():
    """Check if edition mode is available (password is configured)."""
    return bool(_get_edition_password())


def is_edition_authenticated():
    """Check if current session has unlocked edition mode."""
    edition_password = _get_edition_password()
    if not edition_password:
        return False  # No password = edition disabled
    return session.get('edition_authenticated', False)


def require_edition(f):
    """Decorator that returns 403 JSON if edition mode is not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_edition_authenticated():
            return jsonify({'error': 'Edition disabled'}), 403
        return f(*args, **kwargs)
    return decorated


def register_auth_routes(app):
    """Register authentication routes and before_request hook on the app."""

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Handle login form display and submission."""
        from flask import render_template
        password = _get_viewer_password()
        if not password:
            return redirect('/')

        next_url = request.args.get('next', '/')

        if request.method == 'POST':
            if request.form.get('password') == password:
                session['authenticated'] = True
                next_url = request.form.get('next', '/')
                return redirect(next_url)
            from i18n import _ as translate
            return render_template('login.html', error=translate('login.invalid_password'), next_url=next_url)

        return render_template('login.html', error=None, next_url=next_url)

    @app.route('/api/edition/login', methods=['POST'])
    def api_edition_login():
        """Authenticate for edition mode."""
        data = request.get_json() or {}
        password = data.get('password', '')
        edition_password = _get_edition_password()
        if edition_password and password == edition_password:
            session['edition_authenticated'] = True
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Invalid password'}), 401

    @app.route('/api/edition/logout', methods=['POST'])
    def api_edition_logout():
        """Log out of edition mode."""
        session.pop('edition_authenticated', None)
        return jsonify({'success': True})

    @app.route('/api/person/<int:person_id>/share-token')
    def api_person_share_token(person_id):
        """Generate a share URL token for a person page. Only available to local (non-shared) users."""
        if session.get('shared_person_id') is not None:
            abort(403)
        token = generate_person_share_token(person_id)
        return jsonify({'token': token})

    @app.before_request
    def check_access():
        """Check authentication and gate shared visitors."""
        # Allow login route without authentication
        if request.path == '/login':
            return None

        # Allow static assets without authentication
        if request.path.startswith('/static/'):
            return None

        # Check if incoming request has a share token on a person page
        match = re.match(r'^/person/(\d+)', request.path)
        if match and request.args.get('token'):
            person_id = int(match.group(1))
            token = request.args.get('token')
            if verify_person_share_token(person_id, token):
                session['shared_person_id'] = person_id
                # Redirect to strip the token from the URL
                from urllib.parse import urlencode
                args = {k: v for k, v in request.args.items() if k != 'token'}
                clean_url = request.path
                if args:
                    clean_url += '?' + urlencode(args)
                return redirect(clean_url)
            else:
                abort(403)

        # If session marks this visitor as a shared visitor, restrict routes
        shared_pid = session.get('shared_person_id')
        if shared_pid is not None:
            allowed_prefixes = [
                f'/person/{shared_pid}',
                '/thumbnail',
                f'/person_thumbnail/{shared_pid}',
                '/api/download-selected',
                '/api/download',
                '/static/',
            ]
            if not any(request.path.startswith(p) for p in allowed_prefixes):
                # Clear shared session and let normal auth flow handle it
                session.pop('shared_person_id', None)
                # Fall through to password authentication check below

        # Check password authentication for all other routes
        if not _is_authenticated():
            # For API routes, return 401
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            # For regular routes, redirect to login
            from urllib.parse import urlencode
            next_url = request.path
            if request.query_string:
                next_url += '?' + request.query_string.decode('utf-8')
            return redirect(f'/login?next={next_url}')
