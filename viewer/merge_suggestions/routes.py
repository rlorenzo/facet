from flask import render_template, request, redirect, jsonify
from db import DEFAULT_DB_PATH
from viewer.merge_suggestions import merge_suggestions_bp
from viewer.config import load_viewer_config
from viewer.auth import is_edition_authenticated


@merge_suggestions_bp.route('/api/merge_groups')
def api_merge_groups():
    """Return merge groups as JSON for the suggest_merges page."""
    if not is_edition_authenticated():
        return jsonify({'error': 'Edition disabled'}), 403

    # Check if feature is enabled
    viewer_config = load_viewer_config()
    if not viewer_config.get('features', {}).get('show_merge_suggestions', True):
        return jsonify({'error': 'Merge suggestions feature is disabled'}), 403

    threshold = float(request.args.get('threshold', 0.6))
    # Lazy import only when feature is used
    from faces import get_merge_groups
    groups = get_merge_groups(DEFAULT_DB_PATH, threshold)
    return jsonify({'groups': groups, 'threshold': threshold})


@merge_suggestions_bp.route('/suggest_merges')
def suggest_merges_page():
    """Display a page showing merge suggestions grouped by similarity."""
    if not is_edition_authenticated():
        return redirect('/')

    # Check if feature is enabled
    viewer_config = load_viewer_config()
    if not viewer_config.get('features', {}).get('show_merge_suggestions', True):
        return redirect('/')

    threshold = float(request.args.get('threshold', 0.6))

    return render_template('suggest_merges.html', threshold=threshold)
