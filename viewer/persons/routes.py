from flask import render_template, request, redirect, jsonify
from viewer.persons import persons_bp
from viewer.config import VIEWER_CONFIG, load_viewer_config, invalidate_filter_cache
from viewer.auth import is_edition_enabled, is_edition_authenticated, require_edition
from viewer.db_helpers import get_db_connection


@persons_bp.route('/rename_person/<int:person_id>', methods=['POST'])
@require_edition
def rename_person(person_id):
    """Rename a person (set or update their name)."""
    name = request.form.get('name', '').strip()
    conn = get_db_connection()
    conn.execute("UPDATE persons SET name = ? WHERE id = ?", (name or None, person_id))
    conn.commit()
    conn.close()
    invalidate_filter_cache()
    # Return JSON for AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return {'success': True, 'name': name or f'Person {person_id}'}
    return redirect(request.referrer or '/')



@persons_bp.route('/manage_persons')
def manage_persons_page():
    """Display a page to manage persons (merge or delete)."""
    if not is_edition_authenticated():
        return redirect('/')
    sort_by = request.args.get('sort', 'count_desc')

    conn = get_db_connection()
    try:
        if sort_by == 'count_asc':
            order_clause = "ORDER BY p.face_count ASC, p.id"
        elif sort_by == 'quality_asc':
            order_clause = "ORDER BY rep_quality ASC, p.id"
        elif sort_by == 'quality_desc':
            order_clause = "ORDER BY rep_quality DESC, p.id"
        else:  # count_desc (default)
            order_clause = "ORDER BY p.face_count DESC, p.id"

        # Normalize eye_sharpness (0-10) and face_quality (6.5-9.5) to 0-1 scale
        # Weight eye_sharpness 70% since it's the primary blur indicator
        persons = conn.execute(f"""
            SELECT p.id, p.name, p.representative_face_id, p.face_count,
                   (COALESCE(photos.eye_sharpness, 0) / 10.0 * 0.7 +
                    (COALESCE(photos.face_quality, 6.5) - 6.5) / 3.0 * 0.3) as rep_quality
            FROM persons p
            LEFT JOIN faces f ON p.representative_face_id = f.id
            LEFT JOIN photos ON f.photo_path = photos.path
            {order_clause}
        """).fetchall()
        persons = [dict(row) for row in persons]
    finally:
        conn.close()

    return render_template('manage_persons.html', persons=persons, sort_by=sort_by, editing_enabled=is_edition_enabled(), edition_authenticated=is_edition_authenticated(), viewer_config=load_viewer_config())


@persons_bp.route('/merge_persons/<int:source_id>/<int:target_id>', methods=['POST'])
@require_edition
def merge_persons(source_id, target_id):
    """Merge source person into target person."""
    if source_id == target_id:
        return jsonify({'error': 'Cannot merge a person into itself'}), 400

    conn = get_db_connection()
    try:
        # 1. Move all faces from source to target
        conn.execute("UPDATE faces SET person_id = ? WHERE person_id = ?",
                     (target_id, source_id))

        # 2. Update target face_count
        count = conn.execute("SELECT COUNT(*) FROM faces WHERE person_id = ?",
                            (target_id,)).fetchone()[0]
        conn.execute("UPDATE persons SET face_count = ? WHERE id = ?",
                     (count, target_id))

        # 3. Delete source person
        conn.execute("DELETE FROM persons WHERE id = ?", (source_id,))

        conn.commit()
        invalidate_filter_cache()
        return jsonify({'success': True, 'new_count': count})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@persons_bp.route('/merge_persons_batch', methods=['POST'])
@require_edition
def merge_persons_batch():
    """Merge multiple persons into a target person."""
    data = request.get_json()
    source_ids = data.get('source_ids', [])
    target_id = data.get('target_id')

    if not source_ids or not target_id:
        return jsonify({'success': False, 'error': 'Missing source_ids or target_id'}), 400
    if target_id in source_ids:
        return jsonify({'success': False, 'error': 'Target cannot be in source list'}), 400

    conn = get_db_connection()
    try:
        # Move all faces from sources to target
        placeholders = ','.join('?' * len(source_ids))
        conn.execute(f'UPDATE faces SET person_id = ? WHERE person_id IN ({placeholders})',
                     [target_id] + source_ids)

        # Update target face_count
        new_count = conn.execute('SELECT COUNT(*) FROM faces WHERE person_id = ?',
                                 (target_id,)).fetchone()[0]
        conn.execute('UPDATE persons SET face_count = ? WHERE id = ?', (new_count, target_id))

        # Delete source persons
        conn.execute(f'DELETE FROM persons WHERE id IN ({placeholders})', source_ids)
        conn.commit()

        invalidate_filter_cache()
        return jsonify({'success': True, 'target_id': target_id, 'merged_count': len(source_ids), 'new_count': new_count})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@persons_bp.route('/delete_person/<int:person_id>', methods=['POST'])
@require_edition
def delete_person(person_id):
    """Delete a person and unassign all their faces."""
    conn = get_db_connection()
    try:
        # 1. Unassign all faces from this person (set person_id to NULL)
        conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (person_id,))

        # 2. Delete the person
        conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))

        conn.commit()
        invalidate_filter_cache()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@persons_bp.route('/delete_persons_batch', methods=['POST'])
@require_edition
def delete_persons_batch():
    """Delete multiple persons and unassign all their faces."""
    data = request.get_json()
    person_ids = data.get('person_ids', [])

    if not person_ids:
        return jsonify({'success': False, 'error': 'No person_ids provided'}), 400

    conn = get_db_connection()
    try:
        placeholders = ','.join('?' * len(person_ids))
        # 1. Unassign all faces from these persons
        conn.execute(f"UPDATE faces SET person_id = NULL WHERE person_id IN ({placeholders})", person_ids)

        # 2. Delete the persons
        conn.execute(f"DELETE FROM persons WHERE id IN ({placeholders})", person_ids)

        conn.commit()
        invalidate_filter_cache()
        return jsonify({'success': True, 'deleted_count': len(person_ids)})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()
