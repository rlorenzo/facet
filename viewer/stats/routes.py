import json
from flask import render_template, request, jsonify
from viewer.stats import stats_bp
from viewer.config import VIEWER_CONFIG, CORRELATION_X_AXES, CORRELATION_Y_METRICS, _get_stats_cached
from viewer.db_helpers import get_db_connection


@stats_bp.route('/stats')
def stats_page():
    return render_template('stats.html')


@stats_bp.route('/api/stats/gear')
def api_stats_gear():
    def compute():
        conn = get_db_connection()
        cur = conn.cursor()
        # Camera bodies
        cur.execute('''SELECT camera_model, COUNT(*) as cnt, ROUND(AVG(aggregate),2), ROUND(AVG(aesthetic),2)
                       FROM photos WHERE camera_model IS NOT NULL AND camera_model != ''
                       GROUP BY camera_model ORDER BY cnt DESC LIMIT 20''')
        cameras = [{'name': r[0], 'count': r[1], 'avg_aggregate': r[2], 'avg_aesthetic': r[3]} for r in cur.fetchall()]

        # Lenses
        cur.execute('''SELECT lens_model, COUNT(*) as cnt
                       FROM photos WHERE lens_model IS NOT NULL AND lens_model != ''
                       GROUP BY lens_model ORDER BY cnt DESC LIMIT 20''')
        lenses = [{'name': r[0], 'count': r[1]} for r in cur.fetchall()]

        # Combos
        cur.execute('''SELECT camera_model || ' + ' || lens_model as combo, COUNT(*) as cnt, ROUND(AVG(aggregate),2)
                       FROM photos WHERE camera_model IS NOT NULL AND camera_model != '' AND lens_model IS NOT NULL AND lens_model != ''
                       GROUP BY camera_model, lens_model ORDER BY cnt DESC LIMIT 20''')
        combos = [{'name': r[0], 'count': r[1], 'avg_aggregate': r[2]} for r in cur.fetchall()]

        # Category distribution
        cur.execute('''SELECT category, COUNT(*) as cnt
                       FROM photos WHERE category IS NOT NULL AND category != ''
                       GROUP BY category ORDER BY cnt DESC''')
        categories = [{'name': r[0], 'count': r[1]} for r in cur.fetchall()]

        conn.close()
        return {'cameras': cameras, 'lenses': lenses, 'combos': combos, 'categories': categories}

    return jsonify(_get_stats_cached('gear', compute))


@stats_bp.route('/api/stats/settings')
def api_stats_settings():
    def compute():
        conn = get_db_connection()
        cur = conn.cursor()

        # ISO distribution with buckets
        cur.execute('''SELECT
            CASE
                WHEN ISO <= 100 THEN '100'
                WHEN ISO <= 200 THEN '200'
                WHEN ISO <= 400 THEN '400'
                WHEN ISO <= 800 THEN '800'
                WHEN ISO <= 1600 THEN '1600'
                WHEN ISO <= 3200 THEN '3200'
                WHEN ISO <= 6400 THEN '6400'
                WHEN ISO <= 12800 THEN '12800'
                ELSE '25600+'
            END as iso_bucket,
            COUNT(*) as cnt,
            MIN(ISO) as sort_key
            FROM photos WHERE ISO IS NOT NULL AND ISO > 0
            GROUP BY iso_bucket ORDER BY sort_key''')
        iso = [{'label': r[0], 'count': r[1]} for r in cur.fetchall()]

        # Aperture usage
        cur.execute('''SELECT ROUND(f_stop, 1) as ap, COUNT(*) as cnt
                       FROM photos WHERE f_stop IS NOT NULL AND f_stop > 0
                       GROUP BY ap ORDER BY ap''')
        aperture = [{'value': r[0], 'count': r[1]} for r in cur.fetchall()]

        # Focal length distribution (prefer 35mm equivalent)
        cur.execute('''SELECT
            CASE
                WHEN COALESCE(focal_length_35mm, focal_length) < 20 THEN '<20mm'
                WHEN COALESCE(focal_length_35mm, focal_length) < 35 THEN '20-34mm'
                WHEN COALESCE(focal_length_35mm, focal_length) < 50 THEN '35-49mm'
                WHEN COALESCE(focal_length_35mm, focal_length) < 85 THEN '50-84mm'
                WHEN COALESCE(focal_length_35mm, focal_length) < 135 THEN '85-134mm'
                WHEN COALESCE(focal_length_35mm, focal_length) < 200 THEN '135-199mm'
                WHEN COALESCE(focal_length_35mm, focal_length) < 400 THEN '200-399mm'
                ELSE '400mm+'
            END as focal_bucket,
            COUNT(*) as cnt,
            MIN(COALESCE(focal_length_35mm, focal_length)) as sort_key
            FROM photos WHERE COALESCE(focal_length_35mm, focal_length) IS NOT NULL AND COALESCE(focal_length_35mm, focal_length) > 0
            GROUP BY focal_bucket ORDER BY sort_key''')
        focal = [{'label': r[0], 'count': r[1]} for r in cur.fetchall()]

        # Shutter speed distribution with SQL binning
        # Parse TEXT shutter_speed: "1/250" -> fraction, "0.5" -> float
        cur.execute('''SELECT
            CASE
                WHEN ss < 0.00025 THEN '1/8000-1/4000'
                WHEN ss < 0.0005  THEN '1/4000-1/2000'
                WHEN ss < 0.001   THEN '1/2000-1/1000'
                WHEN ss < 0.002   THEN '1/1000-1/500'
                WHEN ss < 0.004   THEN '1/500-1/250'
                WHEN ss < 0.008   THEN '1/250-1/125'
                WHEN ss < 0.0167  THEN '1/125-1/60'
                WHEN ss < 0.0333  THEN '1/60-1/30'
                WHEN ss < 0.25    THEN '1/30-1/4'
                ELSE '1/4s+'
            END as shutter_bucket,
            COUNT(*) as cnt,
            MIN(ss) as sort_key
            FROM (
                SELECT CASE
                    WHEN INSTR(shutter_speed, '/') > 0
                    THEN CAST(SUBSTR(shutter_speed, 1, INSTR(shutter_speed, '/') - 1) AS REAL)
                         / CAST(SUBSTR(shutter_speed, INSTR(shutter_speed, '/') + 1) AS REAL)
                    ELSE CAST(shutter_speed AS REAL)
                END as ss
                FROM photos
                WHERE shutter_speed IS NOT NULL AND shutter_speed != ''
            ) WHERE ss IS NOT NULL AND ss > 0
            GROUP BY shutter_bucket ORDER BY sort_key''')
        shutter = [{'label': r[0], 'count': r[1]} for r in cur.fetchall()]

        # Score distribution in 0.5-point buckets
        cur.execute('''SELECT ROUND(aggregate * 2) / 2.0 AS bucket, COUNT(*) as cnt
                       FROM photos WHERE aggregate IS NOT NULL
                       GROUP BY bucket ORDER BY bucket''')
        score_dist = [{'label': str(r[0]), 'count': r[1]} for r in cur.fetchall()]
        conn.close()

        return {'iso': iso, 'aperture': aperture, 'focal_length': focal, 'shutter_speed': shutter, 'score_distribution': score_dist}

    return jsonify(_get_stats_cached('settings', compute))


@stats_bp.route('/api/stats/timeline')
def api_stats_timeline():
    def compute():
        conn = get_db_connection()
        cur = conn.cursor()

        # Monthly
        cur.execute('''SELECT SUBSTR(date_taken, 1, 7) as month, COUNT(*) as cnt
                       FROM photos WHERE date_taken IS NOT NULL AND date_taken != ''
                       GROUP BY month ORDER BY month''')
        monthly = [{'month': r[0].replace(':', '-'), 'count': r[1]} for r in cur.fetchall()]

        # Yearly
        cur.execute('''SELECT SUBSTR(date_taken, 1, 4) as year, COUNT(*) as cnt
                       FROM photos WHERE date_taken IS NOT NULL AND date_taken != ''
                       GROUP BY year ORDER BY year''')
        yearly = [{'year': r[0], 'count': r[1]} for r in cur.fetchall()]

        # Heatmap (day of week x hour)
        cur.execute('''SELECT
            CAST(STRFTIME('%w', REPLACE(SUBSTR(date_taken,1,10),':','-')) AS INTEGER) as dow,
            CAST(SUBSTR(date_taken, 12, 2) AS INTEGER) as hour,
            COUNT(*) as cnt
            FROM photos WHERE date_taken IS NOT NULL AND LENGTH(date_taken) >= 13
            GROUP BY dow, hour''')
        heatmap = [{'day': r[0], 'hour': r[1], 'count': r[2]} for r in cur.fetchall()]

        # Top days
        cur.execute('''SELECT REPLACE(SUBSTR(date_taken, 1, 10), ':', '-') as day, COUNT(*) as cnt
                       FROM photos WHERE date_taken IS NOT NULL AND date_taken != ''
                       GROUP BY day ORDER BY cnt DESC LIMIT 10''')
        top_days = [{'date': r[0], 'count': r[1]} for r in cur.fetchall()]

        conn.close()
        return {'monthly': monthly, 'yearly': yearly, 'heatmap': heatmap, 'top_days': top_days}

    return jsonify(_get_stats_cached('timeline', compute))


@stats_bp.route('/api/stats/correlations')
def api_stats_correlations():
    # Validate parameters against whitelists
    x = request.args.get('x', 'iso')
    if x not in CORRELATION_X_AXES:
        return jsonify({'error': 'invalid x axis'}), 400

    y_raw = request.args.get('y', 'aggregate')
    y_metrics = [m for m in y_raw.split(',') if m in CORRELATION_Y_METRICS]
    if not y_metrics:
        return jsonify({'error': 'no valid metrics'}), 400

    group_by = request.args.get('group_by', '')
    if group_by and group_by not in CORRELATION_X_AXES:
        return jsonify({'error': 'invalid group_by'}), 400

    try:
        min_samples = max(1, int(request.args.get('min_samples', '3')))
    except ValueError:
        min_samples = 3

    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')

    cache_key = f"corr:{x}:{','.join(sorted(y_metrics))}:{group_by}:{min_samples}:{date_from}:{date_to}"

    def compute():
        conn = get_db_connection()
        cur = conn.cursor()
        x_def = CORRELATION_X_AXES[x]
        x_sql = x_def['sql']
        x_filter = x_def['filter']
        x_sort = x_def['sort']

        # Build date filter clause
        date_clauses = []
        date_params = []
        if date_from:
            date_clauses.append("date_taken >= ?")
            date_params.append(date_from.replace('-', ':'))
        if date_to:
            date_clauses.append("date_taken <= ?")
            date_params.append(date_to.replace('-', ':') + " 23:59:59")
        date_filter = (' AND '.join(date_clauses)) if date_clauses else '1=1'

        # Build metric AVG expressions
        metric_cols = ', '.join(f'ROUND(AVG({m}), 3)' for m in y_metrics)

        if group_by:
            g_def = CORRELATION_X_AXES[group_by]
            g_sql = g_def['sql']
            g_filter = g_def['filter']
            top_n = g_def['top_n']

            # Find top N groups by count
            cur.execute(f"SELECT {g_sql}, COUNT(*) as cnt FROM photos WHERE {g_filter} AND {x_filter} AND {date_filter} GROUP BY {g_sql} ORDER BY cnt DESC LIMIT ?", date_params + [top_n])
            top_groups = [r[0] for r in cur.fetchall()]
            if not top_groups:
                conn.close()
                return {'labels': [], 'groups': {}, 'metrics': y_metrics, 'x_axis': x, 'group_by': group_by}

            placeholders = ','.join('?' for _ in top_groups)
            sql = f"""SELECT {x_sql} AS x_bucket, {g_sql} AS group_val, {metric_cols}, COUNT(*) AS cnt
                      FROM photos
                      WHERE {x_filter} AND {g_filter} AND {date_filter} AND {g_sql} IN ({placeholders})
                      GROUP BY x_bucket, group_val
                      HAVING cnt >= ?
                      ORDER BY {x_sort}"""
            cur.execute(sql, date_params + top_groups + [min_samples])
            rows = cur.fetchall()
            conn.close()

            # Build ordered labels from all x_buckets
            seen = {}
            labels = []
            for r in rows:
                if r[0] not in seen:
                    seen[r[0]] = True
                    labels.append(str(r[0]))

            # Build groups dict: {group_name: {label: {metric: value, count: N}}}
            groups = {}
            for r in rows:
                lbl = str(r[0])
                grp = str(r[1])
                if grp not in groups:
                    groups[grp] = {}
                bucket = {}
                for i, m in enumerate(y_metrics):
                    bucket[m] = r[2 + i]
                bucket['count'] = r[2 + len(y_metrics)]
                groups[grp][lbl] = bucket

            return {'labels': labels, 'groups': groups, 'metrics': y_metrics, 'x_axis': x, 'group_by': group_by}
        else:
            sql = f"""SELECT {x_sql} AS x_bucket, {metric_cols}, COUNT(*) AS cnt
                      FROM photos
                      WHERE {x_filter} AND {date_filter}
                      GROUP BY x_bucket
                      HAVING cnt >= ?
                      ORDER BY {x_sort}"""
            cur.execute(sql, date_params + [min_samples])
            rows = cur.fetchall()
            conn.close()

            labels = [str(r[0]) for r in rows]
            metrics = {}
            for i, m in enumerate(y_metrics):
                metrics[m] = [r[1 + i] for r in rows]
            counts = [r[1 + len(y_metrics)] for r in rows]

            return {'labels': labels, 'metrics': metrics, 'counts': counts, 'x_axis': x, 'group_by': ''}

    return jsonify(_get_stats_cached(cache_key, compute))
