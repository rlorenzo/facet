"""
Database CLI for Facet.

Thin wrapper around the db package for command-line usage.

Usage:
    python database.py              # Initialize/upgrade schema
    python database.py --info       # Show schema information
    python database.py --migrate-tags
    python database.py --refresh-stats
    python database.py --stats-info
    python database.py --vacuum
    python database.py --analyze
    python database.py --optimize
"""

from db import (
    DEFAULT_DB_PATH,
    init_database,
    get_schema_info,
    get_photo_tags_count,
    get_stats_cache_info,
    refresh_stats_cache,
    migrate_tags_to_lookup,
    optimize_database,
    vacuum_database,
    analyze_database,
    cleanup_orphaned_persons,
    export_viewer_db,
)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Initialize Facet database schema'
    )
    parser.add_argument(
        '--db',
        default=DEFAULT_DB_PATH,
        help=f'Database path (default: {DEFAULT_DB_PATH})'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Display schema information'
    )
    parser.add_argument(
        '--migrate-tags',
        action='store_true',
        help='Populate photo_tags lookup table from tags column for fast queries'
    )
    parser.add_argument(
        '--refresh-stats',
        action='store_true',
        help='Refresh statistics cache for improved viewer performance'
    )
    parser.add_argument(
        '--stats-info',
        action='store_true',
        help='Show statistics cache info (age, freshness)'
    )
    parser.add_argument(
        '--vacuum',
        action='store_true',
        help='Reclaim space and defragment the database'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Update query planner statistics for better performance'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run VACUUM + ANALYZE for full database optimization'
    )
    parser.add_argument(
        '--cleanup-orphaned-persons',
        action='store_true',
        help='Delete persons with no assigned faces'
    )
    parser.add_argument(
        '--export-viewer-db',
        nargs='?',
        const='photo_scores_viewer.db',
        metavar='OUTPUT_PATH',
        help='Export lightweight viewer database (strips BLOBs, downsizes thumbnails)'
    )

    args = parser.parse_args()

    if args.stats_info:
        # Show stats cache status
        print("Statistics cache status:")
        cache_info = get_stats_cache_info(args.db)
        if not cache_info:
            print("  No cached statistics found. Run --refresh-stats to populate.")
        else:
            for key, info in cache_info.items():
                fresh_mark = "[fresh]" if info['fresh'] else "[stale]"
                print(f"  {key}: {info['age_human']} old {fresh_mark}")
    elif args.refresh_stats:
        # Refresh the stats cache
        refresh_stats_cache(args.db, verbose=True)
    elif args.info:
        info = get_schema_info()
        print(f"Photos table: {info['photos_columns']} columns")
        print(f"Faces table: {info['faces_columns']} columns")
        print(f"Persons table: {info['persons_columns']} columns")
        print(f"Photo tags table: {info['photo_tags_columns']} columns")
        print(f"Indexes: {info['indexes']}")
        print(f"\nPhotos columns: {', '.join(info['column_names'])}")
        # Show photo_tags status
        tag_count = get_photo_tags_count(args.db)
        print(f"\nPhoto tags lookup: {tag_count} entries")
        if tag_count == 0:
            print("  Run --migrate-tags to populate for faster tag queries")
        # Show stats cache status
        print("\nStatistics cache:")
        cache_info = get_stats_cache_info(args.db)
        if not cache_info:
            print("  No cached statistics. Run --refresh-stats to populate.")
        else:
            fresh_count = sum(1 for info in cache_info.values() if info['fresh'])
            print(f"  {len(cache_info)} cached stats ({fresh_count} fresh)")
    elif args.migrate_tags:
        migrate_tags_to_lookup(args.db)
    elif args.optimize:
        optimize_database(args.db, verbose=True)
    elif args.vacuum:
        vacuum_database(args.db, verbose=True)
    elif args.analyze:
        analyze_database(args.db, verbose=True)
    elif args.cleanup_orphaned_persons:
        cleanup_orphaned_persons(args.db, verbose=True)
    elif args.export_viewer_db:
        export_viewer_db(args.db, output_path=args.export_viewer_db, verbose=True)
    else:
        init_database(args.db)
        print(f"Database initialized: {args.db}")
        info = get_schema_info()
        print(f"  - photos: {info['photos_columns']} columns")
        print(f"  - faces: {info['faces_columns']} columns")
        print(f"  - persons: {info['persons_columns']} columns")
        print(f"  - photo_tags: {info['photo_tags_columns']} columns")
        print(f"  - {info['indexes']} indexes")
