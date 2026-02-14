"""
Capture screenshots of the Facet viewer for README and documentation.

Usage:
    # 1. Start viewer with backup DB on a separate port
    DB_PATH=photo_scores_pro.db.backup.db PORT=65433 python viewer.py

    # 2. Run screenshot capture
    python docs/take_screenshots.py --port 65433 --password user --edition-password admin

    # Or specify all options:
    python docs/take_screenshots.py --port 65433 --password user --edition-password admin --output docs/screenshots

Prerequisites:
    pip install playwright
    playwright install chromium
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)


# CSS to blur all face/person avatar images for privacy
FACE_BLUR_CSS = """
img[src*="face_thumbnail"],
img[src*="person_thumbnail"],
.person-avatar,
img[src*="/person_thumbnail/"] {
    filter: blur(12px) !important;
    -webkit-filter: blur(12px) !important;
}
"""

# CSS to blur main photo thumbnails on cards that contain faces.
# Uses :has() to target only cards with person avatars or face-assign buttons,
# plus always blurs person/face thumbnail images.
SMART_BLUR_CSS = FACE_BLUR_CSS + """
.photo-card:has(img[src*="person_thumbnail"]) .photo-thumb,
.photo-card:has(.assign-face-btn) .photo-thumb {
    filter: blur(12px) !important;
    -webkit-filter: blur(12px) !important;
}
"""

# CSS to blur ALL thumbnails (for pages where any photo may contain faces)
ALL_THUMB_BLUR_CSS = FACE_BLUR_CSS + """
img[src*="/thumbnail"] {
    filter: blur(12px) !important;
    -webkit-filter: blur(12px) !important;
}
"""

DESKTOP_VIEWPORT = {"width": 1920, "height": 1080}
MOBILE_VIEWPORT = {"width": 390, "height": 844}


def login(page, base_url, password):
    """Authenticate with the viewer password."""
    page.goto(f"{base_url}/login")
    page.fill('input[name="password"]', password)
    page.click('button[type="submit"], input[type="submit"]')
    page.wait_for_url(f"{base_url}/**")


def edition_login(page, base_url, edition_password):
    """Authenticate for edition mode via API."""
    page.evaluate(f"""
        fetch('/api/edition/login', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{password: '{edition_password}'}})
        }})
    """)
    time.sleep(0.5)


def blur_face_cards_by_tooltip(page):
    """Use JS to blur thumbnails on cards whose tooltip mentions face metrics."""
    page.evaluate("""
        document.querySelectorAll('.photo-thumb-container[data-tooltip]').forEach(el => {
            if (el.dataset.tooltip && el.dataset.tooltip.includes('Face Quality:')) {
                const img = el.querySelector('.photo-thumb');
                if (img) img.style.filter = 'blur(12px)';
            }
        });
    """)


def inject_blur_css(page, css=FACE_BLUR_CSS):
    """Inject face-blurring CSS into the page."""
    page.add_style_tag(content=css)


def wait_for_photos(page, timeout=15000):
    """Wait for photo grid to load with images."""
    try:
        page.wait_for_selector('.photo-card', timeout=timeout)
        # Wait for at least some thumbnails to load
        page.wait_for_function(
            "document.querySelectorAll('.photo-thumb[src]').length > 3",
            timeout=timeout,
        )
        # Let images finish rendering
        time.sleep(1.5)
    except Exception:
        # Grid may be empty for some filters - that's fine
        time.sleep(1)


def wait_for_charts(page, timeout=15000):
    """Wait for Chart.js charts to render (lazy-loaded via fetch)."""
    try:
        # Wait for loading overlay to disappear (tabs trigger showLoading/hideLoading)
        page.wait_for_function(
            "document.getElementById('loading-overlay')?.classList.contains('hidden') !== false",
            timeout=timeout,
        )
        # Wait for at least one chart title to be populated (set after data renders)
        page.wait_for_function(
            "document.querySelectorAll('.tab-panel:not(.hidden) h3').length === 0 || "
            "[...document.querySelectorAll('.tab-panel:not(.hidden) h3')].some(h => h.textContent.trim().length > 0)",
            timeout=timeout,
        )
        # Give charts time to animate
        time.sleep(2)
    except Exception:
        time.sleep(4)


def screenshot(page, output_dir, filename, full_page=False):
    """Take a screenshot and save it. Uses JPEG for photo-heavy pages, PNG for charts."""
    path = os.path.join(output_dir, filename)
    if filename.endswith('.jpg'):
        # Playwright only saves PNG; capture as temp PNG then convert to JPEG
        from PIL import Image
        tmp_path = path.replace('.jpg', '.tmp.png')
        page.screenshot(path=tmp_path, full_page=full_page)
        img = Image.open(tmp_path).convert('RGB')
        img.save(path, 'JPEG', quality=85, optimize=True)
        os.remove(tmp_path)
    else:
        page.screenshot(path=path, full_page=full_page)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {filename} ({size_kb:.0f} KB)")


def capture_gallery_desktop(page, base_url, output_dir):
    """Hero shot: gallery with landscape/architecture photos, details visible."""
    print("\n[1/10] Gallery desktop (hero shot)...")
    page.set_viewport_size(DESKTOP_VIEWPORT)
    # Use landscape category to avoid faces in hero shot
    page.goto(f"{base_url}/?type=landscape&sort=aesthetic&dir=DESC&hide_details=0&hide_blinks=1&hide_bursts=1")
    page.wait_for_load_state("networkidle")
    wait_for_photos(page)
    inject_blur_css(page)
    screenshot(page, output_dir, "gallery-desktop.jpg")


def capture_gallery_mobile(page, base_url, output_dir):
    """Mobile responsive layout."""
    print("\n[2/10] Gallery mobile...")
    page.set_viewport_size(MOBILE_VIEWPORT)
    page.goto(f"{base_url}/?type=landscape&sort=aesthetic&dir=DESC&hide_blinks=1&hide_bursts=1")
    page.wait_for_load_state("networkidle")
    wait_for_photos(page)
    inject_blur_css(page)
    screenshot(page, output_dir, "gallery-mobile.jpg")
    # Restore desktop viewport
    page.set_viewport_size(DESKTOP_VIEWPORT)


def capture_gallery_compact(page, base_url, output_dir):
    """Compact grid mode with details hidden."""
    print("\n[3/10] Gallery compact (details hidden)...")
    page.set_viewport_size(DESKTOP_VIEWPORT)
    page.goto(f"{base_url}/?hide_details=1&sort=aesthetic&dir=DESC&hide_blinks=1&hide_bursts=1")
    page.wait_for_load_state("networkidle")
    wait_for_photos(page)
    inject_blur_css(page)
    screenshot(page, output_dir, "gallery-compact.jpg")


def capture_filter_drawer(page, base_url, output_dir, edition_password):
    """Filter drawer open with all sections expanded."""
    print("\n[4/10] Filter drawer...")
    page.set_viewport_size(DESKTOP_VIEWPORT)
    edition_login(page, base_url, edition_password)
    # Use landscape category to minimize faces, blur ALL thumbnails since the
    # drawer UI is the focus — background photos are secondary
    page.goto(f"{base_url}/?type=landscape&sort=aggregate&dir=DESC&hide_blinks=1&hide_bursts=1")
    page.wait_for_load_state("networkidle")
    wait_for_photos(page)
    inject_blur_css(page, ALL_THUMB_BLUR_CSS)

    # Click the filter toggle button to open drawer
    page.click('button:has(svg path[d*="M3 4a1"])')
    time.sleep(0.5)

    # Wait for drawer to slide in
    page.wait_for_selector('#filter-drawer:not(.-translate-x-full)', timeout=3000)
    time.sleep(0.3)

    # Expand all <details> sections in the drawer
    page.evaluate("""
        document.querySelectorAll('#filter-drawer details').forEach(d => d.open = true);
    """)
    time.sleep(0.3)

    screenshot(page, output_dir, "filter-drawer.jpg")

    # Close the drawer
    page.click('#filter-drawer button:has-text("×")')
    time.sleep(0.3)


def capture_hover_preview(page, base_url, output_dir):
    """Gallery with hover preview visible on a photo."""
    print("\n[5/10] Hover preview...")
    page.set_viewport_size(DESKTOP_VIEWPORT)
    page.goto(f"{base_url}/?type=landscape&sort=aesthetic&dir=DESC&hide_blinks=1&hide_bursts=1")
    page.wait_for_load_state("networkidle")
    wait_for_photos(page)
    inject_blur_css(page)

    # Hover over the first photo card's thumbnail container
    first_card = page.query_selector('.photo-thumb-container')
    if first_card:
        first_card.hover()
        time.sleep(1)  # Wait for preview to appear

    screenshot(page, output_dir, "hover-preview.jpg")


def capture_stats_dashboard(page, base_url, output_dir):
    """Statistics page — capture all 4 tabs."""
    tabs = [
        ("gear", "stats-gear.png"),
        ("settings", "stats-settings.png"),
        ("timeline", "stats-timeline.png"),
        ("correlations", "stats-correlations.png"),
    ]
    page.set_viewport_size(DESKTOP_VIEWPORT)
    page.goto(f"{base_url}/stats")
    page.wait_for_load_state("networkidle")

    # Title element IDs that get populated after each tab's data loads
    tab_title_ids = {
        "gear": "title-cameras",
        "settings": "title-iso",
        "timeline": "title-monthly",
        "correlations": "corr-status",
    }

    for tab_id, filename in tabs:
        print(f"\n  Stats tab: {tab_id}...")
        page.click(f'button[data-tab="{tab_id}"]')
        # Wait for the tab's data to load by checking its title element
        title_id = tab_title_ids[tab_id]
        try:
            page.wait_for_function(
                f"document.getElementById('{title_id}')?.textContent?.trim().length > 0",
                timeout=30000,
            )
        except Exception:
            pass
        # Extra time for chart animation
        time.sleep(3)
        inject_blur_css(page)
        screenshot(page, output_dir, filename, full_page=True)

    # Also save the gear tab as the default stats-dashboard.png for the README
    page.click('button[data-tab="gear"]')
    time.sleep(0.5)
    wait_for_charts(page)
    inject_blur_css(page)
    screenshot(page, output_dir, "stats-dashboard.png")


def capture_manage_persons(page, base_url, output_dir, edition_password):
    """Person management grid (requires edition auth)."""
    print("\n[7/10] Manage persons...")
    page.set_viewport_size(DESKTOP_VIEWPORT)
    edition_login(page, base_url, edition_password)
    page.goto(f"{base_url}/manage_persons")
    page.wait_for_load_state("networkidle")
    try:
        page.wait_for_selector('.person-card, [class*="person"], .grid', timeout=10000)
    except Exception:
        pass
    time.sleep(2)
    # Blur all person thumbnails
    inject_blur_css(page, ALL_THUMB_BLUR_CSS)
    screenshot(page, output_dir, "manage-persons.jpg")


def capture_compare(page, base_url, output_dir, edition_password):
    """Pairwise comparison page (requires edition auth)."""
    print("\n[8/10] Compare page...")
    page.set_viewport_size(DESKTOP_VIEWPORT)
    edition_login(page, base_url, edition_password)
    page.goto(f"{base_url}/compare?type=landscape")
    page.wait_for_load_state("networkidle")
    # Wait for comparison images to load
    try:
        page.wait_for_function(
            """document.querySelector('#img-a')?.src && document.querySelector('#img-a').naturalWidth > 0""",
            timeout=10000,
        )
        time.sleep(2)
    except Exception:
        time.sleep(3)
    inject_blur_css(page, ALL_THUMB_BLUR_CSS)
    screenshot(page, output_dir, "compare.jpg")


def capture_person_gallery(page, base_url, output_dir):
    """Person-specific gallery (person filter active)."""
    print("\n[9/10] Person gallery...")
    page.set_viewport_size(DESKTOP_VIEWPORT)
    # Get first person ID from the API
    response = page.evaluate("""
        fetch('/api/filter_options/persons')
            .then(r => r.json())
            .then(data => data && data.length > 0 ? data[0][0] : null)
    """)
    if response:
        page.goto(f"{base_url}/?person={response}&sort=aggregate&dir=DESC")
        page.wait_for_load_state("networkidle")
        wait_for_photos(page)
    else:
        # Fallback: just show gallery with a face filter
        page.goto(f"{base_url}/?type=portrait&sort=face_quality&dir=DESC&hide_blinks=1&hide_bursts=1")
        page.wait_for_load_state("networkidle")
        wait_for_photos(page)
    inject_blur_css(page, ALL_THUMB_BLUR_CSS)
    screenshot(page, output_dir, "person-gallery.jpg")


def capture_gallery_top_picks(page, base_url, output_dir, edition_password):
    """Top Picks filter active."""
    print("\n[10/10] Gallery top picks...")
    page.set_viewport_size(DESKTOP_VIEWPORT)
    edition_login(page, base_url, edition_password)
    # max_face_count=0 excludes all photos with detected faces for privacy
    page.goto(f"{base_url}/?type=top_picks&hide_blinks=1&hide_bursts=1&max_face_count=0")
    page.wait_for_load_state("networkidle")
    wait_for_photos(page)
    inject_blur_css(page, FACE_BLUR_CSS)
    screenshot(page, output_dir, "gallery-top-picks.jpg")


def main():
    parser = argparse.ArgumentParser(description="Capture Facet viewer screenshots")
    parser.add_argument("--port", type=int, default=65433, help="Viewer port (default: 65433)")
    parser.add_argument("--password", default="user", help="Viewer password")
    parser.add_argument("--edition-password", default="admin", help="Edition mode password")
    parser.add_argument("--output", default=None, help="Output directory (default: docs/screenshots)")
    parser.add_argument("--only", default=None, help="Capture only specific screenshot (e.g., 'gallery-desktop')")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    output_dir = args.output or os.path.join(os.path.dirname(__file__), "screenshots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Capturing screenshots from {base_url}")
    print(f"Output: {output_dir}")

    # Map screenshot names to capture functions
    captures = {
        "gallery-desktop": lambda p: capture_gallery_desktop(p, base_url, output_dir),
        "gallery-mobile": lambda p: capture_gallery_mobile(p, base_url, output_dir),
        "gallery-compact": lambda p: capture_gallery_compact(p, base_url, output_dir),
        "filter-drawer": lambda p: capture_filter_drawer(p, base_url, output_dir, args.edition_password),
        "hover-preview": lambda p: capture_hover_preview(p, base_url, output_dir),
        "stats-dashboard": lambda p: capture_stats_dashboard(p, base_url, output_dir),
        "manage-persons": lambda p: capture_manage_persons(p, base_url, output_dir, args.edition_password),
        "compare": lambda p: capture_compare(p, base_url, output_dir, args.edition_password),
        "person-gallery": lambda p: capture_person_gallery(p, base_url, output_dir),
        "gallery-top-picks": lambda p: capture_gallery_top_picks(p, base_url, output_dir, args.edition_password),
    }

    if args.only:
        if args.only not in captures:
            print(f"Error: unknown screenshot '{args.only}'. Available: {', '.join(captures.keys())}")
            sys.exit(1)
        captures = {args.only: captures[args.only]}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=DESKTOP_VIEWPORT,
            device_scale_factor=2,  # Retina-quality screenshots
        )
        page = context.new_page()

        # Login
        print("Logging in...")
        login(page, base_url, args.password)
        print("Authenticated.")

        # Capture each screenshot
        for name, capture_fn in captures.items():
            try:
                capture_fn(page)
            except Exception as e:
                print(f"  ERROR capturing {name}: {e}")

        browser.close()

    # Summary
    print(f"\nDone! {len(captures)} screenshots saved to {output_dir}/")
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".png")
    )
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
