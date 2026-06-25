"""X11 active-window geometry for focus adverting (CLAUDE.md §8.5).

Thin python-xlib helpers used by ``ScreenVisionSource`` to point the attention
focus at the currently focused window. Import is lazy and guarded by callers, so
the package works on headless / non-X11 systems. Not exercised in unit tests
(needs a live display).
"""

from __future__ import annotations


def _display():  # pragma: no cover - needs a display
    from Xlib import display

    return display.Display()


def screen_size() -> tuple[int, int]:  # pragma: no cover - needs a display
    """(width, height) of the default screen in pixels."""
    d = _display()
    screen = d.screen()
    return int(screen.width_in_pixels), int(screen.height_in_pixels)


def active_window_rect() -> tuple[int, int, int, int] | None:  # pragma: no cover - needs a display
    """(x, y, w, h) of the focused window in screen pixels, or None."""
    from Xlib import X

    d = _display()
    root = d.screen().root
    net_active = d.intern_atom("_NET_ACTIVE_WINDOW")
    prop = root.get_full_property(net_active, X.AnyPropertyType)
    if not prop or not prop.value:
        return None
    win = d.create_resource_object("window", prop.value[0])
    try:
        geom = win.get_geometry()
        coords = win.translate_coords(root, 0, 0)
    except Exception:
        # Window was destroyed between the property read and the geometry query.
        return None
    return (-coords.x, -coords.y, int(geom.width), int(geom.height))
