"""Turn machine numbers into something a person reads at a glance.

The take-review queue rendered a take's age as ``wartet 18472s``. Nobody
converts that in their head — and the number is the whole reason the row is
there, because a gate that has been waiting five hours is the one blocking a
render. A reviewer scanning the queue on a phone needs "vor 5 h" to land
without arithmetic.

German, because the studio UI is German. Deliberately coarse: a duration is a
"how stale is this" signal, not a stopwatch, so it never shows more than two
units. Each unit truncates rather than rounds — rounding the smaller unit can
carry ("5 h 60 min"), and for a staleness read "at least this long" is the
honest direction to err in anyway.
"""

_MINUTE = 60
_HOUR = 60 * _MINUTE
_DAY = 24 * _HOUR


def human_duration(seconds):
    """``18472`` -> ``"5 h 7 min"``. A short, two-unit-at-most duration.

    Returns ``""`` for a value that is missing or not a number, so a template
    can render it unconditionally — a broken timestamp must not blank the whole
    queue page it appears on. Negative values (a clock that stepped backwards
    between the file's mtime and now) are treated as "just now" rather than
    rendered with a minus sign.
    """
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return ""
    if total != total or total in (float("inf"), float("-inf")):  # NaN / inf
        return ""
    total = int(max(0, total))

    if total < _MINUTE:
        return f"{total} s"
    if total < _HOUR:
        minutes, secs = divmod(total, _MINUTE)
        return f"{minutes} min {secs} s" if secs else f"{minutes} min"
    if total < _DAY:
        hours, rest = divmod(total, _HOUR)
        minutes = rest // _MINUTE
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    days, rest = divmod(total, _DAY)
    hours = rest // _HOUR
    return f"{days} T {hours} h" if hours else f"{days} T"


def install_filters(app):
    """Register the humanizers as Jinja filters (``{{ x|duration }}``)."""
    app.jinja_env.filters["duration"] = human_duration
