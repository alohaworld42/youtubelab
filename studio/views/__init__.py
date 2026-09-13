"""Flask blueprints for the studio UI."""

VIDEO_TYPES = [
    {"id": "reddit", "name": "Reddit Story", "emoji": "📖", "blurb": "AITA / confession drama"},
    {"id": "conversation", "name": "Two Characters", "emoji": "🗣️", "blurb": "Banter that teaches something"},
    {"id": "facts", "name": "Fun Facts", "emoji": "🤯", "blurb": "Rapid-fire 'did you know'"},
    {"id": "scary", "name": "Scary Story", "emoji": "👻", "blurb": "Creepy slow-burn with a twist"},
    # Blurbs describe OUR look, never a protected one. "CoComelon-style" used to
    # stand here: pointing the operator (and, via prompts/kids.txt, the LLM) at a
    # brand this channel treats look-alike drift toward as ship-blocking.
    {"id": "kids", "name": "Kids Song", "emoji": "🎈", "blurb": "Original nursery-rhyme sing-along"},
    {"id": "kidsong", "name": "Kids Song (3D)", "emoji": "🎬", "blurb": "Sing-along with GPU-rendered 3D scenes"},
    {"id": "hyperframes", "name": "Hyperframes Erklärvideo", "emoji": "🧩",
     "blurb": "One concept, hook to takeaway, clearly structured"},
    # NOTE: this is the "brainrot" VIDEO TYPE (a content format: hooky, meme-driven
    # shorts on varied topics) — not the "brainrot" STYLE below (a look-and-sound
    # preset). Same word, two different columns; see studio/ideas.py for why the
    # two must never be compared to each other.
    {"id": "brainrot", "name": "Brainrot Short", "emoji": "🌀", "blurb": "Hooky meme-format short, monetization-safe"},
]

STYLES = ["", "kids", "brainrot", "clean"]

VIDEO_TYPE_IDS = {t["id"] for t in VIDEO_TYPES}


def register_blueprints(app):
    from studio.views.api import bp as api_bp
    from studio.views.channels import bp as channels_bp
    from studio.views.dashboard import bp as dashboard_bp
    from studio.views.ideas import bp as ideas_bp
    from studio.views.review import bp as review_bp
    from studio.views.setup import bp as setup_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(channels_bp)
    app.register_blueprint(ideas_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(setup_bp)
