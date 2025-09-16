from flask import Flask, render_template, request

from .logic import build_delta, match_and_classify, parse_new, parse_prev

app = Flask(__name__)

INPUT_LIMITS = {"prev": 4000, "transcript": 8000}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        delta=None,
        prev_text="",
        transcript="",
        error=None,
        banner="Don't paste PII/regulated data.",
    )


@app.route("/delta", methods=["POST"])
def delta_view():
    prev_text = request.form.get("prev_actions", "")
    transcript = request.form.get("transcript", "")
    error = None

    if len(prev_text) > INPUT_LIMITS["prev"]:
        error = "Previous actions input is too long (max 4,000 characters)."
    elif len(transcript) > INPUT_LIMITS["transcript"]:
        error = "Transcript is too long (max 8,000 characters)."

    delta = None
    if not error:
        prev_data = parse_prev(prev_text)
        new_data = parse_new(transcript)
        delta_raw = match_and_classify(prev_data.get("actions", []), new_data, prev_data.get("decisions", []))
        delta = build_delta(delta_raw)

    return render_template(
        "index.html",
        delta=delta,
        prev_text=prev_text,
        transcript=transcript,
        error=error,
        banner="Don't paste PII/regulated data.",
    )


if __name__ == "__main__":
    app.run(debug=True)
