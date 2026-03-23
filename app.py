# =====================================================
# APP.PY (FLASK)
# =====================================================

from flask import Flask, render_template, request
from model import recommend

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    error = ""

    if request.method == "POST":
        song_name = request.form["song"]

        recs = recommend(song_name)

        if recs:
            results = recs
        else:
            error = "Song not found!"

    return render_template("index.html", results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True)