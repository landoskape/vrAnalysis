# Import matplotlib and set backend before other imports
import matplotlib as mpl

mpl.use("Agg")

from flask import Flask, send_file, request, make_response, jsonify, render_template
import matplotlib.pyplot as plt
import io
import logging
import socket
from argparse import ArgumentParser

# Import your modules here
from place_field_viewer import PlaceFieldViewer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
# Add this line to disable Werkzeug's default logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)  # or logging.WARNING


def fast_mode():
    parser = ArgumentParser()
    parser.add_argument("--fast", default=False, action="store_true")
    args = parser.parse_args()
    return args.fast


# Initialize the viewer globally
viewer = PlaceFieldViewer(fast_mode())


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/print-roi-details")
def print_roi_details():
    mouse_idx = int(request.args.get("mouse_idx", 0))
    idx_target_ses = int(request.args.get("idx_target_ses", 0))
    min_percentile = float(request.args.get("min_percentile", 90))
    max_percentile = float(request.args.get("max_percentile", 100))
    roi_idx = int(request.args.get("roi_idx", 0))

    mouse_name = viewer.mouse_names[mouse_idx]
    true_roi_idx = viewer._gather_idxs(mouse_name, idx_target_ses, min_percentile, max_percentile)[roi_idx]
    statement = f"Mouse: {mouse_name}, Target Session: {idx_target_ses}, ROI Index: {true_roi_idx}"
    print(statement)
    return statement


@app.route("/init-data")
def init_data():
    return jsonify({"mouse_names": viewer.mouse_names.tolist()})


@app.route("/get-true-roi-idx")
def get_true_roi_idx():
    mouse_idx = int(request.args.get("mouse_idx", 0))
    roi_idx = int(request.args.get("roi_idx", 0))
    min_percentile = float(request.args.get("min_percentile", 90))
    max_percentile = float(request.args.get("max_percentile", 100))

    mouse_name = viewer.mouse_names[mouse_idx]
    idx_target_ses = int(request.args.get("idx_target_ses", 0))
    idxs = viewer._gather_idxs(mouse_name, idx_target_ses, min_percentile, max_percentile)
    if len(idxs) == 0:
        return jsonify({"true_roi_idx": -1, "total_rois": 0})

    roi_idx = roi_idx % len(idxs)  # Wrap around if too large
    true_roi_idx = idxs[roi_idx]

    return jsonify(
        {
            "true_roi_idx": int(true_roi_idx),
            "total_rois": len(idxs),
        }
    )


@app.route("/plot")
def plot():
    try:
        mouse_idx = int(request.args.get("mouse_idx", 0))
        roi_idx = int(request.args.get("roi_idx", 0))
        min_percentile = float(request.args.get("min_percentile", 90))
        max_percentile = float(request.args.get("max_percentile", 100))
        target_ses = int(request.args.get("target_ses", 0))
        dead_trials = int(request.args.get("dead_trials", 5))

        mouse_name = viewer.mouse_names[mouse_idx]
        fig = viewer.get_plot(
            mouse_name=mouse_name,
            roi_idx=roi_idx,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
            idx_target_ses=target_ses,
            dead_trials=dead_trials,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        plt.close(fig)

        response = make_response(send_file(buf, mimetype="image/png"))
        response.headers["Cache-Control"] = "no-cache"
        return response

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return f"Error generating plot: {str(e)}", 500


if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Interactive plot server running on http://{local_ip}:5000")
    app.run(host="0.0.0.0", port=5000)
