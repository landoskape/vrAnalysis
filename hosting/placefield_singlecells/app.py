# Import matplotlib and set backend before other imports
import matplotlib as mpl

mpl.use("Agg")

from flask import Flask, send_file, request, make_response, jsonify, render_template
import matplotlib.pyplot as plt
import io
import logging
from .place_field_viewer import PlaceFieldViewer


def create_app(fast_mode=False):
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # Initialize the viewer globally
    viewer = PlaceFieldViewer(fast_mode)

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

    @app.route("/get-sessions")
    def get_sessions():
        mouse_idx = int(request.args.get("mouse_idx", 0))
        mouse_name = viewer.mouse_names[mouse_idx]
        return jsonify({"num_sessions": viewer.ses_per_mouse[mouse_name]})

    @app.route("/init-data")
    def init_data():
        return jsonify(
            {
                "mouse_names": viewer.mouse_names,
                "num_sessions": viewer.ses_per_mouse[viewer.mouse_names[0]],  # Initial sessions for first mouse
            }
        )

    @app.route("/get-true-roi-idx")
    def get_true_roi_idx():
        mouse_idx = int(request.args.get("mouse_idx", 0))
        roi_idx = int(request.args.get("roi_idx", 0))
        min_percentile = float(request.args.get("min_percentile", 90))
        max_percentile = float(request.args.get("max_percentile", 100))

        mouse_name = viewer.mouse_names[mouse_idx]
        idx_target_ses = int(request.args.get("idx_target_ses", 0))
        red_cells = request.args.get("red_cells", "true").lower() == "true"
        idxs = viewer._gather_idxs(mouse_name, idx_target_ses, min_percentile, max_percentile, red_cells)
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
            red_cells = request.args.get("red_cells", "true").lower() == "true"

            mouse_name = viewer.mouse_names[mouse_idx]
            fig = viewer.get_plot(
                mouse_name=mouse_name,
                roi_idx=roi_idx,
                min_percentile=min_percentile,
                max_percentile=max_percentile,
                idx_target_ses=target_ses,
                dead_trials=dead_trials,
                red_cells=red_cells,
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

    return app
