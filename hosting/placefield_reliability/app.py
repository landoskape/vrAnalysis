# Import matplotlib and set backend before other imports
import matplotlib as mpl

mpl.use("Agg")

from flask import Flask, send_file, request, make_response, jsonify, render_template
import matplotlib.pyplot as plt
import io
import logging
from .reliability_viewer import ReliabilityViewer


def create_app(fast_mode=False):
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # Initialize the viewer globally
    viewer = ReliabilityViewer(fast_mode)

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/init-data")
    def init_data():
        return jsonify({"mouse_names": viewer.mouse_names.tolist()})

    @app.route("/plot")
    def plot():
        try:
            mouse_idx = int(request.args.get("mouse_idx", 0))
            use_relcor = request.args.get("use_relcor", "true").lower() == "true"
            tracked = request.args.get("tracked", "true").lower() == "true"
            average = request.args.get("average", "true").lower() == "true"
            min_session = request.args.get("min_session")
            max_session = request.args.get("max_session")

            # Convert session limits to integers if they're provided
            min_session = int(min_session) if min_session is not None else None
            max_session = int(max_session) if max_session is not None else None

            mouse_name = viewer.mouse_names[mouse_idx]
            fig = viewer.get_plot(
                mouse_name=mouse_name, use_relcor=use_relcor, tracked=tracked, average=average, min_session=min_session, max_session=max_session
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

    @app.route("/session-range")
    def session_range():
        try:
            mouse_idx = int(request.args.get("mouse_idx", 0))
            mouse_name = viewer.mouse_names[mouse_idx]
            min_session = min(viewer.rel_idx_ses[mouse_name])
            max_session = max(viewer.rel_idx_ses[mouse_name])
            return jsonify({"min_session": min_session, "max_session": max_session})
        except Exception as e:
            app.logger.error(f"Error: {str(e)}")
            return f"Error getting session range: {str(e)}", 500

    return app
