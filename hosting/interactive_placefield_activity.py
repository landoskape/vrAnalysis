# Import matplotlib and set backend before other imports
import matplotlib as mpl

mpl.use("Agg")

from flask import Flask, send_file, request, make_response, jsonify
import matplotlib.pyplot as plt
import io
import numpy as np
import logging
import socket
import sys
import os
from tqdm import tqdm
from argparse import ArgumentParser

# Import your modules here
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from vrAnalysis import analysis, database, tracking

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
# Add this line to disable Werkzeug's default logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)  # or logging.WARNING


def fast_mode():
    parser = ArgumentParser()
    parser.add_argument("--fast", default=False, action="store_true")
    args = parser.parse_args()
    return args.fast


class PlaceFieldViewer:
    def __init__(self, fast_mode=False):
        mousedb = database.vrDatabase("vrMice")
        df = mousedb.getTable(trackerExists=True)
        self.mouse_names = df["mouseName"].unique()
        if fast_mode:
            self.mouse_names = self.mouse_names[:2]
        print(self.mouse_names)
        self.env_selection = {}
        self.idx_ses_selection = {}
        self.spkmaps = {}
        self.extras = {}

        for mouse_name in tqdm(self.mouse_names, desc="Preparing mouse data", leave=True):
            self._prepare_mouse_data(mouse_name, fast_mode)

    def _prepare_mouse_data(self, mouse_name, fast_mode):
        keep_planes = [1] if fast_mode else [1, 2]
        track = tracking.tracker(mouse_name)  # get tracker object for mouse
        pcm = analysis.placeCellMultiSession(track, autoload=False, keep_planes=keep_planes)
        envnum, idx_ses = pcm.env_idx_ses_selector(envmethod="second", sesmethod=4)
        spkmaps, extras = pcm.get_spkmaps(envnum=envnum, idx_ses=idx_ses, trials="full", average=False, tracked=True)

        self.env_selection[mouse_name] = (envnum, idx_ses)
        self.idx_ses_selection[mouse_name] = idx_ses
        self.spkmaps[mouse_name] = spkmaps
        self.extras[mouse_name] = extras

    def _make_roi_trajectory(self, spkmaps, roi_idx, dead_trials=None):
        if dead_trials is None:
            dead_trials = 5
        roi_activity = [s[roi_idx] for s in spkmaps]
        dead_space = [np.full((dead_trials, roi_activity[0].shape[1]), np.nan) for _ in range(len(roi_activity) - 1)]
        dead_space.append(None)
        interleaved = [item for pair in zip(roi_activity, dead_space) for item in pair if item is not None]

        trial_env = [ises * np.ones(r.shape[0]) for ises, r in enumerate(roi_activity)]
        dead_trial_env = [np.nan * np.ones(dead_trials) for _ in range(len(roi_activity) - 1)]
        dead_trial_env.append(None)
        env_trialnum = [item for pair in zip(trial_env, dead_trial_env) for item in pair if item is not None]
        return np.concatenate(interleaved, axis=0), np.concatenate(env_trialnum)

    def _gather_idxs(self, mouse_name, idx_target_ses, min_percentile=90, max_percentile=100):
        all_values = np.concatenate(self.extras[mouse_name]["relcor"])
        reliability_values = self.extras[mouse_name]["relcor"][idx_target_ses]
        min_threshold = np.percentile(all_values, min_percentile)
        max_threshold = np.percentile(all_values, max_percentile)
        return np.where((reliability_values > min_threshold) & (reliability_values < max_threshold))[0]

    def _com(self, data, axis=-1):
        x = np.arange(data.shape[axis])
        com = np.sum(data * x, axis=axis) / (np.sum(data, axis=axis) + 1e-10)
        com[np.any(data < 0, axis=axis)] = np.nan
        return com

    def get_plot(self, mouse_name, roi_idx, min_percentile=90, max_percentile=100, idx_target_ses=0, dead_trials=5):
        spkmaps = self.spkmaps[mouse_name]
        idxs = self._gather_idxs(mouse_name, idx_target_ses, min_percentile, max_percentile)

        if len(idxs) == 0:
            # Create an empty figure with a message
            fig = plt.figure(figsize=(12, 8))
            plt.text(
                0.5,
                0.5,
                f"No ROIs found between {min_percentile}th and {max_percentile}th percentiles",
                horizontalalignment="center",
                verticalalignment="center",
                transform=fig.transFigure,
                fontsize=14,
            )
            return fig

        # Ensure roi_idx doesn't exceed available ROIs
        roi_idx = roi_idx % len(idxs)  # Wrap around if too large
        idx_roi_to_plot = idxs[roi_idx]

        roi_trajectory, env_trialnum = self._make_roi_trajectory(spkmaps, idx_roi_to_plot, dead_trials=dead_trials)

        idx_not_nan = ~np.any(np.isnan(roi_trajectory), axis=1)
        pfmax = np.where(idx_not_nan, np.max(roi_trajectory, axis=1), np.nan)
        pfcom = np.where(idx_not_nan, self._com(roi_trajectory, axis=1), np.nan)
        pfloc = np.where(idx_not_nan, np.argmax(roi_trajectory, axis=1), np.nan)

        cmap = mpl.colormaps["gray_r"]
        cmap.set_bad((1, 0.8, 0.8))  # Light red color
        ses_col = plt.cm.Set1(np.linspace(0, 1, len(self.idx_ses_selection[mouse_name])))

        fig = plt.figure(1, figsize=(10, 8))
        fig.clf()

        ax = fig.add_subplot(131)
        ax.cla()
        ax.imshow(roi_trajectory, aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=10)

        # Add vertical bar at x==0 to show which environment is target
        idx_trials_target = np.where(env_trialnum == idx_target_ses)[0]
        if np.any(idx_trials_target):  # Check if there are any target trials
            min_y = np.nanmin(idx_trials_target)
            max_y = np.nanmax(idx_trials_target)
            ax.plot([1, 1], [min_y, max_y], color=ses_col[idx_target_ses], linestyle="-", linewidth=5)
            ax.plot(
                [roi_trajectory.shape[1] - 1, roi_trajectory.shape[1] - 1],
                [min_y, max_y],
                color=ses_col[idx_target_ses],
                linestyle="-",
                linewidth=5,
            )
        ax.text(0, (min_y + max_y) / 2, f"Target Session", color=ses_col[idx_target_ses], ha="right", va="center", rotation=90)
        ax.set_xlim(0, roi_trajectory.shape[1])
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.set_ylabel("Trial")
        ax.set_yticks([])
        ax.set_xlabel("Virtual Position")

        alpha_values = np.where(~np.isnan(pfmax), pfmax / np.nanmax(pfmax), 0)
        ax = fig.add_subplot(132)
        ax.cla()
        ax.scatter(pfcom, range(len(pfcom)), s=10, color="k", alpha=alpha_values, linewidth=2)
        ax.scatter(pfloc, range(len(pfloc)), s=10, color="r", alpha=alpha_values, linewidth=2)
        ax.scatter([-10], [-10], color="k", s=10, alpha=1.0, linewidth=2, label="CoM")
        ax.scatter([-10], [-10], color="r", s=10, alpha=1.0, linewidth=2, label="MaxLoc")
        ax.set_xlim(0, roi_trajectory.shape[1])
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.legend(loc="upper center")
        ax.set_yticks([])
        ax.set_title("PF Location")
        ax.set_xlabel("Virtual Position")

        ax = fig.add_subplot(133)
        ax.cla()
        ax.scatter(pfmax, range(len(pfmax)), color="k", s=10, alpha=alpha_values)
        ax.set_xlim(0, np.nanmax(pfmax))
        ax.set_ylim(roi_trajectory.shape[0], 0)
        ax.set_title("PF Amplitude")
        ax.set_yticks([])
        ax.set_xlabel("Activity (sigma)")

        fig.suptitle(f"Mouse: {mouse_name}, ROI: {idx_roi_to_plot}, Target Session: {idx_target_ses}")

        return fig

    def __call__(self, mouse_name, roi_idx, min_percentile=90, max_percentile=100, idx_target_ses=0, dead_trials=5):
        return self.get_plot(mouse_name, roi_idx, min_percentile, max_percentile, idx_target_ses, dead_trials)


# Initialize the viewer globally
viewer = PlaceFieldViewer(fast_mode())


@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>Place Field Viewer</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1000px; 
                margin: 40px auto; 
                padding: 0 20px; 
            }
            .controls {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }
            .control-group {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .control-label {
                font-weight: bold;
                min-width: 100px;
            }
            button {
                padding: 8px 16px;
                font-size: 16px;
                cursor: pointer;
            }
            input {
                width: 80px;
                padding: 8px;
                font-size: 16px;
            }
            .plot-container {
                margin: 20px 0;
            }
            #true-roi-idx {
                margin-left: 10px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>Place Field Viewer</h1>
        
        <div class="controls">
            <div class="control-group">
                <span class="control-label">Mouse:</span>
                <button onclick="updateMouse(-1)">←</button>
                <span id="mouse-display"></span>
                <button onclick="updateMouse(1)">→</button>
            </div>
            
            <div class="control-group">
                <span class="control-label">ROI Index:</span>
                <button onclick="updateRoiIdx(-1)">←</button>
                <input type="number" id="roi-idx-input" value="0" onchange="setRoiIdx(this.value)">
                <button onclick="updateRoiIdx(1)">→</button>
                <span id="true-roi-idx"></span>
            </div>
            
            <div class="control-group">
                <span class="control-label">Percentiles:</span>
                <input type="number" style="width: auto;" id="min-percentile" value="90" min="0" max="100" 
                       onchange="setPercentiles()" style="width: 60px"> -
                <input type="number" style="width: auto;" id="max-percentile" value="100" min="0" max="100" 
                       onchange="setPercentiles()" style="width: 60px">
            </div>
            
            <div class="control-group">
                <span class="control-label">Target Session:</span>
                <button onclick="updateTargetSes(-1)">←</button>
                <input type="number" id="target-ses-input" value="0" onchange="setTargetSes(this.value)">
                <button onclick="updateTargetSes(1)">→</button>
            </div>
            
            <div class="control-group">
                <span class="control-label">Dead Trials:</span>
                <input type="number" style="width: auto;" id="dead-trials" value="5" min="0" onchange="setDeadTrials(this.value)">
            </div>

            <div class="control-group">
                <span class="control-label">Remember ROI:</span>
                <input type="button" style="width: auto;" id="remember-roi" value="Print ROI Details" onclick="printRoiDetails()">
            </div>
        </div>
        
        <div class="plot-container">
            <img id="plot-image" width="100%">
        </div>
        
        <script>
            let state = {
                mouseIndex: 0,
                roiIdx: 0,
                minPercentile: 90,
                maxPercentile: 100,
                targetSes: 0,
                deadTrials: 5,
                mouseNames: [],
                trueRoiIdx: 0
            };
            
            // Fetch initial data from server
            fetch('/init-data')
                .then(response => response.json())
                .then(data => {
                    state.mouseNames = data.mouse_names;
                    updateMouseDisplay();
                    updatePlot();
                });
            
            function updatePlot(reset_roi_idx=false) {
                if (reset_roi_idx) {
                    resetRoiIdx();
                    fetchAndUpdateRoi(state.roiIdx, updatePlotAfter=false);
                }

                const url = `/plot?mouse_idx=${state.mouseIndex}&roi_idx=${state.roiIdx}&min_percentile=${state.minPercentile}&max_percentile=${state.maxPercentile}&target_ses=${state.targetSes}&dead_trials=${state.deadTrials}`;
                document.getElementById('plot-image').src = url;
                
                // Update true ROI index display
                fetch(`/get-true-roi-idx?mouse_idx=${state.mouseIndex}&roi_idx=${state.roiIdx}&min_percentile=${state.minPercentile}&max_percentile=${state.maxPercentile}&idx_target_ses=${state.targetSes}`)
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('true-roi-idx').textContent = 
                            data.total_rois > 0 
                                ? `(True ROI: ${data.true_roi_idx}, Total: ${data.total_rois})`
                                : '(No ROIs in this percentile range)';
                    });
            }

            function resetRoiIdx() {
                // Reset the ROI index to 0 whenever the list of true ROIs changes.
                state.roiIdx = 0;
                document.getElementById('roi-idx-input').value = 0;
            }
            
            function updateMouse(delta) {
                state.mouseIndex = (state.mouseIndex + delta + state.mouseNames.length) % state.mouseNames.length;
                updateMouseDisplay();
                updatePlot(reset_roi_idx=true);
            }
            
            function updateMouseDisplay() {
                document.getElementById('mouse-display').textContent = state.mouseNames[state.mouseIndex];
            }

            function handleRoiUpdate(newRoiIdx, data, updatePlotAfter = true) {
                if (data.total_rois > 0) {
                    state.roiIdx = newRoiIdx % data.total_rois;
                    document.getElementById('true-roi-idx').textContent = 
                        `(True ROI: ${data.true_roi_idx}, Total: ${data.total_rois})`;
                } else {
                    state.roiIdx = 0;
                    document.getElementById('true-roi-idx').textContent = 
                        '(No ROIs in this percentile range)';
                }
                document.getElementById('roi-idx-input').value = state.roiIdx;
                
                if (updatePlotAfter) {
                    updatePlot(reset_roi_idx=false);
                }
            }

            function fetchAndUpdateRoi(newRoiIdx, updatePlotAfter = true) {
                fetch(`/get-true-roi-idx?mouse_idx=${state.mouseIndex}&roi_idx=${newRoiIdx}&min_percentile=${state.minPercentile}&max_percentile=${state.maxPercentile}&idx_target_ses=${state.targetSes}`)
                    .then(response => response.json())
                    .then(data => handleRoiUpdate(newRoiIdx, data, updatePlotAfter));
            }

            function updateRoiIdx(delta) {
                fetchAndUpdateRoi(state.roiIdx + delta);
            }

            function setRoiIdx(value) {
                const newValue = Math.max(0, parseInt(value) || 0);
                fetchAndUpdateRoi(newValue);
            }

            function setPercentiles() {
                let min = parseFloat(document.getElementById("min-percentile").value);
                let max = parseFloat(document.getElementById("max-percentile").value);
                
                min = Math.max(0, Math.min(100, min));
                max = Math.max(0, Math.min(100, max));
                
                if (min > max) {
                    let temp = min;
                    min = max;
                    max = temp;
                }
                
                state.minPercentile = min;
                state.maxPercentile = max;
                
                document.getElementById('min-percentile').value = min;
                document.getElementById('max-percentile').value = max;
                
                updatePlot(reset_roi_idx=true);
            }
            
            function updateTargetSes(delta) {
                state.targetSes = (state.targetSes + delta + 4) % 4;  // Hardcoded to 4 sessions
                document.getElementById('target-ses-input').value = state.targetSes;
                updatePlot(reset_roi_idx=true);
            }
            
            function setTargetSes(value) {
                state.targetSes = Math.max(0, parseInt(value) || 0) % 4;
                document.getElementById('target-ses-input').value = state.targetSes;
                updatePlot(reset_roi_idx=true);
            }
            
            function setDeadTrials(value) {
                state.deadTrials = Math.max(0, parseInt(value) || 5);
                document.getElementById('dead-trials').value = state.deadTrials;
                updatePlot();
            }

            function printRoiDetails() {
                fetch(`/print-roi-details?mouse_idx=${state.mouseIndex}&idx_target_ses=${state.targetSes}&roi_idx=${state.roiIdx}&min_percentile=${state.minPercentile}&max_percentile=${state.maxPercentile}`)
                    .then(response => response.text())
                    .then(data => {
                        console.log(data);
                    });
            }
        </script>
    </body>
    </html>
    """


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
