let state = {
    mouseIndex: 0,
    roiIdx: 0,
    minPercentile: 90,
    maxPercentile: 100,
    targetSes: 0,
    deadTrials: 5,
    mouseNames: [],
    trueRoiIdx: 0,
    redCells: true,
};

// Fetch initial data from server
fetch('/init-data')
    .then(response => response.json())
    .then(data => {
        state.mouseNames = data.mouse_names;
        updateMouseDisplay();
        updateRedCellsDisplay();
        updatePlot(reset_roi_idx=true);
    });

function updateRedCells(value) {
    state.redCells = value;
    updatePlot(reset_roi_idx=true);
}

function updatePlot(reset_roi_idx=false) {
    if (reset_roi_idx) {
        resetRoiIdx();
        fetchAndUpdateRoi(state.roiIdx, updatePlotAfter=false);
    }

    const url = `/plot?mouse_idx=${state.mouseIndex}&roi_idx=${state.roiIdx}&min_percentile=${state.minPercentile}&max_percentile=${state.maxPercentile}&target_ses=${state.targetSes}&dead_trials=${state.deadTrials}&red_cells=${state.redCells}`;
    document.getElementById('plot-image').src = url;
    
    // Update true ROI index display
    fetch(`/get-true-roi-idx?mouse_idx=${state.mouseIndex}&roi_idx=${state.roiIdx}&min_percentile=${state.minPercentile}&max_percentile=${state.maxPercentile}&idx_target_ses=${state.targetSes}&red_cells=${state.redCells}`)
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

function updateRedCellsDisplay() {
    document.getElementById('red-cells-display').checked = state.redCells;
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
    fetch(`/get-true-roi-idx?mouse_idx=${state.mouseIndex}&roi_idx=${newRoiIdx}&min_percentile=${state.minPercentile}&max_percentile=${state.maxPercentile}&idx_target_ses=${state.targetSes}&red_cells=${state.redCells}`)
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
    // Get the number of sessions for current mouse from the server
    fetch(`/get-sessions?mouse_idx=${state.mouseIndex}`)
        .then(response => response.json())
        .then(data => {
            state.targetSes = (state.targetSes + delta + data.num_sessions) % data.num_sessions;
            document.getElementById('target-ses-input').value = state.targetSes;
            updatePlot(reset_roi_idx=true);
        });
}

function setTargetSes(value) {
    fetch(`/get-sessions?mouse_idx=${state.mouseIndex}`)
        .then(response => response.json())
        .then(data => {
            state.targetSes = Math.max(0, parseInt(value) || 0) % data.num_sessions;
            document.getElementById('target-ses-input').value = state.targetSes;
            updatePlot(reset_roi_idx=true);
        });
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