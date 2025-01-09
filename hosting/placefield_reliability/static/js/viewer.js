let state = {
    mouseIndex: 0,
    mouseNames: [],
    sessionRange: {
        min: null,
        max: null,
        current: {
            min: null,
            max: null
        }
    }
};

// Fetch initial data from server
fetch('/init-data')
    .then(response => response.json())
    .then(data => {
        state.mouseNames = data.mouse_names;
        updateMouseDisplay();
        updateSessionRange();
    });

function updateSessionRange() {
    fetch(`/session-range?mouse_idx=${state.mouseIndex}`)
        .then(response => response.json())
        .then(data => {
            state.sessionRange.min = data.min_session;
            state.sessionRange.max = data.max_session;
            
            // Only update current values if they're not set or out of range
            if (state.sessionRange.current.min === null || 
                state.sessionRange.current.min < data.min_session || 
                state.sessionRange.current.min > data.max_session) {
                state.sessionRange.current.min = data.min_session;
            }
            if (state.sessionRange.current.max === null || 
                state.sessionRange.current.max < data.min_session || 
                state.sessionRange.current.max > data.max_session) {
                state.sessionRange.current.max = data.max_session;
            }
            
            updateSessionInputs();
        });
}

function updateSessionInputs() {
    const minInput = document.getElementById('min-session');
    const maxInput = document.getElementById('max-session');
    
    // Get current values
    let currentMin = parseInt(minInput.value);
    let currentMax = parseInt(maxInput.value);
    
    // Validate and adjust values if needed
    if (isNaN(currentMin) || currentMin < state.sessionRange.min) {
        currentMin = state.sessionRange.min;
    } else if (currentMin > state.sessionRange.max) {
        currentMin = state.sessionRange.max;
    }
    
    if (isNaN(currentMax) || currentMax > state.sessionRange.max) {
        currentMax = state.sessionRange.max;
    } else if (currentMax < state.sessionRange.min) {
        currentMax = state.sessionRange.min;
    }
    
    // Ensure min doesn't exceed max
    if (currentMin > currentMax) {
        if (minInput === document.activeElement) {
            currentMax = currentMin;
        } else {
            currentMin = currentMax;
        }
    }
    
    // Update state and inputs
    state.sessionRange.current.min = currentMin;
    state.sessionRange.current.max = currentMax;
    
    minInput.value = currentMin;
    maxInput.value = currentMax;
    
    // Update input constraints
    minInput.min = state.sessionRange.min;
    minInput.max = state.sessionRange.max;
    maxInput.min = state.sessionRange.min;
    maxInput.max = state.sessionRange.max;

    updatePlot();
}

function resetSessionRange() {
    // Reset state values
    state.sessionRange.current.min = state.sessionRange.min;
    state.sessionRange.current.max = state.sessionRange.max;
    
    // Update input elements directly
    const minInput = document.getElementById('min-session');
    const maxInput = document.getElementById('max-session');
    minInput.value = state.sessionRange.min;
    maxInput.value = state.sessionRange.max;
    
    updateSessionInputs();
}

function updatePlot() {
    const useRelcor = document.getElementById('plot-type').value === 'relcor';
    const average = document.getElementById('view-type').value === 'average';
    const tracked = document.getElementById('tracked').checked;
    
    const url = `/plot?mouse_idx=${state.mouseIndex}&use_relcor=${useRelcor}&tracked=${tracked}&average=${average}&min_session=${state.sessionRange.current.min}&max_session=${state.sessionRange.current.max}`;
    document.getElementById('plot-image').src = url;
}

function updateMouse(delta) {
    state.mouseIndex = (state.mouseIndex + delta + state.mouseNames.length) % state.mouseNames.length;
    updateMouseDisplay();
    updateSessionRange();
}

function updateMouseDisplay() {
    document.getElementById('mouse-display').textContent = state.mouseNames[state.mouseIndex];
}