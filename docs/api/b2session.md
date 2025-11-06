# B2 Session API Reference

B2Sessions are the core class for loading and managing VR session data. They are the primary [`SessionData`](sessions.md#vrAnalysis.sessions.base.SessionData) object built specifically for data coming from the B2 rig. 

## Main Classes and Functions

::: vrAnalysis.sessions.create_b2session

::: vrAnalysis.sessions.B2Session
    options:
      members:
        - __init__
        - get_spks
        - update_params
        - load_s2p
        - get_roi_position
        - get_validity_indices
        - get_red_idx
        - spks
        - timestamps
        - positions
        - trial_environment
        - environments
        - num_trials
        - idx_rois

::: vrAnalysis.sessions.B2SessionParams
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members:
        - from_dict
        - update
        - set_good_labels
        - good_label_idx

::: vrAnalysis.sessions.b2session.B2RegistrationOpts
    options:
      show_root_heading: true
      show_root_toc_entry: true