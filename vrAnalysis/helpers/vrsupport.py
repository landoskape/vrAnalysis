def get_env_order(mousedb, mouse_name):
    env_order = mousedb.getTable(mouseName=mouse_name)["environmentOrder"].item()
    return [int(env) for env in env_order.split(".")]
