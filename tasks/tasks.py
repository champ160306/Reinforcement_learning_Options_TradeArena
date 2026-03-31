def get_task_config(task_name):
    if task_name == "easy":
        return {"noise": 0}

    elif task_name == "medium":
        return {"noise": 0.1}

    elif task_name == "hard":
        return {"noise": 0.2}

    else:
        raise ValueError("Invalid task")