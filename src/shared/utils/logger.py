import wandb

class Logger(object):

    def __init__(self, project_name, group, model_name, run_id, args: None):
        wandb.init(project=project_name, group=group, name=model_name, id=run_id)
        config = wandb.config
        config.args = args

    def watch(self, model):
        wandb.watch(model)
    
    def log(self, to_log):
        wandb.log(to_log)