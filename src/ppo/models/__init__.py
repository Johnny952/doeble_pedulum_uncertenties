from .model import Model

def make_model(
        state_stack: int,
        input_dim: int=11,
        output_dim: int=1,
        architecture: "list[int]"=[256, 128, 64],
        model='base',
    ):
    switcher = {
        'base': Model,
        # 'dropout': DropoutTrainerModel,
        # 'dropout2': DropoutTrainerModel2,
        # 'bootstrap': BootstrapTrainerModel,
        # 'bootstrap2': BootstrapTrainerModel2,
        # 'sensitivity': SensitivityTrainerModel,
        # 'bnn': BNNTrainerModel,
        # 'bnn2': BNNTrainerModel2,
        # 'aleatoric': AleatoricTrainerModel,
        # 'vae': VAETrainerModel,
    }
    return switcher.get(model, Model)(
        state_stack,
        input_dim=input_dim,
        output_dim=output_dim,
        architecture=architecture,
    )
