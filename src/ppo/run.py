import os
import argparse
import torch
import uuid
import glob
from tqdm import tqdm
import numpy as np
from termcolor import colored
from pyvirtualdisplay import Display
from collections import namedtuple

import sys
sys.path.append('..')
from shared.utils.utils import init_uncert_file
from shared.components.env import Env
from shared.utils.replay_buffer import ReplayMemory
from shared.components.logger import Logger
from shared.components.evaluator import Evaluator
from components.uncert_agents import make_agent
from models import make_model
from components.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for Inverted Double Pendulum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment Config
    env_config = parser.add_argument_group("Environment config")
    env_config.add_argument(
        "-AR", "--action-repeat", type=int, default=1, help="repeat action in N frames"
    )
    env_config.add_argument(
        "-TS",
        "--train-seed",
        type=float,
        default=0,
        help="Train Environment Random seed",
    )
    env_config.add_argument(
        "-ES",
        "--eval-seed",
        type=float,
        default=10,
        help="Evaluation Environment Random seed",
    )
    env_config.add_argument(
        "-TSE",
        "--test-seed",
        type=float,
        default=20,
        help="Test Environment Random seed",
    )
    env_config.add_argument(
        "-N",
        "--noise",
        type=str,
        default="0,0.1",
        default=None,
        help='Whether to use noise or not, and standard deviation bounds separated by comma (ex. "0,0.5")',
    )
    env_config.add_argument(
        "-NS", "--noise-steps", type=int, default=50, help="Number of noise steps",
    )

    # Agent Config
    agent_config = parser.add_argument_group("Agent config")
    agent_config.add_argument(
        "-M",
        "--model",
        type=str,
        default="base",
        help='Type of uncertainty model: "base", "sensitivity", "dropout", "bootstrap", "aleatoric", "bnn" or "custom"',
    )
    agent_config.add_argument(
        "-NN",
        "--nb-nets",
        type=int,
        default=10,
        help="Number of networks to estimate uncertainties",
    )
    agent_config.add_argument(
        "-G", "--gamma", type=float, default=0.99, help="discount factor"
    )
    agent_config.add_argument(
        "-SS", "--state-stack", type=int, default=6, help="Number of state stack as observation"
    )
    agent_config.add_argument(
        "-A",
        "--architecture",
        type=str,
        default="1024",
        help='Base network architecture',
    )
    agent_config.add_argument(
        "-PE",
        "--ppo-epoch",
        type=int,
        default=20,
        help='Number of training updates each time buffer is full',
    )
    agent_config.add_argument(
        '-FC',
        '--from-checkpoint', 
        type=str, 
        default=None, 
        help='Path to trained model')
    agent_config.add_argument(
        '-CP',
        '--clip-param', 
        type=float, 
        default=0.1, 
        help='Clip Parameter')

    # Training Config
    train_config = parser.add_argument_group("Train config")
    train_config.add_argument(
        "-E", "--episodes", type=int, default=50000, help="Number of training episode"
    )
    train_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    train_config.add_argument(
        "-EI",
        "--eval-interval",
        type=int,
        default=20,
        help="Interval between evaluations",
    )
    train_config.add_argument(
        "-EV",
        "--evaluations",
        type=int,
        default=3,
        help="Number of evaluations episodes every x training episode",
    )
    train_config.add_argument(
        "-ER",
        "--eval-render",
        action="store_true",
        help="render the environment on evaluation",
    )
    train_config.add_argument(
        "-DB",
        "--debug",
        action="store_true",
        help="debug mode",
    )

    # Update
    update_config = parser.add_argument_group("Update config")
    update_config.add_argument(
        "-BC", "--buffer-capacity", type=int, default=1000, help="Buffer Capacity"
    )
    update_config.add_argument(
        "-BS", "--batch-size", type=int, default=128, help="Batch Capacity"
    )
    update_config.add_argument(
        "-LR", "--learning-rate", type=float, default=0.001, help="Learning Rate"
    )

    # Test Config
    test_config = parser.add_argument_group("Test config")
    test_config.add_argument(
        "-TE",
        "--test-episodes",
        type=int,
        default=3,
        help="Number evaluations each noise step",
    )
    test_config.add_argument(
        "-TR",
        "--test-render",
        action="store_true",
        help="render the environment on testing",
    )
    test_config.add_argument(
        "-OT",
        "--ommit-training",
        action="store_true",
        help="Whether to ommit training the agent or not",
    )

    args = parser.parse_args()
    
    run_id = uuid.uuid4()
    # run_name = f"{args.model}_{run_id}"
    run_name = args.model
    render_path = "render"
    render_model_path = f"{render_path}/train"
    train_render_model_path = f"{render_model_path}/{run_name}"
    render_test_path = f"{render_path}/test"
    test_render_model_path = f"{render_test_path}/{run_name}"
    param_path = "param"
    uncertainties_path = "uncertainties"
    uncertainties_train_path = f"{uncertainties_path}/train"
    uncertainties_file_path = f"{uncertainties_train_path}/{run_name}.txt"
    uncertainties_test_path = f"{uncertainties_path}/test"
    uncertainties_test_file_path = f"{uncertainties_test_path}/{run_name}.txt"

    print(colored("Initializing data folders", "blue"))
    # Init model checkpoint folder and uncertainties folder
    if not args.debug:
        if not os.path.exists(param_path):
            os.makedirs(param_path)
        if not os.path.exists(uncertainties_path):
            os.makedirs(uncertainties_path)
        if not os.path.exists(render_path):
            os.makedirs(render_path)
        if not os.path.exists(render_model_path):
            os.makedirs(render_model_path)
        if not os.path.exists(train_render_model_path):
            os.makedirs(train_render_model_path)
        elif not args.ommit_training:
            files = glob.glob(f"{train_render_model_path}/*")
            for f in files:
                os.remove(f)
        if not os.path.exists(test_render_model_path):
            os.makedirs(test_render_model_path)
        else:
            files = glob.glob(f"{test_render_model_path}/*")
            for f in files:
                os.remove(f)
        if not os.path.exists(uncertainties_train_path):
            os.makedirs(uncertainties_train_path)
        if not os.path.exists(uncertainties_test_path):
            os.makedirs(uncertainties_test_path)
        init_uncert_file(file=uncertainties_file_path)
        init_uncert_file(file=uncertainties_test_file_path)
    print(colored("Data folders created successfully", "green"))

    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Whether to use cuda or cpu
    if args.device == "auto":
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.train_seed)
        if use_cuda:
            torch.cuda.manual_seed(args.train_seed)
    else:
        device = args.device
    print(colored(f"Using: {device}", "green"))

    # Init logger
    logger = Logger("inv-pendulum-ppo", args.model, run_name, str(run_id), args=vars(args))
    config = logger.get_config()

    # Noise parser
    if config["noise"]:
        add_noise = [float(bound) for bound in config["noise"].split(",")]
    else:
        add_noise = None

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    env = Env(
        state_stack=config["state_stack"],
        action_repeat=config["action_repeat"],
        seed=config["train_seed"],
        noise=add_noise,
        done_reward_threshold=-1000
    )
    eval_env = Env(
        state_stack=config["state_stack"],
        action_repeat=config["action_repeat"],
        seed=config["eval_seed"],
        path_render=train_render_model_path if config["eval_render"] else None,
        evaluations=config["evaluations"],
        done_reward_threshold=-1000
    )
    Transition = namedtuple(
        "Transition", ("state", "action", "reward", "next_state", "a_logp")
    )
    buffer = ReplayMemory(
        config["buffer_capacity"],
        config["batch_size"],
        Transition
    )
    architecture = [int(l) for l in config["architecture"].split("-")]
    model = make_model(
        model=config["model"],
        state_stack=config["state_stack"],
        input_dim=env.observation_dims,
        output_dim=env.action_dims,
        architecture=architecture,
    ).to(device)
    agent = make_agent(
        model=model,
        gamma=config["gamma"],
        buffer=buffer,
        logger=logger,
        device=device,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        nb_nets=config["nb_nets"],
        ppo_epoch=config["ppo_epoch"],
        clip_param=config["clip_param"]
    )
    # evaluator = None
    # if not args.ommit_training:
    #     evaluator = Evaluator(
    #         args.img_stack,
    #         args.action_repeat,
    #         args.model,
    #         device=device,
    #     )
    init_epoch = 0
    if config["from_checkpoint"]:
        init_epoch = agent.load(config["from_checkpoint"])
    print(colored("Agent and environments created successfully", "green"))

    noise_print = "not using noise"
    if env.use_noise:
        if env.generate_noise:
            noise_print = f"using noise with [{env.noise_lower}, {env.noise_upper}] std bounds"
        else:
            noise_print = f"using noise with [{env.random_noise}] std"

    episodes = config["episodes"]
    print(
        colored(
            f"Training {type(agent)} during {episodes} epochs and {noise_print}",
            "magenta",
        )
    )

    for name, param in config.items():
        print(colored(f"{name}: {param}", "cyan"))

    trainer = Trainer(
        agent,
        env,
        eval_env,
        logger,
        config["episodes"],
        init_ep=init_epoch,
        nb_evaluations=config["evaluations"],
        eval_interval=config["eval_interval"],
        model_name=run_name,
        checkpoint_every=10,
        debug=config["debug"],
        # evaluator=evaluator,
    )

    if not args.ommit_training:
        trainer.run()
    else:
        print(colored("\nTraining Ommited", "magenta"))
    env.close()
    eval_env.close()

    del env
    del eval_env
    del trainer
    del agent
    # del evaluator

    print(colored("\nTraining completed, now testing", "green"))
    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    agent = make_agent(
        model=model,
        gamma=config["gamma"],
        buffer=buffer,
        logger=logger,
        device=device,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        nb_nets=config["nb_nets"],
        ppo_epoch=config["ppo_epoch"],
        clip_param=config["clip_param"]
    )
    test_env = Env(
        state_stack=config["state_stack"],
        action_repeat=config["action_repeat"],
        seed=config["eval_seed"],
        path_render=test_render_model_path if config["test_render"] else None,
        evaluations=config["test_episodes"],
        done_reward_threshold=-1000,
        noise=add_noise,
    )
    agent.load(f"param/best_{run_name}.pkl", eval_mode=True)
    print(colored("Agent and environments created successfully", "green"))
    
    noise_print = "not using noise"
    if test_env.use_noise:
        if test_env.generate_noise:
            noise_print = f"using noise with [{test_env.noise_lower}, {test_env.noise_upper}] std bounds"
        else:
            noise_print = f"using noise with [{test_env.random_noise}] std"

    print(
        colored(
            "Testing {} during {} noise steps and {}".format(run_name, config["noise_steps"], noise_print),
            "magenta",
        )
    )

    # Test increasing noise
    for idx, noise in enumerate(tqdm(np.linspace(add_noise[0], add_noise[1], config["noise_steps"]))):
        test_env.set_noise_value(noise)
        # if evaluator:
        #     evaluator.set_noise_value(noise)
        trainer = Trainer(
            agent,
            None,
            test_env,
            logger,
            0,
            nb_evaluations=config["test_episodes"],
            model_name=run_name,
            debug=config["debug"],
            # evaluator=evaluator,
        )
        trainer.eval(idx, mode="test")
    
    # Test noise 0
    #     evaluator = Evaluator(
    #         args.img_stack,
    #         args.action_repeat,
    #         args.model,
    #         device=device,
    #         base_path='uncertainties/customtest0'
    #     )
    test_env.use_noise = False
    for idx in tqdm(range(config["noise_steps"])):
        trainer = Trainer(
            agent,
            None,
            test_env,
            logger,
            0,
            nb_evaluations=config["test_episodes"],
            model_name=run_name,
            debug=config["debug"],
            # evaluator=evaluator,
        )
        trainer.eval(idx, mode="test0")
    
    # Test controller 1 and 2
    evaluator = Evaluator(
        state_stack=config["state_stack"],
        action_repeat=config["action_repeat"],
        model_name=run_name, 
        device=device,
        base_path='uncertainties/customtest1',
        nb=1,
    )
    evaluator.eval(0, agent)

    evaluator = Evaluator(
        state_stack=config["state_stack"],
        action_repeat=config["action_repeat"],
        model_name=run_name, 
        device=device,
        base_path='uncertainties/customtest2',
        nb=2,
    )
    evaluator.eval2(0, agent)

    test_env.close()
    print(colored("\nTest completed", "green"))