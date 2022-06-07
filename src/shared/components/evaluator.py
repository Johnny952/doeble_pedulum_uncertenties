import numpy as np
import os

from ppo.components.uncert_agents import make_agent
from ppo.components.trainer import adjust_range
from shared.components.env import Env
from shared.utils.utils import save_uncert, init_uncert_file

class Evaluator:
    def __init__(self, state_stack, action_repeat, model_name, device='cpu', evaluations=1, seed=123, base_path='uncertainties/customeval', nb=1) -> None:
        self.evaluations = evaluations
        self.state_stack = state_stack
        self.action_repeat = action_repeat
        self.seed = seed
        self.evaluations = evaluations
        self.model_name = model_name
        self.base_path = base_path
        self.noise = None
        self.nb = nb

        self._eval_env = self.load_env()

        self._agent = make_agent(
            model="base",
            gamma=0.99,
            buffer=None,
            logger=None,
            device=device,
            batch_size=64,
            lr=0.001,
            nb_nets=1,
            ppo_epoch=10,
            clip_param=0.2,
        )
        self._agent.load('../shared/components/controller_ppo.pkl', eval_mode=True)

        self.evaluation_nb = 0

        if not os.path.exists(base_path):
            os.makedirs(base_path)
        init_uncert_file(file=f"{self.base_path}/{self.model_name}.txt")
    
    def load_env(self):
        self._eval_env = Env(
            state_stack=self.state_stack,
            action_repeat=self.action_repeat,
            seed=self.seed,
            path_render=f"../shared/components/render/{self.nb}-{self.evaluation_nb}",
            evaluations=self.evaluations,
            done_reward_threshold=-1000,
            noise=self.noise
        )
    
    def set_noise_value(self, noise):
        self.noise = noise
    
    def eval(self, episode_nb, agent):
        self._eval(episode_nb, agent)
    
    def ppo_step(self, action):
        return action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])

    def _eval(self, episode_nb, agent, default_action=0, default_steps=[70, 71]):
        self.load_env()
        for i_val in range(self.evaluations):
            score = 0
            steps = 0
            state = self._eval_env.reset()
            die = False

            uncert = []
            i_step = 0
            while not die:
                action = self._agent.select_action(state, eval=True)[0]
                epis, aleat = agent.select_action(state, eval=True)[-1]
                uncert.append(
                    [epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]]
                )
                action = default_action if i_step in default_steps else action
                action = adjust_range(action, target_range=self._eval_env.observation_space)

                state_, reward, _, die = self._eval_env.step(action)[:4]
                score += reward
                state = state_
                steps += 1
                i_step += 1

            uncert = np.array(uncert)
            save_uncert(
                episode_nb,
                i_val,
                score,
                uncert,
                file=f"{self.base_path}/{self.model_name}.txt",
                sigma=self._eval_env.random_noise,
            )

            self.evaluation_nb += 1
        self._eval_env.close()

    def eval2(self, episode_nb, agent):
        self._eval(episode_nb, agent, default_steps=[25, 100])