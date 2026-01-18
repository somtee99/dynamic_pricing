from stable_baselines3.common.callbacks import BaseCallback


class MetricLogger(BaseCallback):
    def __init__(self,log_frequency=100, verbose=1):
        super(MetricLogger, self).__init__(verbose)
        self.verbose=verbose
        self.log_frequency=log_frequency
        self.value_lossess=[]

    def _on_step(self) -> bool:
        if self.n_calls % self.log_frequency == 0:
            if (self.verbose == 1):
                print(f"iterations: {self.model.logger.name_to_value['train/n_updates']}")
                print(f"ep_rew_mean: {self.model.logger.name_to_value['train/ep_rew_mean']}")
                print(f"policy_loss: {self.model.logger.name_to_value['train/policy_loss']}")
                print(f"value_loss: {self.model.logger.name_to_value['train/value_loss']}")
                print(f"entropy_loss: {self.model.logger.name_to_value['train/entropy_loss']}")
                print("--------------------------------")
                self.value_lossess.append(self.model.logger.name_to_value['train/value_loss'])

        return True