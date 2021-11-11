from typing import Any, Dict, Mapping, Tuple, Union

from stable_baselines3.common.logger import KVWriter
import wandb


class WandbOutputFormat(KVWriter):
    def __init__(self, wb_options: Mapping[str, Any], config: Mapping[str, Any]):
        wandb.init(
            project="reward_preprocessing",
            job_type="agent",
            config=config,
            **wb_options,
        )

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        # TODO: This doesn't support all use cases
        # (e.g. key_excluded is just ignored and videos won't work).
        wandb.log(key_values, step=step)

    def close(self) -> None:
        wandb.finish()
