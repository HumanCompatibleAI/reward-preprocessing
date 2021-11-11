from imitation.rewards.reward_nets import RewardNet
import torch


class Preprocessor(RewardNet):
    """A Wrapper around a RewardNet which modifies its output
    to make it easier to interpret.
    This class implements the identity preprocessor and is meant to be
    used as a base class for more complex preprocessors.

    Note: the weights of the wrapped RewardNet are frozen
    because when training the Proprocessor, the original model
    weights should stay fixed. The model is also put into eval mode.
    Both changes can be undone using unfreeze_model().

    Args:
        model: the RewardNet to be wrapped
    """

    def __init__(self, model: RewardNet, freeze_model: bool = True):
        super().__init__(
            model.observation_space, model.action_space, model.normalize_images
        )
        self.model = model
        if freeze_model:
            self.freeze_model()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(state, action, next_state, done)

    def freeze_model(self):
        """Freeze the weights of the wrapped model and put it into eval mode.

        This method is called automatically on instantiation.
        Can be reversed with unfreeze_model()."""
        for param in self.model.parameters():
            param.requires_grad = False
        # store whether the model was originally in train mode,
        # so we know which mode to put it in when unfreezing the weights
        self._was_training = self.model.training
        self.model.eval()

    def unfreeze_model(self):
        """Unfreeze the weights of the wrapped model and put it into its
        original mode (eval or train).
        """
        for param in self.model.parameters():
            param.requires_grad = True
        if self._was_training:
            self.model.train()

    @property
    def unwrapped(self):
        if isinstance(self.model, Preprocessor):
            return self.model.unwrapped
        else:
            return self.model


class ScaleShift(Preprocessor):
    def __init__(
        self,
        model: RewardNet,
        scale: bool = True,
        shift: bool = True,
        freeze_model: bool = True,
    ):
        super().__init__(model, freeze_model=freeze_model)
        if scale:
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
        if shift:
            self.shift = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(state, action, next_state, done)
        if hasattr(self, "shift"):
            out = out - self.shift
        if hasattr(self, "scale"):
            out = self.scale * out
        return out
