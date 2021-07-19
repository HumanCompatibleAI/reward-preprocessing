import torch

from reward_preprocessing.models import RewardModel
from reward_preprocessing.transition import Transition


class Preprocessor(RewardModel):
    """A Wrapper around a RewardModel which modifies its output
    to make it easier to interpret.
    This class implements the identity preprocessor and is meant to be
    used as a base class for more complex preprocessors.

    Note: the weights of the wrapped RewardModel are frozen
    because when training the Proprocessor, the original model
    weights should stay fixed. The model is also put into eval mode.
    Both changes can be undone using unfreeze_model().

    Args:
        model: the RewardModel to be wrapped
    """

    def __init__(self, model: RewardModel):
        super().__init__(model.state_shape)
        self.model = model
        self.freeze_model()

    def forward(self, transitions: Transition) -> torch.Tensor:
        return self.model(transitions)

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
