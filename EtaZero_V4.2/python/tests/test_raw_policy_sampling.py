import numpy as np
import torch

from katago.game.board import Board
from katago.game.gamestate import GameState
from katago.train import modelconfigs
from katago.train.model_pytorch import Model


def test_genmove_samples_only_the_mass_of_legal_moves(monkeypatch):
    model = Model(modelconfigs.config_of_name["b2c16"], 5)
    model.initialize()
    with torch.no_grad():
        for parameter in model.policy_head.parameters():
            parameter.zero_()
    game = GameState(5, GameState.RULES_TT)
    game.play(Board.BLACK, game.board.loc(2, 2))
    # 24 board moves and pass share the legal mass. Quantile .945 is
    # the final board move, regardless of the occupied point's logit.
    monkeypatch.setattr(np.random, "beta", lambda a, b: .945)
    expected = game.board.loc(4, 4)
    assert game.get_model_outputs(model)["genmove_result"] == expected

    def boost_occupied(_module, _inputs, logits):
        logits[:, :, 12] = 1000

    hook = model.policy_head.register_forward_hook(boost_occupied)
    try:
        outputs = game.get_model_outputs(model)
        assert outputs["policy0"][12] == 1.0
        assert outputs["genmove_result"] == expected
    finally:
        hook.remove()
