"""Translate directory responsibilities to existing native consumers."""
from copy import deepcopy

from .profiles import read_sections, settings


def translate_profile(directory, selfplay_path):
    from katago.train.modelconfigs import config_of_name
    env_path = directory / 'env.cfg'
    env = read_sections(env_path, ['environment'])['environment'].copy()
    identity = settings({k: env.pop(k) for k in ('game', 'canvas')},
                        dict(game=str, canvas=int), env_path)
    game, canvas = identity['game'], identity['canvas']
    sections = read_sections(selfplay_path, ['agent', 'selfplay', 'gumbel'])
    agent = settings(sections.get('agent', {}),
                     dict(algorithm=str, root_search_algo=str, nonroot_search_algo=str), selfplay_path)
    family = agent['algorithm']
    root, nonroot = agent['root_search_algo'], agent['nonroot_search_algo']
    if family not in ('alphazero', 'muzero') or root not in ('puct', 'gumbel') or nonroot not in ('puct', 'gumbel'):
        raise ValueError('agent requires alphazero/muzero and puct/gumbel search choices')
    if root == 'puct' and nonroot != 'puct':
        raise ValueError('Gumbel nonroot search requires a Gumbel root')
    values = dict(game=game, canvas=canvas, algorithm=('gumbel_' if root == 'gumbel' else '') + family)
    net_path = directory / 'net.cfg'
    net = read_sections(net_path, ['network', 'muzero'])
    network = settings(net.get('network', {}), dict(architecture=str), net_path)
    if family == 'muzero':
        mu = settings(net.get('muzero', {}), dict(representation=str, dynamics=str, prediction=str,
                      latent_channels=int, unroll_steps=int, prediction_blocks=int), net_path)
        bundle = dict(game=game, latent_channels=mu['latent_channels'])
        for role in ('representation', 'dynamics', 'prediction'):
            component = deepcopy(config_of_name[mu[role]])
            component.update(version=17 if game == 'go' else 19 if game == 'xiangqi' else 18,
                             predict_q_values=True)
            if role == "prediction":
                count = mu.get("prediction_blocks", 0)
                if count < 0:
                    raise ValueError("prediction_blocks must be nonnegative")
                kinds = component["block_kind"]
                component["block_kind"] = [[f"block{i+1}", kinds[i % len(kinds)][1]] for i in range(count)]
                if count == 0:
                    component["trunk_num_channels"] = mu["latent_channels"]
            bundle[role] = component
        values.update(model_kind=None, muzero_network=bundle, muzero_unroll_steps=mu['unroll_steps'])
    else:
        values.update(model_kind=network['architecture'], muzero_network=None, muzero_unroll_steps=None)

    # Native keys remain native keys; only the responsibilities above are translated.
    native = sections.get('selfplay', {}).copy()
    if root == 'gumbel':
        native.update(sections.get('gumbel', {}))
    # These real-board mechanisms cannot be used by latent/fresh-root searches.
    if family == 'muzero':
        native.update(useGraphSearch='false', subtreeValueBiasFactor='0', avoidRepeatedPatternUtility='0')
    if root == 'gumbel':
        native.update(useGraphSearch='false', rootSymmetryPruning='false')
    # Environment owns the board domain; don't let a shared Go default leak into Xiangqi.
    for key in ('game', 'dataBoardLen', 'bSizes', 'bSizesXY', 'bSizeRelProbs', 'allowRectangleProb'):
        native.pop(key, None)
    native.update(env)
    native.update(game=game, dataBoardLen=str(canvas), rootSearchAlgorithm=root,
                  nonrootSearchAlgorithm=nonroot, useMuZero=str(family == 'muzero').lower(),
                  switchNetsMidGame='false')
    if game != 'go':
        native.update(komiMean='0', handicapProb='0')
    lines = []
    for key, raw in native.items():
        value = (raw or '').split('#', 1)[0].strip()
        if value:
            lines.append(f'{key} = {value}')
    values['native_selfplay'] = '\n'.join(lines) + '\n'
    return values
