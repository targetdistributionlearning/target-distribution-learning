from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'value', 'action_mean', 'action', 'y', 'mask', 'next_state', 'reward'))