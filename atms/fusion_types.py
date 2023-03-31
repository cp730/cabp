from collections import namedtuple

Genotype = namedtuple('Genotype', 'steps' )
StepGenotype = namedtuple('StepGenotype', 'inner_steps')

PRIMITIVES = [
    'none',
    'skip'
]

STEP_EDGE_PRIMITIVES = [
    'none',
    'skip'
]

STEP_STEP_PRIMITIVES = [
    'Sum',
    'ScaleDotAttn',
    # 'LinearGLU',
    # 'ConcatFC'
]

FUSION_CHOICES = [

    # 'tirg',
    'add',
    'Combiner',
    'artemis_attention',
    'GLG',
    'CMR',
    'rtic',
    'film',
    'gf'
]
