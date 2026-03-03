# Lazy imports to avoid circular import issues when running modules directly
def __getattr__(name):
    if name == 'Emerson2017Evaluator':
        from .emerson_2017_disease_classification import Emerson2017Evaluator
        return Emerson2017Evaluator
    if name == 'GIANA2020Evaluator':
        from .giana_2020_disease_classification import GIANA2020Evaluator
        return GIANA2020Evaluator
    if name == 'DemographicFeaturesEvaluator':
        from .demographic_features_disease_classification import DemographicFeaturesEvaluator
        return DemographicFeaturesEvaluator
    if name == 'MLBaselineEvaluator':
        from .ml_baseline_disease_classification import MLBaselineEvaluator
        return MLBaselineEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['Emerson2017Evaluator', 'GIANA2020Evaluator', 'DemographicFeaturesEvaluator',
           'MLBaselineEvaluator']
