# Lazy imports to avoid circular import issues when running modules directly
def __getattr__(name):
    if name == 'Emerson2017Evaluator':
        from .emerson_2017_disease_classification import Emerson2017Evaluator
        return Emerson2017Evaluator
    if name == 'GIANAEvaluator':
        from .giana_2021_disease_classification import GIANAEvaluator
        return GIANAEvaluator
    if name == 'DemographicFeaturesEvaluator':
        from .demographic_features_disease_classification import DemographicFeaturesEvaluator
        return DemographicFeaturesEvaluator
    if name == 'EnsembleRegressionEvaluator':
        from .ensemble_regression_disease_classification import EnsembleRegressionEvaluator
        return EnsembleRegressionEvaluator
    if name == 'VJDemographicsEvaluator':
        from .vjgene_demographics_disease_classification import VJDemographicsEvaluator
        return VJDemographicsEvaluator
    if name == 'ExternalEvaluator':
        from .external_evaluation import ExternalEvaluator
        return ExternalEvaluator
    if name == 'Emerson2017DriverIdentificationEvaluator':
        from .emerson_2017_driver_identification import Emerson2017DriverIdentificationEvaluator
        return Emerson2017DriverIdentificationEvaluator
    if name == 'EnsembleRegressionDriverIdentificationEvaluator':
        from .ensemble_regression_driver_identification import EnsembleRegressionDriverIdentificationEvaluator
        return EnsembleRegressionDriverIdentificationEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['Emerson2017Evaluator', 'GIANAEvaluator', 'DemographicFeaturesEvaluator',
           'EnsembleRegressionEvaluator', 'VJDemographicsEvaluator',
           'ExternalEvaluator',
           'Emerson2017DriverIdentificationEvaluator',
           'EnsembleRegressionDriverIdentificationEvaluator']