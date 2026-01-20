# Lazy imports to avoid circular import issues when running modules directly
def __getattr__(name):
    if name == 'Emerson2017Evaluator':
        from .emerson_2017_disease_classification import Emerson2017Evaluator
        return Emerson2017Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['Emerson2017Evaluator']
