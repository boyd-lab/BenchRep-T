# Lazy imports to avoid circular import issues when running modules directly
def __getattr__(name):
    if name == 'CMV_Immunosequencing_Model':
        from .emerson_2017 import CMV_Immunosequencing_Model
        return CMV_Immunosequencing_Model
    if name == 'GIANA_Classifier':
        from .giana_2020 import GIANA_Classifier
        return GIANA_Classifier
    if name == 'Gapped_4mer_VJgene':
        from .ml_baseline import Gapped_4mer_VJgene
        return Gapped_4mer_VJgene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['CMV_Immunosequencing_Model', 'GIANA_Classifier', 'Gapped_4mer_VJgene']
