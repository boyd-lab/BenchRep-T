# Lazy imports to avoid circular import issues when running modules directly
def __getattr__(name):
    if name == 'CMV_Immunosequencing_Model':
        from .emerson_2017 import CMV_Immunosequencing_Model
        return CMV_Immunosequencing_Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['CMV_Immunosequencing_Model']
