"""Complete-case filtering and encoding for concatenated covariates."""

import numpy as np


DEFAULT_FEATURES = ('age', 'sex', 'ancestry')


def complete_cases(data, features=DEFAULT_FEATURES):
    missing = set(features).difference(data.columns)
    if missing:
        raise ValueError(f"Missing demographic columns: {sorted(missing)}")
    filtered = data.dropna(subset=list(features)).copy()
    for column in features:
        if column != 'age':
            filtered = filtered[filtered[column].astype(str).str.strip() != '']
    return filtered


def encode_demographics(data, features=DEFAULT_FEATURES, ancestry_categories=None):
    """Encode selected covariates; reuse training categories for held-out rows.

    The default encoding is identical to the historical concatenation runs.
    Rawat can explicitly select age and sex without supplying ancestry.
    """
    if not features or len(set(features)) != len(features) or set(features).difference(DEFAULT_FEATURES):
        raise ValueError('Select distinct features from age, sex, ancestry')
    arrays, names = [], []
    categories = []
    for column in features:
        if column == 'age':
            arrays.append(data['age'].to_numpy(dtype=float)[:, None])
            names.append('age')
        elif column == 'sex':
            arrays.append((data['sex'] == 'M').to_numpy(dtype=float)[:, None])
            names.append('sex')
        else:
            categories = (sorted(data['ancestry'].unique().tolist())
                          if ancestry_categories is None else list(ancestry_categories))
            arrays.append(np.column_stack([(data['ancestry'] == c).to_numpy(dtype=float)
                                            for c in categories]) if categories else
                          np.empty((len(data), 0)))
            names.extend(f'ancestry_{c}' for c in categories)
    return np.hstack(arrays), categories, names
