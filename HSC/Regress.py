import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import CategoricalDtype
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
from scanpy.get import _get_obs_rep, _set_obs_rep
from scanpy._settings import settings as sett
from scanpy import logging as logg
from scanpy._utils import (
    sanitize_anndata,
    deprecated_arg_names,
    view_to_actual,
    AnyRandom,
    _check_array_function_arguments,
)
from typing import Union, Optional, Tuple, Collection, Sequence, Iterable, Literal
def regress(
    adata: AnnData,
    keys: Union[str, Sequence[str]],
    layer: Optional[str] = None,
    n_jobs: Optional[int] = None,
    copy: bool = False,
    add_intercept: bool = False
) -> Optional[AnnData]:
    """\
    Regress out (mostly) unwanted sources of variation.

    Uses simple linear regression. This is inspired by Seurat's `regressOut`
    function in R [Satija15]. Note that this function tends to overcorrect
    in certain circumstances as described in :issue:`526`.

    Parameters
    ----------
    adata
        The annotated data matrix.
    keys
        Keys for observation annotation on which to regress.
    layer
        If provided, which element of layers to use in regression.
    n_jobs
        Number of jobs for parallel computation.
        `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
    copy
        Determines whether a copy of `adata` is returned.
    add_intercept
        If True, regress_out will add intercept back to residuals in order to transform results back into gene-count space. Defaults to False

    Returns
    -------
    Depending on `copy` returns or updates `adata` with the corrected data matrix.
    """
    start = logg.info(f"regressing out {keys}")
    adata = adata.copy() if copy else adata

    sanitize_anndata(adata)

    view_to_actual(adata)

    if isinstance(keys, str):
        keys = [keys]

    X = _get_obs_rep(adata, layer=layer)

    if issparse(X):
        logg.info("    sparse input is densified and may " "lead to high memory use")
        X = X.toarray()

    n_jobs = sett.n_jobs if n_jobs is None else n_jobs

    # regress on a single categorical variable
    variable_is_categorical = False
    if keys[0] in adata.obs_keys() and isinstance(
        adata.obs[keys[0]].dtype, CategoricalDtype
    ):
        if len(keys) > 1:
            raise ValueError(
                "If providing categorical variable, "
                "only a single one is allowed. For this one "
                "we regress on the mean for each category."
            )
        logg.debug("... regressing on per-gene means within categories")
        regressors = np.zeros(X.shape, dtype="float32")
        for category in adata.obs[keys[0]].cat.categories:
            mask = (category == adata.obs[keys[0]]).values
            for ix, x in enumerate(X.T):
                regressors[mask, ix] = x[mask].mean()
        variable_is_categorical = True
    # regress on one or several ordinal variables
    else:
        # create data frame with selected keys (if given)
        if keys:
            regressors = adata.obs[keys]
        else:
            regressors = adata.obs.copy()

        # add column of ones at index 0 (first column)
        regressors.insert(0, "ones", 1.0)

    len_chunk = np.ceil(min(1000, X.shape[1]) / n_jobs).astype(int)
    n_chunks = np.ceil(X.shape[1] / len_chunk).astype(int)

    tasks = []
    # split the adata.X matrix by columns in chunks of size n_chunk
    # (the last chunk could be of smaller size than the others)
    chunk_list = np.array_split(X, n_chunks, axis=1)
    if variable_is_categorical:
        regressors_chunk = np.array_split(regressors, n_chunks, axis=1)
    for idx, data_chunk in enumerate(chunk_list):
        # each task is a tuple of a data_chunk eg. (adata.X[:,0:100]) and
        # the regressors. This data will be passed to each of the jobs.
        if variable_is_categorical:
            regres = regressors_chunk[idx]
        else:
            regres = regressors
        tasks.append(tuple((data_chunk, regres, variable_is_categorical)))

    from joblib import Parallel, delayed

    # TODO: figure out how to test that this doesn't oversubscribe resources
    res = Parallel(n_jobs=n_jobs)(delayed(_regress_out_chunk)(task) for task in tasks)

    # res is a list of vectors (each corresponding to a regressed gene column).
    # The transpose is needed to get the matrix in the shape needed
    _set_obs_rep(adata, np.vstack(res).T, layer=layer)
    logg.info("    finished", time=start)
    return adata if copy else None


def _regress_out_chunk(data, add_intercept = True):
    # data is a tuple containing the selected columns from adata.X
    # and the regressors dataFrame
    data_chunk = data[0]
    regressors = data[1]
    variable_is_categorical = data[2]

    responses_chunk_list = []
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    for col_index in range(data_chunk.shape[1]):
        # if all values are identical, the statsmodel.api.GLM throws an error;
        # but then no regression is necessary anyways...
        if not (data_chunk[:, col_index] != data_chunk[0, col_index]).any():
            responses_chunk_list.append(data_chunk[:, col_index])
            continue

        if variable_is_categorical:
            regres = np.c_[np.ones(regressors.shape[0]), regressors[:, col_index]]
        else:
            regres = regressors
        try:
            # print(regres)
            # print(data_chunk[:, col_index])
            if add_intercept:
                result = sm.GLM(
                    data_chunk[:, col_index], regres, family=sm.families.Gaussian()
                ).fit()
                new_column = result.resid_response + result.params[0]
            else:
                result = sm.GLM(
                    data_chunk[:, col_index], regres, family=sm.families.Gaussian()
                ).fit()
                new_column = result.resid_response
        except PerfectSeparationError:  # this emulates R's behavior
            logg.warning("Encountered PerfectSeparationError, setting to 0 as in R.")
            new_column = np.zeros(data_chunk.shape[0])

        responses_chunk_list.append(new_column)

    return np.vstack(responses_chunk_list)
