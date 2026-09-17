import numpy as np
import pandas as pd


def permutation(df, columns=None, permute_levels=None):
    """Return a DataFrame whose values along =columns= are a random
    permutation of those in df.

    When =permute_levels= is None, the rows of df[columns] are shuffled
    along the full row axis (a complete permutation).

    When =permute_levels= is supplied, the *fixed* index levels (those
    names in df.index not in =permute_levels=) are held in place while
    coordinates of the =permute_levels= are jointly relabelled.  In
    particular, the values at every coordinate of =permute_levels= are
    reassigned, as a block across fixed levels, the values from some
    randomly chosen other coordinate of =permute_levels=.

    Earlier versions used =df.loc[:, columns].unstack(fixed).sample(
    frac=1).stack(fixed)= to realize the within-block permutation, but
    that idiom moves labels along with values and therefore never
    induces a true permutation of the data; any apparent shuffling came
    only from =stack='s lexsort and was misaligned with respect to the
    original row index.  This implementation permutes the underlying
    NumPy values while keeping the row labels of the wide pivot in
    place, which produces the intended permutation.
    """

    df = pd.DataFrame(df)  # Make sure we have a DataFrame.

    if columns is None:
        columns = df.columns
    sub = df.loc[:, columns]

    if permute_levels is None:
        return pd.DataFrame(
            np.random.permutation(sub.values), index=df.index, columns=columns
        )

    if isinstance(permute_levels, str):
        permute_levels = [permute_levels]

    fixed = [n for n in df.index.names if n not in permute_levels]
    if not fixed:
        return pd.DataFrame(
            np.random.permutation(sub.values), index=df.index, columns=columns
        )

    # Pivot so each row is one permute-level coordinate and the columns
    # enumerate fixed-level coordinates (times the original variables).
    wide = sub.unstack(fixed)

    # Shuffle values *without* moving the row labels.  This is what
    # actually realizes the permutation: each permute-level coordinate
    # is reassigned the block of values from a randomly chosen other
    # coordinate, while the alignment along the fixed levels is kept.
    shuffled = pd.DataFrame(
        np.random.permutation(wide.values), index=wide.index, columns=wide.columns
    )

    long = shuffled.stack(fixed)
    long = long.reindex(df.index)

    if isinstance(long, pd.Series):
        long = long.to_frame()
    long.columns = columns
    return long
