__doc__ = """ Boundary conditions for rod """
__all__ = [
    "FreeRod",
    "OneEndFixedRod",
    "HelicalBucklingBC",
    "GeneralConstraint",
]

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._boundary_conditions import (
        FreeRod,
        OneEndFixedRod,
        GeneralConstraint,
        HelicalBucklingBC,
        _ConstrainPeriodicBoundaries,
        _ConstrainPeriodicBoundariesMuscleRod,
    )
else:
    from elastica._elastica_numpy._boundary_conditions import (
        FreeRod,
        OneEndFixedRod,
        GeneralConstraint,
        HelicalBucklingBC,
        _ConstrainPeriodicBoundaries,
        _ConstrainPeriodicBoundariesMuscleRod,
    )
