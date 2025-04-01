__doc__ = """ Numba implementation module for boundary condition implementations that constrain or
define displacement conditions on the rod"""
__all__ = ["FreeRod", "OneEndFixedRod", "HelicalBucklingBC","GeneralConstraint"]
import numpy as np

from typing import Optional

import numba
from numba import njit

from elastica._linalg import _batch_matvec, _batch_matrix_transpose
from elastica._rotations import _get_rotation_matrix
from elastica.typing import SystemType, RodType

from elastica._elastica_numba._synchronize_functions_for_periodic_boundary._synchronize_periodic_boundary import (
    _synchronize_periodic_boundary_of_matrix_collection,
    _synchronize_periodic_boundary_of_vector_collection,
    _synchronize_periodic_boundary_of_scalar_collection,
)


class FreeRod:
    """
    This is the base class for displacement boundary conditions. It applies no constraints or displacements to the rod.

    Note
    ----
    Every new displacement boundary condition class must be
    derived from FreeRod class.
    """

    def __init__(self):
        """
        Free rod has no input parameters.
        """
        pass

    def constrain_values(self, rod, time):
        """
        Constrain values (position and/or directors) of a rod object.

        In FreeRod class, this routine simply passes.

        Parameters
        ----------
        rod : object
            Rod-like object.
        time : float
            The time of simulation.

        Returns
        -------

        """
        pass

    def constrain_rates(self, rod, time):
        """
        Constrain rates (velocity and/or omega) of a rod object.

        In FreeRod class, this routine simply passes.

        Parameters
        ----------
        rod : object
            Rod-like object.
        time : float
            The time of simulation.

        Returns
        -------

        """
        pass


class OneEndFixedRod(FreeRod):
    """
    This boundary condition class fixes one end of the rod. Currently,
    this boundary condition fixes position and directors
    at the first node and first element of the rod.

        Attributes
        ----------
        fixed_positions : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
    """

    def __init__(self, fixed_position, fixed_directors):
        """

        Parameters
        ----------
        fixed_position : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
        """
        FreeRod.__init__(self)
        self.fixed_position = fixed_position
        self.fixed_directors = fixed_directors

    def constrain_values(self, rod, time):
        # rod.position_collection[..., 0] = self.fixed_position
        # rod.director_collection[..., 0] = self.fixed_directors
        self.compute_contrain_values(
            rod.position_collection,
            self.fixed_position,
            rod.director_collection,
            self.fixed_directors,
        )

    def constrain_rates(self, rod, time):
        # rod.velocity_collection[..., 0] = 0.0
        # rod.omega_collection[..., 0] = 0.0
        self.compute_constrain_rates(rod.velocity_collection, rod.omega_collection)

    @staticmethod
    @njit(cache=True)
    def compute_contrain_values(
        position_collection, fixed_position, director_collection, fixed_directors
    ):
        """
        Computes constrain values in numba njit decorator
        Parameters
        ----------
        position_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        fixed_position : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        director_collection : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with `float` type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.

        Returns
        -------

        """
        position_collection[..., 0] = fixed_position
        director_collection[..., 0] = fixed_directors

    @staticmethod
    @njit(cache=True)
    def compute_constrain_rates(velocity_collection, omega_collection):
        """
        Compute contrain rates in numba njit decorator
        Parameters
        ----------
        velocity_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        omega_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.

        Returns
        -------

        """
        velocity_collection[..., 0] = 0.0
        omega_collection[..., 0] = 0.0


class HelicalBucklingBC(FreeRod):
    """
    This is the boundary condition class for Helical
    Buckling case in Gazzola et. al. RSoS (2018).
    The applied boundary condition is twist and slack on to
    the first and last nodes and elements of the rod.

        Attributes
        ----------
        twisting_time: float
            Time to complete twist.
        final_start_position: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Position of first node of rod after twist completed.
        final_end_position: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Position of last node of rod after twist completed.
        ang_vel: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Angular velocity of rod during twisting time.
        shrink_vel: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Shrink velocity of rod during twisting time.
        final_start_directors: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Directors of first element of rod after twist completed.
        final_end_directors: numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Directors of last element of rod after twist completed.


    """

    def __init__(
        self,
        position_start,
        position_end,
        director_start,
        director_end,
        twisting_time,
        slack,
        number_of_rotations,
    ):
        """

        Parameters
        ----------

        position_start : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Initial position of first node.
        position_end : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Initial position of last node.
        director_start : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Initial director of first element.
        director_end : numpy.ndarray
            3D (dim, dim, blocksize) array containing data with 'float' type.
            Initial director of last element.
        twisting_time : float
            Time to complete twist.
        slack : float
            Slack applied to rod.
        number_of_rotations : float
            Number of rotations applied to rod.
        """
        FreeRod.__init__(self)
        self.twisting_time = twisting_time

        angel_vel_scalar = (
            2.0 * number_of_rotations * np.pi / self.twisting_time
        ) / 2.0
        shrink_vel_scalar = slack / (self.twisting_time * 2.0)

        direction = (position_end - position_start) / np.linalg.norm(
            position_end - position_start
        )

        self.final_start_position = position_start + slack / 2.0 * direction
        self.final_end_position = position_end - slack / 2.0 * direction

        self.ang_vel = angel_vel_scalar * direction
        self.shrink_vel = shrink_vel_scalar * direction

        theta = number_of_rotations * np.pi

        self.final_start_directors = (
            _get_rotation_matrix(theta, direction.reshape(3, 1)).reshape(3, 3)
            @ director_start
        )  # rotation_matrix wants vectors 3,1
        self.final_end_directors = (
            _get_rotation_matrix(-theta, direction.reshape(3, 1)).reshape(3, 3)
            @ director_end
        )  # rotation_matrix wants vectors 3,1

    def constrain_values(self, rod, time):
        if time > self.twisting_time:
            rod.position_collection[..., 0] = self.final_start_position
            rod.position_collection[..., -1] = self.final_end_position

            rod.director_collection[..., 0] = self.final_start_directors
            rod.director_collection[..., -1] = self.final_end_directors

    def constrain_rates(self, rod, time):
        if time > self.twisting_time:
            rod.velocity_collection[..., 0] = 0.0
            rod.omega_collection[..., 0] = 0.0

            rod.velocity_collection[..., -1] = 0.0
            rod.omega_collection[..., -1] = 0.0

        else:
            rod.velocity_collection[..., 0] = self.shrink_vel
            rod.omega_collection[..., 0] = self.ang_vel

            rod.velocity_collection[..., -1] = -self.shrink_vel
            rod.omega_collection[..., -1] = -self.ang_vel


class _ConstrainPeriodicBoundaries(FreeRod):
    """
    This class is used only when ring rods are present in the simulation. This class is a wrapper and its purpose
    is to synchronize periodic boundaries of ring rod.
    """

    def __init__(self):
        pass

    def constrain_values(self, rod, time):
        _synchronize_periodic_boundary_of_vector_collection(
            rod.position_collection, rod.periodic_boundary_nodes_idx
        )
        _synchronize_periodic_boundary_of_matrix_collection(
            rod.director_collection, rod.periodic_boundary_elems_idx
        )

    def constrain_rates(self, rod, time):
        _synchronize_periodic_boundary_of_vector_collection(
            rod.velocity_collection, rod.periodic_boundary_nodes_idx
        )
        _synchronize_periodic_boundary_of_vector_collection(
            rod.omega_collection, rod.periodic_boundary_elems_idx
        )


class _ConstrainPeriodicBoundariesMuscleRod(_ConstrainPeriodicBoundaries):
    """
    This class is used only when muscular ring rods are present in the simulation. This class is a wrapper and its
     purpose is to synchronize periodic boundaries of muscular ring rod.
    """

    def __init__(self):
        pass

    def constrain_values(self, rod, time):
        _synchronize_periodic_boundary_of_vector_collection(
            rod.position_collection, rod.periodic_boundary_nodes_idx
        )
        _synchronize_periodic_boundary_of_matrix_collection(
            rod.director_collection, rod.periodic_boundary_elems_idx
        )
        _synchronize_periodic_boundary_of_scalar_collection(
            rod.fiber_activation, rod.periodic_boundary_elems_idx
        )



class GeneralConstraint(FreeRod):
    """
    This boundary condition class allows the specified node/link to have a configurable constraint.
    Index can be passed to fix either or both the position or the director.
    Constraining position is equivalent to setting 0 translational DOF.
    Constraining director is equivalent to setting 0 rotational DOF.

    Examples
    --------
    How to fix all translational and rotational dof except allowing twisting around the z-axis in an inertial frame:

    >>> simulator.constrain(system).using(
    ...    GeneralConstraint,
    ...    constrained_position_idx=(0,),
    ...    constrained_director_idx=(0,),
    ...    translational_constraint_selector=np.array([True, True, True]),
    ...    rotational_constraint_selector=np.array([True, True, False]),
    ... )

    How to allow the end of the rod to move in the XY plane and allow all rotational dof:

    >>> simulator.constrain(rod).using(
    ...    GeneralConstraint,
    ...    constrained_position_idx=(-1,),
    ...    translational_constraint_selector=np.array([True, True, False]),
    ... )
    """

    def __init__(
        self,
        *fixed_data,
        translational_constraint_selector: Optional[np.ndarray] = None,
        rotational_constraint_selector: Optional[np.array] = None,
        rod_frame_bool: Optional[bool] = None,
        **kwargs,
    ):
        """

        Initialization of the constraint. Any parameter passed to 'using' will be available in kwargs.

        Parameters
        ----------
        constrained_position_idx : tuple
            Tuple of position-indices that will be constrained
        constrained_director_idx : tuple
            Tuple of director-indices that will be constrained
        translational_constraint_selector: Optional[np.ndarray]
            np.array of type bool indicating which translational degrees of freedom (dof) to constrain.
            If entry is True, the corresponding dof will be constrained. If None, we constrain all dofs.
        rotational_constraint_selector: Optional[np.ndarray]
            np.array of type bool indicating which translational degrees of freedom (dof) to constrain.
            If entry is True, the corresponding dof will be constrained.
        """
        super().__init__(**kwargs)
        pos, dir = [], []
        for data in fixed_data:
            if isinstance(data, np.ndarray) and data.shape == (3,):
                pos.append(data)
            elif isinstance(data, np.ndarray) and data.shape == (
                3,
                3,
            ):
                dir.append(data)
            else:
                # TODO: This part is prone to error.
                break

        if len(pos) > 0:
            # transpose from (blocksize, dim) to (dim, blocksize)
            self.fixed_positions = np.array(pos).transpose((1, 0))

        if len(dir) > 0:
            # transpose from (blocksize, dim, dim) to (dim, dim, blocksize)
            self.fixed_directors = np.array(dir).transpose((1, 2, 0))

        if translational_constraint_selector is None:
            translational_constraint_selector = np.array([True, True, True])
        if rotational_constraint_selector is None:
            rotational_constraint_selector = np.array([True, True, True])

        if rod_frame_bool is None:
            rod_frame_bool = False

        assert isinstance(
            rod_frame_bool, bool), "rod_frame_bool must be a boolean value (True if the BCs are defined with respect to the rod frame)."

            

        assert (
            type(translational_constraint_selector) == np.ndarray
            and translational_constraint_selector.dtype == bool
            and translational_constraint_selector.shape == (3,)
        ), "Translational constraint selector must be a 1D boolean array of length 3."
        assert (
            type(rotational_constraint_selector) == np.ndarray
            and rotational_constraint_selector.dtype == bool
            and rotational_constraint_selector.shape == (3,)
        ), "Rotational constraint selector must be a 1D boolean array of length 3."


        # cast booleans to int
        self.translational_constraint_selector = (
            translational_constraint_selector.astype(int)
        )
        self.rotational_constraint_selector = rotational_constraint_selector.astype(int)
        self.rod_frame_bool = rod_frame_bool
        self.constrained_position_idx = np.array(kwargs.get("constrained_position_idx", []), dtype=int)
        self.constrained_director_idx = np.array(kwargs.get("constrained_director_idx", []), dtype=int)

    def constrain_values(self, rod, time: float) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_values(
                rod.director_collection,
                rod.position_collection,
                self.fixed_positions,
                self.constrained_position_idx,
                self.translational_constraint_selector,
                self.rod_frame_bool,
            )

    def constrain_rates(self, rod, time: float) -> None:
        if self.constrained_position_idx.size:
            self.nb_constrain_translational_rates(                
                rod.director_collection,
                rod.velocity_collection,
                self.constrained_position_idx,
                self.translational_constraint_selector,
                self.rod_frame_bool,
            )
        if self.constrained_director_idx.size:
            self.nb_constrain_rotational_rates(
                rod.director_collection,
                rod.omega_collection,
                self.constrained_director_idx,
                self.rotational_constraint_selector,
                self.rod_frame_bool,
            )
    @staticmethod
    @njit(cache=True)
    def nb_constrain_translational_values(
        director_collection, position_collection, fixed_position_collection, indices, constraint_selector, rod_frame_bool
    ) -> None:
        """
        Computes constrained position values in the local (material) frame.

        Parameters
        ----------
        director_collection : numpy.ndarray
            3D (3, 3, blocksize) array containing directors for frame transformation.
        position_collection : numpy.ndarray
            2D (3, blocksize) array containing position data in the lab frame.
        fixed_position_collection : numpy.ndarray
            2D (3, blocksize) array containing target positions.
        indices : numpy.ndarray
            1D array containing the indices of constraining nodes.
        constraint_selector: numpy.ndarray
            1D array (3,) indicating which translational DoFs to constrain.
        rod_frame_bool : bool
            Whether constraints are applied in the rod frame (local frame).
        """

        if rod_frame_bool:
            # Select the relevant directors
            directors = director_collection[..., indices]

            # Convert positions to local frame
            position_local = _batch_matvec(_batch_matrix_transpose(directors), position_collection[..., indices])
            fixed_position_local = _batch_matvec(_batch_matrix_transpose(directors), fixed_position_collection)

            # Apply constraints in the local frame
            position_local = (1 - constraint_selector[:, np.newaxis]) * position_local + constraint_selector[:, np.newaxis] * fixed_position_local

            # Convert back to lab frame
            position_collection[..., indices] = _batch_matvec(directors, position_local)

        else:
            # Apply constraints directly in the lab frame
            position_collection[..., indices] = (
                (1 - constraint_selector[:, np.newaxis]) * position_collection[..., indices]
                + constraint_selector[:, np.newaxis] * fixed_position_collection
            )

    @staticmethod
    @njit(cache=True)
    def nb_constrain_translational_rates(
        director_collection, velocity_collection, indices, constraint_selector, rod_frame_bool
    ) -> None:
        """
        Compute constrained velocity rates in the local (material) frame.

        Parameters
        ----------
        director_collection : numpy.ndarray
            3D (3, 3, blocksize) array containing directors for frame transformation.
        velocity_collection : numpy.ndarray
            2D (3, blocksize) array containing velocity data in the lab frame.
        indices : numpy.ndarray
            1D array containing the indices of constraining nodes.
        constraint_selector: numpy.ndarray
            1D array (3,) indicating which translational DoFs to constrain.
        rod_frame_bool : bool
            Whether constraints are applied in the rod frame (local frame).
        """

        if rod_frame_bool:
            # Select the relevant directors
            directors = director_collection[..., indices]

            # Convert velocities to local frame
            velocity_local = _batch_matvec(_batch_matrix_transpose(directors), velocity_collection[..., indices])

            # Apply constraints in the local frame
            velocity_local = (1 - constraint_selector[:, np.newaxis]) * velocity_local

            # Convert back to lab frame
            velocity_collection[..., indices] = _batch_matvec(directors, velocity_local)

        else:
            # Apply constraints directly in the lab frame
            velocity_collection[..., indices] = (
                (1 - constraint_selector[:, np.newaxis]) * velocity_collection[..., indices]
            )

    @staticmethod
    #@njit(cache=True)
    def nb_constrain_rotational_rates(
        director_collection, omega_collection, indices, constraint_selector
    ) -> None:
        """
        Compute constrain rates in numba njit decorator

        Parameters
        ----------
        director_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        omega_collection : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        indices : numpy.ndarray
            1D array containing the index of constraining nodes
        constraint_selector: numpy.ndarray
            1D array of type int and size (3,) indicating which rotational Degrees of Freedom (DoF) to constrain.
            Entries are integers in {0, 1} (e.g. a binary values of either 0 or 1).
            If an entry is 1, the rotation around the respective axis will be constrained,
            otherwise the system can freely rotate around the axis.
            The selector shall be specified in the lab frame
        """
        directors = director_collection[..., indices]

        # rotate angular velocities to lab frame
        omega_collection_lab_frame = _batch_matvec(
            _batch_matrix_transpose(directors), omega_collection[..., indices]
        )

        # apply constraint selector to angular velocities in lab frame
        omega_collection_not_constrained = (
            1 - np.expand_dims(constraint_selector, 1)
        ) * omega_collection_lab_frame

        # rotate angular velocities vector back to local frame and apply to omega_collection
        omega_collection[..., indices] = _batch_matvec(
            directors, omega_collection_not_constrained
        )
