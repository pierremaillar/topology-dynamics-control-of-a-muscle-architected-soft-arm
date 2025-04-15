__doc__ = """ Ribbon model equations implementation for Elastica Numpy/Numba implementation. The model keep the same kinematic relation as the rod model but consist of an adapted constitutive law to account for the "plate-like" behavior. The model is derived and detailed in: "A one-dimensional model for elastic ribbons: a little stretching makes a big difference " by Basile Audoly and Sebastien Neukirch"""

__all__ = ["Ribbon1D"]
import numpy as np
import functools
import numba
from elastica.rod import RodBase
from elastica._elastica_numba._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from elastica._elastica_numba._rotations import _inv_rotate
from elastica.rod.factory_function import allocate_ribbon
from elastica._calculus import (
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
    _difference,
    _average,
)
from elastica._elastica_numba._interaction import node_to_element_pos_or_vel
from elastica.utils import Tolerance

position_difference_kernel = _difference
position_average = _average


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


def _compute_sigma_kappa_for_blockstructure(memory_block):
    """
    This function is a wrapper to call functions which computes shear stretch, strain and bending twist and strain.

    Parameters
    ----------
    memory_block : object

    Returns
    -------

    """


    _compute_shear_stretch_strains(
        memory_block.position_collection,
        memory_block.volume,
        memory_block.lengths,
        memory_block.tangents,
        memory_block.thickness,
        memory_block.width,
        memory_block.rest_lengths,
        memory_block.rest_voronoi_lengths,
        memory_block.dilatation,
        memory_block.voronoi_dilatation,
        memory_block.director_collection,
        memory_block.sigma,
    )

    # Compute bending twist strains for the block
    _compute_bending_twist_strains(
        memory_block.director_collection,
        memory_block.rest_voronoi_lengths,
        memory_block.kappa,
    )


class Ribbon1D(RodBase):
    def __init__(
        self,
        n_elements,
        position,
        velocity,
        omega,
        acceleration,
        angular_acceleration,
        directors,
        thickness,
        width,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_constants,
        density,
        volume,
        mass,
        dissipation_constant_for_forces,
        dissipation_constant_for_torques,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
        damping_forces,
        damping_torques,
        phi,
        phi_p,
        args,
        kwargs,
    ):
        self.n_elems = n_elements
        self.position_collection = position
        self.velocity_collection = velocity
        self.omega_collection = omega
        self.acceleration_collection = acceleration
        self.alpha_collection = angular_acceleration
        self.director_collection = directors
        self.thickness = thickness
        self.width = width
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.shear_matrix = shear_matrix
        self.bend_constants = bend_constants
        self.density = density
        self.volume = volume
        self.mass = mass
        self.dissipation_constant_for_forces = dissipation_constant_for_forces
        self.dissipation_constant_for_torques = dissipation_constant_for_torques
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.rest_lengths = rest_lengths
        self.tangents = tangents
        self.dilatation = dilatation
        self.dilatation_rate = dilatation_rate
        self.voronoi_dilatation = voronoi_dilatation
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple
        self.damping_forces = damping_forces
        self.damping_torques = damping_torques
        self.phi = phi
        self.phi_p = phi_p


        # rest base area
        #self.rest_area = self.width * self.thickness

        # if kwargs.__contains__("stretch_optimal"):
        #     self.stretch_optimal = np.ones((n_elements)) * kwargs.get("stretch_optimal")
        # else:
        #     # raise AttributeError("Did you forget to input stretch_optimal in kwargs ?")
        #     self.stretch_optimal = np.ones((n_elements))


    @classmethod
    def straight_ribbon(
        cls,
        n_elements,
        start,
        direction,
        normal,
        base_length,
        base_thickness,
        base_width,
        density,
        nu,
        youngs_modulus,
        shear_modulus,
        poisson_ratio,
        alpha_c=4.0/3.0,
        *args,
        **kwargs,
    ):
        (
            n_elements,
            position,
            velocity,
            omega,
            acceleration,
            angular_acceleration,
            directors,
            thickness,
            width,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            bend_constants,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
            phi,
            phi_p,
            args,
            kwargs,
        ) = allocate_ribbon(
            n_elements,
            start,
            direction,
            normal,
            base_length,
            base_thickness,
            base_width,
            density,
            nu,
            youngs_modulus,
            shear_modulus,
            poisson_ratio,
            alpha_c=4.0/3.0,
            *args,
            **kwargs,
        )

        return cls(
            n_elements,
            position,
            velocity,
            omega,
            acceleration,
            angular_acceleration,
            directors,
            thickness,
            width,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_constants,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
            phi,
            phi_p,
            args,
            kwargs,
        )


    def compute_internal_forces_and_torques(self, time):
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.
        Parameters
        ----------
        time

        Returns
        -------

        """
        
        _compute_internal_forces(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.thickness,
            self.width,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.dilatation_rate,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
            self.rest_sigma,
            self.kappa,
            self.rest_kappa,
            self.shear_matrix,
            self.bend_constants,
            self.internal_stress,
            self.velocity_collection,
            self.dissipation_constant_for_forces,
            self.damping_forces,
            self.internal_forces,
            self.ghost_elems_idx,
        )


        _compute_internal_torques(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.bend_constants,
            self.rest_kappa,
            self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.dissipation_constant_for_torques,
            self.damping_torques,
            self.internal_torques,
            self.ghost_elems_idx,
            self.volume,
            self.sigma,
            self.rest_sigma,
            self.phi,
            self.phi_p,
        )

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time):
        """
        This class method function is only a wrapper to call Numba njit function, which
        updates the acceleration

        Parameters
        ----------
        time

        Returns
        -------

        """
        _update_accelerations(
            self.acceleration_collection,
            self.internal_forces,
            self.external_forces,
            self.mass,
            self.alpha_collection,
            self.inv_mass_second_moment_of_inertia,
            self.internal_torques,
            self.external_torques,
            self.dilatation,
        )

    def zeroed_out_external_forces_and_torques(self, time):
        _zeroed_out_external_forces_and_torques(
            self.external_forces, self.external_torques
        )

    def compute_translational_energy(self):
        return (
            0.5
            * (
                self.mass
                * np.einsum(
                    "ij, ij-> j", self.velocity_collection, self.velocity_collection
                )
            ).sum()
        )

    def compute_rotational_energy(self):
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self):
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self):
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()

    def compute_bending_energy(self):
        kappa_diff = self.kappa - self.rest_kappa

        return 0

    def compute_shear_energy(self):
        sigma_diff = self.sigma - self.rest_sigma
        shear_internal_torques = _batch_matvec(self.shear_matrix, sigma_diff)

        return (
            0.5
            * (_batch_dot(sigma_diff, shear_internal_torques) * self.rest_lengths).sum()
        )


@numba.njit(cache=True)
def _compute_geometry_from_state(
    position_collection, volume, lengths, tangents, thickness, width
):
    """
    Returns
    -------

    """
    # Compute eq (3.3) from 2018 RSOS paper

    # Note : we can use the two-point difference kernel, but it needs unnecessary padding
    # and hence will always be slower
    position_diff = position_difference_kernel(position_collection)
    # FIXME: Here 1E-14 is added to fix ghost lengths, which is 0, and causes division by zero error!
    lengths[:] = _batch_norm(position_diff) + 1e-14
    # _reset_scalar_ghost(lengths, ghost_elems_idx, 1.0)

    
    for k in range(lengths.shape[0]):
        tangents[0, k] = position_diff[0, k] / lengths[k]
        tangents[1, k] = position_diff[1, k] / lengths[k]
        tangents[2, k] = position_diff[2, k] / lengths[k]
        # resize based on volume conservation
        # Here we assume that the deformation du to volumne conservation will strech only the thickness but conserve the width !
        thickness[k] = np.sqrt(volume[k] / lengths[k] / width[k])



@numba.njit(cache=True)
def _compute_all_dilatations(
    position_collection,
    volume,
    lengths,
    tangents,
    thickness,
    width,
    dilatation,
    rest_lengths,
    rest_voronoi_lengths,
    voronoi_dilatation,
):
    """
    Compute element and Voronoi region dilatations
    Returns
    -------

    """
    _compute_geometry_from_state(position_collection, volume, lengths, tangents, thickness, width)
    # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
    # Put in initialization
    for k in range(lengths.shape[0]):
        dilatation[k] = lengths[k] / rest_lengths[k]
    # Compute eq (3.4) from 2018 RSOS paper
    # Note : we can use trapezoidal kernel, but it has padding and will be slower
    voronoi_lengths = position_average(lengths)

    # Compute eq (3.45 from 2018 RSOS paper
    for k in range(voronoi_lengths.shape[0]):
        voronoi_dilatation[k] = voronoi_lengths[k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)
def _compute_dilatation_rate(
    position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
):
    """

    Returns
    -------

    """
    # TODO Use the vector formula rather than separating it out
    # self.lengths = l_i = |r^{i+1} - r^{i}|
    r_dot_v = _batch_dot(position_collection, velocity_collection)
    r_plus_one_dot_v = _batch_dot(
        position_collection[..., 1:], velocity_collection[..., :-1]
    )
    r_dot_v_plus_one = _batch_dot(
        position_collection[..., :-1], velocity_collection[..., 1:]
    )

    blocksize = lengths.shape[0]

    for k in range(blocksize):
        dilatation_rate[k] = (
            (r_dot_v[k] + r_dot_v[k + 1] - r_dot_v_plus_one[k] - r_plus_one_dot_v[k])
            / lengths[k]
            / rest_lengths[k]
        )


@numba.njit(cache=True)
def _compute_shear_stretch_strains(
    position_collection,
    volume,
    lengths,
    tangents,
    thickness,
    width,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    voronoi_dilatation,
    director_collection,
    sigma,
):

    # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
    _compute_all_dilatations(
        position_collection,
        volume,
        lengths,
        tangents,
        thickness,
        width,
        dilatation,
        rest_lengths,
        rest_voronoi_lengths,
        voronoi_dilatation,
    )

    z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
    sigma[:] = dilatation * _batch_matvec(director_collection, tangents) - z_vector



@numba.njit(cache=True)
def _compute_internal_shear_stretch_stresses_from_model(
    position_collection,
    volume,
    lengths,
    tangents,
    thickness,
    width,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    dilatation_rate,
    voronoi_dilatation,
    director_collection,
    sigma,
    rest_sigma,
    kappa,
    rest_kappa,
    shear_matrix,
    bend_constants,
    internal_stress,
):
    """
    Relation between internal strain (shear/strech) and internal forces. This relation is essential for the numerical solver used in elastica but is inforcing nmerically the lagrangian mutiplier of the constrain: dr/ds = d3 for the 1D ribbon hybrid model.
    
    Linear force functional
    Operates on
    S : (3,3,n) tensor and sigma (3,n)
    
    Returns
    -------
    
    """
    
    _compute_shear_stretch_strains(
        position_collection,
        volume,
        lengths,        
        tangents,
        thickness,
        width,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
        director_collection,
        sigma,
    )

    #Note: Not sure this is efficient and is not entirely true but is needed as kappa is compute between element
    #and not for every element (nbr of element - 1)
    #kappa_padded = np.hstack((kappa-rest_kappa, np.zeros((3, 1))))


    #internal_stress[0,:] = shear_matrix[0,0,:]*(sigma[0,:]-rest_sigma[0,:])
    #internal_stress[1,:] = shear_matrix[1,1,:]*(sigma[1,:]-rest_sigma[1,:])
    #internal_stress[2,:] = (2*bend_constants[0,0]*((sigma[2,:]-rest_sigma[2,:])+bend_constants[1,2]*(kappa_padded[2,:])**2))

        
    internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)



@numba.njit(cache=True)
def _compute_bending_twist_strains(director_collection, rest_voronoi_lengths, kappa):
    temp = _inv_rotate(director_collection)
    blocksize = rest_voronoi_lengths.shape[0]
    for k in range(blocksize):
        kappa[0, k] = temp[0, k] / rest_voronoi_lengths[k]
        kappa[1, k] = temp[1, k] / rest_voronoi_lengths[k]
        kappa[2, k] = temp[2, k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)
def _compute_internal_bending_twist_stresses_from_model(
    position_collection,
    director_collection,
    rest_voronoi_lengths,
    internal_couple,
    bend_constants,
    kappa,
    rest_kappa,
    volume,
    sigma,
    rest_sigma,
    phi,
    phi_p,
):
    """
    Linear force functional
    Operates on
    B : (3,3,n) tensor and curvature kappa (3,n)

    Returns
    -------

    """
    _compute_bending_twist_strains(
        director_collection, rest_voronoi_lengths, kappa
    )  # concept : needs to compute kappa

    
    _compute_phi_and_phiprime(bend_constants, kappa, phi, phi_p)

    blocksize = kappa.shape[1]
    k2_temp = 0
    k3_temp = 0

    #Note: build for uniform ribbon (constant width, thickness, Youngs modulus)
    [[A, B, C],[ D, E, F],[ poisson_ratio, OneOver_kStar,_]] = bend_constants[:,:,0]
    
    for k in range(blocksize):
        k2_temp = kappa[1, k] - rest_kappa[1, k]
        k3_temp = kappa[2, k] - rest_kappa[2, k]

        #NOTE: Can be optimized if needed 
        internal_couple[0, k] = 2*B*(kappa[0, k] - rest_kappa[0, k])
        internal_couple[1, k] = 2*C*k2_temp + 4*E*(poisson_ratio*k2_temp**2+k3_temp**2)*poisson_ratio*phi[k]*k2_temp+E*(poisson_ratio*k2_temp**2+k3_temp**2)**2*phi_p[k]*OneOver_kStar
        internal_couple[2, k] = 2*D*k3_temp+4*E*(poisson_ratio*k2_temp**2+k3_temp**2)*k3_temp*phi[k]

@numba.njit(cache=True)
def _compute_phi_and_phiprime(
    bend_constants,
    kappa,
    phi,
    phi_p,
):
    """
    Compute phi(k2) and phi_p(k2) with respect to k2 following the piece-wise approximation 
    from Audoly et al.: "A one-dimensional model for elastic ribbons: a little stretching makes a big difference."

    Operates on
    curvature kappa (3, n)

    Parameters
    ----------
    bend_constants : array
        Material and geometric constants.
    kappa : array
        Curvature tensor (3, n).
    phi : array
        Output array for phi values.
    phi_p : array
        Output array for phi_p values.

    Returns
    -------
    None (modifies phi and phi_p in place)
    """
    kappa2b = kappa[1, :] * bend_constants[2, 1, 0]
    kappa2b2 = kappa2b ** 2
    kappa2b4 = kappa2b2 ** 2
    abs_kappa2b = np.abs(kappa2b)
    sgn_kappa2b = np.sign(kappa2b)

    # Small kappa2b (|kappa2b| < 0.3)
    mask_small = abs_kappa2b < 0.3
    phi[mask_small] = (
        0.002777777777777778
        + (-5.5114638447971785e-6) * kappa2b2[mask_small]
        + (1.1008092191954626e-8) * kappa2b4[mask_small]
    )
    phi_p[mask_small] = kappa2b[mask_small] * (
        2 * (-5.5114638447971785e-6) + 4 * (1.1008092191954626e-8) * kappa2b2[mask_small]
    )

    # Large kappa2b (|kappa2b| > 1800)
    mask_large = abs_kappa2b > 1800
    sqrt_abs_kappa2b = np.sqrt(abs_kappa2b[mask_large])
    phi[mask_large] = (-5.656854249492381) / (
        sqrt_abs_kappa2b * kappa2b2[mask_large]
    ) + 2 / kappa2b2[mask_large]
    phi_p[mask_large] = (
        -2.5
        * sgn_kappa2b[mask_large]
        * (-5.656854249492381)
        / (kappa2b4[mask_large] * sqrt_abs_kappa2b)
        - 4 / (kappa2b[mask_large] * kappa2b2[mask_large])
    )

    # Intermediate kappa2b (0.3 ≤ |kappa2b| ≤ 1800)
    mask_mid = ~(mask_small | mask_large)
    k_mid = kappa2b[mask_mid]
    q = np.sqrt(np.abs(k_mid) / 2)
    q2 = q**2
    q3 = q * q2
    q5, q6 = q2 * q3, q3**2

    cosh_q, sinh_q = np.cosh(q), np.sinh(q)
    cos_q, sin_q = np.cos(q), np.sin(q)

    cosh_q2, cos_q2 = cosh_q**2, cos_q**2
    cos_q3 = cos_q * cos_q2

    sn_sum = sinh_q + sin_q
    sn_sum2 = sn_sum**2

    f0 = (0.5 * (-2 * cosh_q + 2 * cos_q + q * sn_sum)) / (q5 * sn_sum)
    f1 = (
        cosh_q2 * q
        - cos_q2 * q
        + 5 * cosh_q * sn_sum
        - 5 * cos_q * sn_sum
        - 3 * q * sn_sum2
    ) / (q6 * sn_sum2)

    phi[mask_mid] = f0
    phi_p[mask_mid] = f1 * sgn_kappa2b[mask_mid] / (4 * q)
    

    
@numba.njit(cache=True)
def _compute_damping_forces(
    damping_forces,
    velocity_collection,
    dissipation_constant_for_forces,
    lengths,
    ghost_elems_idx,
):
    # Internal damping foces.
    elemental_velocities = node_to_element_pos_or_vel(velocity_collection)

    blocksize = elemental_velocities.shape[1]
    elemental_damping_forces = np.zeros((3, blocksize))

    for i in range(3):
        for k in range(blocksize):
            elemental_damping_forces[i, k] = (
                dissipation_constant_for_forces[k]
                * elemental_velocities[i, k]
                * lengths[k]
            )

    damping_forces[:] = quadrature_kernel_for_block_structure(
        elemental_damping_forces, ghost_elems_idx
    )


@numba.njit(cache=True)
def _compute_internal_forces(
    position_collection,
    volume,
    lengths,
    tangents,
    thickness,
    width,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    dilatation_rate,
    voronoi_dilatation,
    director_collection,
    sigma,
    rest_sigma,
    kappa,
    rest_kappa,
    shear_matrix,
    bend_constants,
    internal_stress,
    velocity_collection,
    dissipation_constant_for_forces,
    damping_forces,
    internal_forces,
    ghost_elems_idx,
):
    # Compute n_l and cache it using internal_stress
    # Be careful about usage though
    _compute_internal_shear_stretch_stresses_from_model(
        position_collection,
        volume,
        lengths,
        tangents,
        thickness,
        width,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        director_collection,
        sigma,
        rest_sigma,
        kappa,
        rest_kappa,
        shear_matrix,
        bend_constants,
        internal_stress,
    )

    # Signifies Q^T n_L / e
    # Not using batch matvec as I don't want to take directors.T here

    blocksize = internal_stress.shape[1]
    cosserat_internal_stress = np.zeros((3, blocksize))

    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                cosserat_internal_stress[i, k] += (
                    director_collection[j, i, k] * internal_stress[j, k]
                )

    cosserat_internal_stress /= dilatation

    _compute_damping_forces(
        damping_forces,
        velocity_collection,
        dissipation_constant_for_forces,
        lengths,
        ghost_elems_idx,
    )

    internal_forces[:] = (
        difference_kernel_for_block_structure(cosserat_internal_stress, ghost_elems_idx)
        - damping_forces
    )


@numba.njit(cache=True)
def _compute_damping_torques(
    damping_torques, omega_collection, dissipation_constant_for_torques, lengths
):
    blocksize = damping_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            damping_torques[i, k] = (
                dissipation_constant_for_torques[k]
                * omega_collection[i, k]
                * lengths[k]
            )


@numba.njit(cache=True)
def _compute_internal_torques(
    position_collection,
    velocity_collection,
    tangents,
    lengths,
    rest_lengths,
    director_collection,
    rest_voronoi_lengths,
    bend_constants,
    rest_kappa,
    kappa,
    voronoi_dilatation,
    mass_second_moment_of_inertia,
    omega_collection,
    internal_stress,
    internal_couple,
    dilatation,
    dilatation_rate,
    dissipation_constant_for_torques,
    damping_torques,
    internal_torques,
    ghost_voronoi_idx,
    volume,
    sigma,
    rest_sigma,
    phi,
    phi_p,
):
    # Compute \tau_l and cache it using internal_couple
    # Be careful about usage though
    _compute_internal_bending_twist_stresses_from_model(
        position_collection,
        director_collection,
        rest_voronoi_lengths,
        internal_couple,
        bend_constants,
        kappa,
        rest_kappa,
        volume,
        sigma,
        rest_sigma,
        phi,
        phi_p,
    )
    # # Compute dilatation rate when needed, dilatation itself is done before
    # # in internal_stresses
    # _compute_dilatation_rate(
    #     position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
    # )

    # FIXME: change memory overload instead for the below calls!
    voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation**3
    # Delta(\tau_L / \Epsilon^3)
    bend_twist_couple_2D = difference_kernel_for_block_structure(
        internal_couple * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx
    )
    # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
    bend_twist_couple_3D = quadrature_kernel_for_block_structure(
        _batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths
        * voronoi_dilatation_inv_cube_cached,
        ghost_voronoi_idx,
    )
    # (Qt x n_L) * \hat{l}
    shear_stretch_couple = (
        _batch_cross(_batch_matvec(director_collection, tangents), internal_stress)
        * rest_lengths
    )

    # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
    # terms
    # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
    J_omega_upon_e = (
        _batch_matvec(mass_second_moment_of_inertia, omega_collection) / dilatation
    )

    # (J \omega_L / e) x \omega_L
    # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
    # but this causes confusion and violates SRP
    lagrangian_transport = _batch_cross(J_omega_upon_e, omega_collection)

    # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
    # a dilatation-like term in the numerator, which we cancel here
    # (J \omega_L / e^2) . (de/dt)
    unsteady_dilatation = J_omega_upon_e * dilatation_rate / dilatation

    _compute_damping_torques(
        damping_torques, omega_collection, dissipation_constant_for_torques, lengths
    )

    blocksize = internal_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            internal_torques[i, k] = (
                bend_twist_couple_2D[i, k]
                + bend_twist_couple_3D[i, k]
                + shear_stretch_couple[i, k]
                + lagrangian_transport[i, k]
                + unsteady_dilatation[i, k]
                - damping_torques[i, k]
            )


@numba.njit(cache=True)
def _update_accelerations(
    acceleration_collection,
    internal_forces,
    external_forces,
    mass,
    alpha_collection,
    inv_mass_second_moment_of_inertia,
    internal_torques,
    external_torques,
    dilatation,
):
    blocksize_acc = internal_forces.shape[1]
    blocksize_alpha = internal_torques.shape[1]

    for i in range(3):
        for k in range(blocksize_acc):
            acceleration_collection[i, k] = (
                internal_forces[i, k] + external_forces[i, k]
            ) / mass[k]

    alpha_collection *= 0.0
    for i in range(3):
        for j in range(3):
            for k in range(blocksize_alpha):
                alpha_collection[i, k] += (
                    inv_mass_second_moment_of_inertia[i, j, k]
                    * (internal_torques[j, k] + external_torques[j, k])
                ) * dilatation[k]


@numba.njit(cache=True)
def _zeroed_out_external_forces_and_torques(external_forces, external_torques):
    """
    This function is to zeroed out external forces and torques.

    Parameters
    ----------
    external_forces
    external_torques

    Returns
    -------

    Note
    ----
    Microbenchmark results 100 elements
    python version: 3.32 µs ± 44.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    this version: 583 ns ± 1.94 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """
    n_nodes = external_forces.shape[1]
    n_elems = external_torques.shape[1]

    for i in range(3):
        for k in range(n_nodes):
            external_forces[i, k] = 0.0

    for i in range(3):
        for k in range(n_elems):
            external_torques[i, k] = 0.0
