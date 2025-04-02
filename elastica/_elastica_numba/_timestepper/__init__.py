__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes of Elastica Numba implementation"""

import numpy as np
import numba
from numba import typeof

# from ._explicit_steppers import ExplicitStepper
# from ._symplectic_steppers import SymplecticStepper
from ._explicit_steppers import ExplicitStepperTag
from ._symplectic_steppers import SymplecticStepperTag

# from elastica.timesteppers.hybrid_rod_steppers import SymplecticCosseratRodStepper
from elastica.timestepper._stepper_interface import _StatefulStepper


def extend_stepper_interface(Stepper, System):
    from elastica.utils import extend_instance
    from elastica._elastica_numba._systems import is_system_a_collection

    # Check if system is a "collection" of smaller systems
    # by checking for the [] method
    is_this_system_a_collection = is_system_a_collection(System)

    ConcreteStepper = (
        Stepper.stepper if _StatefulStepper in Stepper.__class__.mro() else Stepper
    )

    if type(ConcreteStepper.Tag) == SymplecticStepperTag:
        from elastica.timestepper.symplectic_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
            SymplecticStepperMethods as StepperMethodCollector,
        )
    elif type(ConcreteStepper.Tag) == ExplicitStepperTag:
        from elastica.timestepper.explicit_steppers import (
            _SystemInstanceStepper,
            _SystemCollectionStepper,
            ExplicitStepperMethods as StepperMethodCollector,
        )
    # elif SymplecticCosseratRodStepper in ConcreteStepper.__class__.mro():
    #    return  # hacky fix for now. remove HybridSteppers in a future version.
    else:
        raise NotImplementedError(
            "Only explicit and symplectic steppers are supported, given stepper is {}".format(
                ConcreteStepper.__class__.__name__
            )
        )

    # FOR NUMBA JitClass implementation
    # if typeof(ConcreteStepper.Tag) == SymplecticStepperTag.class_type.instance_type:
    #     from ._symplectic_steppers import (
    #         _SystemInstanceStepper,
    #         _SystemCollectionStepper,
    #         SymplecticStepperMethods as StepperMethodCollector,
    #     )
    # elif typeof(ConcreteStepper.Tag) == ExplicitStepperTag.class_type.instance_type:
    #     from ._explicit_steppers import (
    #         _SystemInstanceStepper,
    #         _SystemCollectionStepper,
    #         ExplicitStepperMethods as StepperMethodCollector,
    #     )
    # else:
    #     raise NotImplementedError(
    #         "Only explicit and symplectic steppers are supported, given stepper is {}".format(
    #             ConcreteStepper.__class__.__name__
    #         )
    #     )

    stepper_methods = StepperMethodCollector(ConcreteStepper)
    do_step_method = (
        _SystemCollectionStepper.do_step
        if is_this_system_a_collection
        else _SystemInstanceStepper.do_step
    )
    return do_step_method, stepper_methods.step_methods()


# TODO Improve interface of this function to take args and kwargs for ease of use
def integrate(StatefulStepper, System, final_time: float, n_steps: int = 1000, adaptive_time_step=False, error_tolerance=1e-3, safety_factor=0.22):
    assert final_time > 0.0, "Final time is negative!"
    assert n_steps > 0, "Number of integration steps is negative!"

    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, System)

    dt = np.float64(final_time / n_steps)
    time = np.float64(0.0)
    sf = np.float64(safety_factor)

    from tqdm import tqdm
    import copy

    with tqdm(total=n_steps) as pbar:
        for _ in range(n_steps):
            
            if adaptive_time_step:
                System_copy = copy.deepcopy(System)

            time = do_step(StatefulStepper, stages_and_updates, System, time, dt)

            if adaptive_time_step: 
                half_dt = dt / 2
                do_step(StatefulStepper, stages_and_updates, System_copy, time, half_dt)
                do_step(StatefulStepper, stages_and_updates, System_copy, time, half_dt)

                errors = np.array([
                    np.linalg.norm(copy.kinematic_states.position_collection - system.kinematic_states.position_collection) /
                    np.linalg.norm(system.kinematic_states.position_collection)
                    for system, copy in zip(System._memory_blocks, System_copy._memory_blocks)
                ])
                error = np.max(errors)

                dt *= (error_tolerance / (error + 1e-10)) ** sf

            pbar.update(1)

    print("Final time of simulation is:", time)
