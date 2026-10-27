"""Exact fixed Holosoma function body; minimal IsaacSim-only global bindings.

Full module is deliberately not imported: its setup imports would add unrelated
dependencies to the empty-app comparison. Standard-library logging replaces
the loguru binding only; SimulatorType/get_simulator_type bind ISAACSIM only.
Original function text SHA256: 16c60deb4d5fb77a7f524e02085039e15b95209c56dbcfd1a722f6ada9eb9801.
Full source sim_utils.py SHA256: 6a9373d35f1560125af521a35fae3208457acd1639d19bb2555ed12cf7b0f3a2.
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SimulatorType(Enum):
    ISAACSIM = "isaacsim"

def get_simulator_type():
    return SimulatorType.ISAACSIM

def close_simulation_app(simulation_app):
    """Close simulation app with workarounds for known issues.

    Parameters
    ----------
    simulation_app : Any
        The simulation app instance returned by init_sim_imports().
        Can be None for simulators that don't have an app (e.g., IsaacGym).
    """
    if simulation_app is not None and get_simulator_type() == SimulatorType.ISAACSIM:
        logger.info("Shutting down simulation app...")
        try:
            # Work-around for IsaacLab hanging headless.
            # Patch the close_stage method to avoid hanging
            import omni.usd

            context = omni.usd.get_context()
            context_class = context.__class__

            # Replace with a no-op version
            def noop_close_stage(self, *args, **kwargs):
                logger.info("Skipping close_stage() to avoid hanging")
                return True

            # Apply the patch
            context_class.close_stage = noop_close_stage
            logger.info("Successfully patched close_stage method")
        except Exception as e:
            logger.warning(f"Could not patch close_stage method: {e}")

        try:
            # Work-around for IsaacLab SimulationContext._app_control_on_stop_handle_fn
            # hanging in an infinite render() loop on shutdown. When simulation_app.close()
            # triggers a timeline STOP event, the callback spins waiting for the timeline to
            # start playing again — which never happens. Disabling the callback prevents this.
            from isaaclab.sim import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context._disable_app_control_on_stop_handle = True
                logger.info("Disabled SimulationContext app_control_on_stop_handle to prevent shutdown hang")
        except Exception as e:
            logger.warning(f"Could not disable app_control_on_stop_handle: {e}")

        # Now close the app
        simulation_app.close(wait_for_replicator=False)
        logger.info("Simulation app closed.")
    else:
        logger.info("Simulation app closed.")