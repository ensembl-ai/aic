"""Full insertion: approach, guarded descent, spiral search, and push-in.

See ``aic_perception.policies.insertion`` for the phases and the information
the policy is allowed to use.
"""

from __future__ import annotations

from aic_perception.policies.insertion import PortInsertion


class PerceptionInsert(PortInsertion):
    """Insert the plug into the perceived port with force-guided search."""
