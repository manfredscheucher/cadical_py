# src/cadical_py/propagator.py
class ExternalPropagator:
    """Python mirror of CaDiCaL::ExternalPropagator. Override as needed."""
    def init(self, solver): pass
    def notify_new_decision_level(self): pass
    def notify_backtrack(self, new_level: int): pass
    def notify_assignment(self, lits: list[int]): pass
    def cb_propagate(self) -> int: return 0
    def cb_add_reason_clause_lit(self, propagated_lit: int) -> int: return 0
    def cb_decide(self) -> int: return 0
    def cb_check_found_model(self, model: list[int]) -> bool: return True
    def cb_has_external_clause(self) -> tuple[bool, bool]: return (False, True)
    def cb_add_external_clause_lit(self) -> int: return 0
