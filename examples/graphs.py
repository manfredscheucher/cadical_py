from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
import sys

import cadical_py


# ---------- Combinatorics / Helpers ----------

@dataclass
class EdgeMap:
    """Bidirectional mapping between undirected edges (i<j) and var IDs 1..m for a graph with n vertices."""
    n: int

    def __post_init__(self):
        self.id2edge: List[Tuple[int, int]] = [(0, 0)] * (self.n * (self.n - 1) // 2 + 1)  # 1-based
        self.edge2id: dict[Tuple[int, int], int] = {}
        id_ = 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.id2edge[id_] = (i, j)
                # Undirected: store both orientations
                self.edge2id[(i, j)] = id_
                self.edge2id[(j, i)] = id_
                id_ += 1

    def id_of(self, i: int, j: int) -> int:
        return self.edge2id.get((i, j), 0)


def colex_pairs(k: int) -> List[Tuple[int, int]]:
    """Colex order of 2-combinations over {0..k-1}: (0,1),(0,2),(1,2),(0,3),(1,3),(2,3),…"""
    out = []
    for j in range(1, k):
        for i in range(0, j):
            out.append((i, j))
    return out


def permute_sub(A: List[List[int]], perm: List[int]) -> List[List[int]]:
    """Return the perm×perm submatrix (rows/cols permuted by 'perm'). Values: -1 unset, 0 false, 1 true."""
    k = len(perm)
    B = [[0] * k for _ in range(k)]
    for rr in range(k):
        for cc in range(k):
            B[rr][cc] = A[perm[rr]][perm[cc]]
    return B


def fingerprint_colex(M: List[List[int]]) -> List[int]:
    """Fingerprint of the upper triangle in colex order."""
    k = len(M)
    fp = []
    for (i, j) in colex_pairs(k):
        fp.append(M[i][j])
    return fp


def replace_unset(L: List[int], newvalue: int) -> List[int]:
    """Replace -1 (unset) by newvalue (0/1)."""
    return [newvalue if x < 0 else x for x in L]


# ---------- Propagator ----------

class GraphProp:
    """
    Enforces colex-minimal adjacency under permutations.
    On violation, produces a unit implication with a reason clause.
    Blocks each found model via external clause streaming (like your ChooseK example).
    """

    def __init__(self, n: int):
        self.S = None
        self.n = int(n)
        self.edges = EdgeMap(self.n)

        # Assignment tracking
        self.level = 0
        self.assigned_lits_at_level = [set()]  # signed literals per level
        self.reason_lits_at_level = [set()]
        self.pos_ids: set[int] = set()         # edge IDs currently TRUE
        self.neg_ids: set[int] = set()         # edge IDs currently FALSE

        # Propagation
        self.prop_queue: deque[int] = deque()
        self.reason_of: dict[int, list[int]] = {}  # implied lit -> reason clause (as stack)

        # External clause queue (blocking, etc.)
        self._ext_clauses: deque[list[int]] = deque()
        self._cur_ext_clause: list[int] | None = None

        # Stats
        self.propagate_calls = 0
        self.solutions = 0

    # Build current n×n partial adjacency matrix
    # Values: -1 = unset, 0 = false, 1 = true
    def adj_matrix(self) -> List[List[int]]:
        A = [[0 if i == j else -1 for j in range(self.n)] for i in range(self.n)]
        m = self.n * (self.n - 1) // 2
        for id_ in range(1, m + 1):
            r, c = self.edges.id2edge[id_]
            if id_ in self.pos_ids:
                A[r][c] = A[c][r] = 1
            elif id_ in self.neg_ids:
                A[r][c] = A[c][r] = 0
            else:
                A[r][c] = A[c][r] = -1
        return A

    # Try to derive a conflict/implication:
    # Search for a permutation producing B_fp < A_fp (colex).
    # The clause is: [+x for edges that are 0 in B] ∪ [-x for edges that are 1 in A],
    # taken up to the first difference. Empty => undecidable yet.
    def first_conflict_clause(self) -> list[int]:
        A = self.adj_matrix()

        perm: list[int] = []
        used = [False] * self.n
        clause: list[int] = []

        def dfs() -> bool:
            k = len(perm)
            base_idx = list(range(k))

            A_ = permute_sub(A, base_idx)
            B_ = permute_sub(A, perm)

            # Unset handling: A optimistic (0), B pessimistic (1)
            A_fp = replace_unset(fingerprint_colex(A_), 0)
            B_fp = replace_unset(fingerprint_colex(B_), 1)

            # Prune if B > A (no better permutation ahead)
            if B_fp > A_fp:
                return False

            if k == self.n:
                if B_fp < A_fp:
                    ordA = colex_pairs(k)  # pairs over 0..k-1
                    ordB = [(perm[i], perm[j]) for (i, j) in ordA]

                    must_pos: set[int] = set()
                    must_neg: set[int] = set()
                    for l in range(len(B_fp)):
                        if B_fp[l] == 0:
                            idb = self.edges.id_of(ordB[l][0], ordB[l][1])
                            must_pos.add(idb)
                        if A_fp[l] == 1:
                            ida = self.edges.id_of(ordA[l][0], ordA[l][1])
                            must_neg.add(ida)
                        if B_fp[l] != A_fp[l]:
                            clause.clear()
                            clause.extend(+x for x in must_pos)
                            clause.extend(-x for x in must_neg)
                            return True
            else:
                # Extend permutation
                for v in range(self.n):
                    if not used[v]:
                        used[v] = True
                        perm.append(v)
                        if dfs():
                            return True
                        perm.pop()
                        used[v] = False
            return False

        if dfs():
            return clause
        return []

    # ---------- ExternalPropagator interface (duck-typed) ----------

    def init(self, solver):
        self.S = solver

    def notify_new_decision_level(self):
        self.level += 1
        self.assigned_lits_at_level.append(set())
        self.reason_lits_at_level.append(set())

    def notify_backtrack(self, new_level: int):
        while self.level > new_level:
            for lit in self.assigned_lits_at_level[self.level]:
                if lit > 0:
                    self.pos_ids.discard(lit)
                else:
                    self.neg_ids.discard(-lit)
            self.assigned_lits_at_level.pop()

            for lit in self.reason_lits_at_level[self.level]:
                self.reason_of.pop(lit, None)
            self.reason_lits_at_level.pop()

            self.level -= 1

        # Drop scheduled propagations (solver will re-query)
        self.prop_queue.clear()
        self._cur_ext_clause = None

    def notify_assignment(self, lits: List[int]):
        for lit in lits:
            self.assigned_lits_at_level[self.level].add(lit)
            if lit > 0:
                self.pos_ids.add(lit)
            else:
                self.neg_ids.add(-lit)

        # Try to derive a symmetry cut as a unit implication
        cls = self.first_conflict_clause()
        if cls:
            impl = cls[0]  # take first literal as unit implication (mirrors the C++ sample)
            if impl not in self.reason_of:
                self.prop_queue.append(impl)
                # Store full reason clause as a stack (pop from the end)
                self.reason_of[impl] = list(cls)[::-1]
                self.reason_lits_at_level[self.level].add(impl)

    def cb_propagate(self) -> int:
        if not self.prop_queue:
            return 0
        self.propagate_calls += 1
        return self.prop_queue.popleft()

    def cb_add_reason_clause_lit(self, propagated_lit: int) -> int:
        cls = self.reason_of.get(propagated_lit)
        if not cls:
            self.reason_of.pop(propagated_lit, None)
            return 0
        lit = cls.pop()
        if not cls:
            self.reason_of.pop(propagated_lit, None)
        return lit

    def cb_decide(self) -> int:
        return 0

    def cb_check_found_model(self, model: List[int]) -> bool:
        """
        Called on SAT. Count solution and push a blocking clause into the external queue.
        The model vector contains ±v at index v-1 for the observed main variables 1..m.
        """
        # Count solution
        self.solutions += 1

        # Build blocking clause (negate each assignment for the m main vars)
        m = len(model)
        block = [(- (i + 1)) if model[i] > 0 else (i + 1) for i in range(m)]
        self._ext_clauses.append(block)  # streamed out via cb_has_external_clause / cb_add_external_clause_lit
        return True

    # External clauses (blocking)
    def cb_has_external_clause(self) -> tuple[bool, bool]:
        # (has_clause, is_forgettable)
        return (len(self._ext_clauses) > 0, True)

    def cb_add_external_clause_lit(self) -> int:
        """
        Stream the current external clause literal-by-literal; return 0 to terminate a clause.
        """
        if self._cur_ext_clause is None:
            if not self._ext_clauses:
                return 0
            # Reverse so we can pop() from the end
            self._cur_ext_clause = list(self._ext_clauses.popleft())[::-1]

        if not self._cur_ext_clause:
            self._cur_ext_clause = None
            return 0  # clause end
        return self._cur_ext_clause.pop()


# OEIS A000088 (number of unlabeled graphs), small values as a sanity check.
EXPECTED_UNLABELED = [
    1, 1, 2, 4, 11, 34, 156, 1044, 12346,
    274668, 12005168, 1018997864, 165091172592,
]


def run_graph_enum(n: int) -> tuple[int, int]:
    """
    Build m = nC2 variables (edges), enumerate models and block each
    via the propagator's external clause callbacks.
    Returns: (solutions, propagate_calls).
    """
    if n < 0:
        raise ValueError("n >= 0")

    m = n * (n - 1) // 2

    S = cadical_py.Solver()
    P = GraphProp(n)
    S.connect_external_propagator(P)
    P.init(S)

    # Register observed variables; otherwise callbacks won't fire.
    for v in range(1, m + 1):
        S.add_observed_var(v)

    while True:
        code = S.solve()  # 10 = SAT, 20 = UNSAT
        if code == 20:
            break
        if code != 10:
            raise RuntimeError("UNKNOWN solver status")
        # On SAT, cb_check_found_model() counted + queued the blocking clause.

    # Optional sanity check
    if 0 <= n < len(EXPECTED_UNLABELED):
        assert P.solutions == EXPECTED_UNLABELED[n], (
            f"unexpected count: {P.solutions} vs {EXPECTED_UNLABELED[n]} (OEIS A000088)"
        )

    return P.solutions, P.propagate_calls


if __name__ == "__main__":
    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    else:
        print(f"Usage: {sys.argv[0]} n [k]")
        sys.exit(1)
        
    sols, calls = run_graph_enum(n)
    print(f"{sols} solutions")
    print(f"propagate_calls: {calls}")
