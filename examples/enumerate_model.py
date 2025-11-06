# examples/enumerate_models.py
from cadical_py import Solver

def block(model: list[int]) -> list[int]:
    return [-lit for lit in model if lit != 0]

if __name__ == "__main__":
    with Solver() as s:
        s.add_clause([+1, +2])   # (x1 ∨ x2)
        s.add_clause([-1, -2])   # (¬x1 ∨ ¬x2)  -> exactly one true

        n = 0
        while True:
            r = s.solve()              # 10 = SAT, 20 = UNSAT
            if r == 20: break
            if r != 10:  print("UNKNOWN"); break
            m = s.model()
            print("model:", m)
            s.add_clause(block(m))     # block current model
            n += 1
        print("total:", n)
