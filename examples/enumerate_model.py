from cadical_py import Solver

s = Solver()
s.add_clause([1,2])
s.add_clause([-1,-2])

while s.solve() != 20:
	m = s.model()
	print("found model:",m)
	s.add_clause([-v for v in m])

print("stop.")