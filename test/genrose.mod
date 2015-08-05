# Generalized Rosenbrock function.
#
#   Source: 
#   Y.-W. Shang and Y.-H. Qiu,
#   A note on the extended Rosenbrock function,
#   Evolutionary Computation, 14(1):119–126, 2006.
#
# Shang and Qiu claim the "extended" Rosenbrock function
# previously appeared in
#
#   K. A. de Jong,
#   An analysis of the behavior of a class of genetic
#   adaptive systems,
#   PhD Thesis, University of Michigan, Ann Arbor,
#   Michigan, 1975,
#   (http://hdl.handle.net/2027.42/4507)
#
# but I could not find it there, and in
#
#   D. E. Goldberg,
#   Genetic algorithms in search, optimization and
#   machine learning,
#   Reading, Massachusetts: Addison-Wesley, 1989,
#
# but I don't have access to that book.
#
# This unconstrained problem is analyzed in
#
#   S. Kok and C. Sandrock,
#   Locating and Characterizing the Stationary Points of
#   the Extended Rosenbrock Function,
#   Evolutionary Computation 17, 2009.
#   https://dx.doi.org/10.1162%2Fevco.2009.17.3.437
# 
#   classification SUR2-AN-V-0
#
# D. Orban, Montreal, 08/2015.

param n := 100;

var x{i in 1..n} := i/(n+1);

minimize f:
	1.0 +
	sum {i in 2..n} 100*(x[i]-x[i-1]^2)^2 +
	sum {i in 1..n-1} (x[i]-1.0)^2;
