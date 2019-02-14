# Brown and Dennis functions
#
#   Source: Problem 16 in
#   J.J. Moré, B.S. Garbow and K.E. Hillstrom,
#   "Testing Unconstrained Optimization Software",
#   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981
#
#   classification SUR2-AN-4-0
using ForwardDiff

function brownden_autodiff()

  x0 = [25.0; 5.0; -5.0; -1.0]
  f(x) = begin
    s = 0.0
    for i = 1:20
      s += ((x[1] + x[2] * i/5 - exp(i/5))^2 + (x[3] + x[4]*sin(i/5) -
          cos(i/5))^2)^2
    end
    return s
  end

  return ADNLPModel(f, x0)
end
