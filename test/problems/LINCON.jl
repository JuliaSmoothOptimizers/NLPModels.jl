function LINCON_autodiff()

  A = [1 2; 3 4]
  b = [5; 6]
  B = diagm([3 * i for i = 3:5])
  c = [1; 2; 3]
  C = [0 -2; 4 0]
  d = [1; -1]

  x0 = zeros(15)
  f(x) = sum(i + x[i]^4 for i = 1:15)
  con(x) = [15 * x[15];
            c' * x[10:12];
            d' * x[13:14];
            b' * x[8:9];
            C * x[6:7];
            A * x[1:2];
            B * x[3:5]]

  lcon = [22.0; 1.0; -Inf; -11.0; -d;            -b; -Inf * ones(3)]
  ucon = [22.0; Inf; 16.0;   9.0; -d; Inf * ones(2);              c]

  return ADNLPModel(f, x0, con, lcon, ucon, name="LINCON_autodiff")
end
