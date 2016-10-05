nlp = ADNLPModel(x->dot(x,x), zeros(2))
@test !has_bounds(nlp)
@test !bound_constrained(nlp)
@test unconstrained(nlp)
@test !linearly_constrained(nlp)
@test !equality_constrained(nlp)
@test !inequality_constrained(nlp)

nlp = ADNLPModel(x->dot(x,x), zeros(2),
                 lvar=zeros(2))
@test has_bounds(nlp)
@test bound_constrained(nlp)
@test !unconstrained(nlp)
@test !linearly_constrained(nlp)
@test !equality_constrained(nlp)
@test !inequality_constrained(nlp)

nlp = ADNLPModel(x->dot(x,x), zeros(2),
                 c=x->[prod(x)-1], lcon=[0.0], ucon=[0.0])
@test !has_bounds(nlp)
@test !bound_constrained(nlp)
@test !unconstrained(nlp)
@test !linearly_constrained(nlp)
@test equality_constrained(nlp)
@test !inequality_constrained(nlp)

nlp = ADNLPModel(x->dot(x,x), zeros(2),
                 c=x->[prod(x)-1], lcon=[0.0], ucon=[1.0])
@test !has_bounds(nlp)
@test !bound_constrained(nlp)
@test !unconstrained(nlp)
@test !linearly_constrained(nlp)
@test !equality_constrained(nlp)
@test inequality_constrained(nlp)

nlp = ADNLPModel(x->dot(x,x), zeros(2),
                 c=x->[prod(x)-1; sum(x)-1],
                 lcon=zeros(2), ucon=[0.0; Inf])
@test !has_bounds(nlp)
@test !bound_constrained(nlp)
@test !unconstrained(nlp)
@test !linearly_constrained(nlp)
@test !equality_constrained(nlp)
@test !inequality_constrained(nlp)

nlp = ADNLPModel(x->dot(x,x), zeros(2),
                 c=x->[sum(x)-1], lcon=zeros(1), ucon=zeros(1),
                 lin=[1])
@test !has_bounds(nlp)
@test !bound_constrained(nlp)
@test !unconstrained(nlp)
@test linearly_constrained(nlp)
@test equality_constrained(nlp)
@test !inequality_constrained(nlp)

nlp = ADNLPModel(x->dot(x,x), zeros(2), lvar=zeros(2), uvar=ones(2),
                 c=x->[sum(x)-1], lcon=zeros(1), ucon=zeros(1),
                 lin=[1])
@test has_bounds(nlp)
@test !bound_constrained(nlp)
@test !unconstrained(nlp)
@test linearly_constrained(nlp)
@test equality_constrained(nlp)
@test !inequality_constrained(nlp)
