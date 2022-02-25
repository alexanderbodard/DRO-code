module epigraphs


import MathOptInterface as MOI 
import LinearAlgebra as LA 

using MathOptInterface.Utilities

struct ConvexCone
subcones:: Array{MOI.AbstractVectorSet}
end

abstract type AbstractRiskMeasure end 

struct  Riskmeasure <: AbstractRiskMeasure
    A:: Matrix{Float64}
    B:: Matrix{Float64}
    b:: Vector{Float64}
    C:: ConvexCone
    D:: ConvexCone
end

"""
x: primal vector (v, t) with v the argument, t the epigraph variable 
y: dual vector 
""" 
function in_epigraph(r:: Riskmeasure, x::Vector, y::Vector)
    v, t = x[1:end-1], x[end]    

    return in(r.A' * v + r.B' * y , r.C.subcones[1])

    
    
end

function add_risk_epi_constraint(model:: MOI., r:: Riskmeasure)
    y = MOI.@variable(...)
    MOI.add_constraint(model, )  
    


    ... 


    ... 


    ... 


end 
# struct AVAR <: Riskmeasure
# alpha:: Float64
# p_ref:: Vector{Float64}
# end


function main()

    # r = AVAR(0.1, [0.1,0.9])
    r = Riskmeasure(LA.I(3), LA.I(3), [3,2,2], ConvexCone([MOI.Nonnegatives(3)]), ConvexCone([])) 
    x = vcat(ones(3), 3) 
    # display(x) 
    y = ones(3)
    in_epigraph(r, x, y)

    
    # display(r)
    # A = MOI.Nonnegatives(3)
    # display(A)
    # B = MOI.Zeros(2)
    # AB = AVARD{Float64}()
    # MOI.Utilities.add_set(AB, A) 
    # MOI.Utilities.add_set(AB, B) 
    # display( r )
end
main()

end # module
