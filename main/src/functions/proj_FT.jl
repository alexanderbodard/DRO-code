function psi(g, func)
    p, l_value = ProximalOperators.prox(f, x, g)
    return func(p) - g - s    
end

function bisection_method!(g_lb, g_ub, tol, func)
    if psi(g_lb, func)*psi(g_ub, func) > 0
        error("Incorrect initial interval. Found $(psi(g_lb, func)) and $(psi(g_ub, func))")
    end

    while abs(g_ub-g_lb) > tol
        g_new = (g_lb + g_ub) / 2.
        if psi(g_lb, func) * psi(g_new, func) < 0
            g_ub = g_new
        else
            g_lb = g_new
        end
    end
    return (g_lb + g_ub) / 2.
end

function proj_FT!(x, s, g_lb, g_ub, tol, func)
    # TODO: Find out whether to use pass by value or pass by reference in Julia

    if func(x) <= s
        return x, s # Return copy or same object?
    else
        g = bisection_method!(g_lb, g_ub, tol, func)
        p, l_value = ProximalOperators.prox(func, x, g)
        return p, s + g
    end
end