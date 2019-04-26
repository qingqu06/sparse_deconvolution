classdef func_simple < handle
% Define interface of simple closed convex function Psi(x)
    methods (Abstract)
        [fval, subgrad] = oracle(Psi, x)
        % Return function value Psi(x), and a subgradient

        u = prox_mapping(Psi, z, t)
        % Return: argmin_u { (1/2)||u-z||_2^2 + t*Psi(u) }
        % same as argmin_u { (1/2*t)||u-z||_2^2 + Psi(u) }

        mu = strong_convex_parameter(Psi)
        % Return (strong) convexity parameter
    end
end