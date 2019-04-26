function [x1, Gx, Grad] = prox_grad(f, Psi, x, gx, t)

x1 = Psi.prox_mapping(x - t*gx, t);
Gx = (x - x1)/t;

if nargout <= 2; return; end
Grad = Gx - t * f.Hess(Gx) ;

end