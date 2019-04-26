function [a_shift, x_shift] = shift_correction( a, x, opts)


a_0 = shift_opts.A_0;
x_0 = shift_opts.x_0;

n_0 = length(a_0);
n = length(a);
m = length(x);

if(opts.grouth_truth)
    Corr = cconv(reversal(a_0),a,m);
    [~,ind] = max(abs(Corr));
    Corr_max = Corr(ind);
    
    if(Corr_max>0)
        a_shift = circshift(a, ind-1);
        x_shift = circshift(x,  -(ind-1));
    else
        a_shift = - circshift(a, ind-1);
        x_shift = - circshift(x, -(ind-1));
    end
           
end


end