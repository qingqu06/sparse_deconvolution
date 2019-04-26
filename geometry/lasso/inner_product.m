% inner product function that works for both vector and matrix
function z = inner_product(x, y)
    z = sum( sum( x.*y ) );
end