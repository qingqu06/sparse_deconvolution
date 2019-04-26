function [ revX ] = reversal( X, m )
if nargin > 1
    X = [X(1:min(size(X,1), m), :) ; zeros(max(m - size(X,1), 0), size(X,2))];
end

revX = [X(1,:) ; flipud(X(2:end,:))];
end