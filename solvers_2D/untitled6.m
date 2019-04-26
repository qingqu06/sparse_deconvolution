%% plot the results
% shift correction
[A_shift, X_shift] = shift_correction_2D(A, X);

figure(1);
imagesc(Y);

figure(2);
for k = 1:K
    subplot(1,K,k); imagesc(abs(A_shift(:,:,k)));
    colormap('jet');
    axis off;
end

for t = 1:T
    figure(2+t);
    for k = 1:K
        subplot(1,K,k); imagesc(abs(X_shift(:,:,k,t)));
        colormap('jet');
        axis off;
    end
end