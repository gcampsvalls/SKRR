function X2 = scale(X)

[filas columnas] = size(X);

for c=1:columnas
    X2(:,c) = (X(:,c) - min(X(:,c))) ./ (max(X(:,c)) - min(X(:,c)));
end

