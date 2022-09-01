function Yc = codifica(Y,code)

u = unique(Y)';

c = length(u);

Yc = zeros(size(Y,1),length(u));

for i=1:c
    idx = find(Y==u(i));
    Yc(idx,:)=repmat(code(i,:),size(idx,1),1);
end    
