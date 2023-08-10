function [Rq,Rp,J]=RqRpJ(N)

Rq=sym('Rq',[N N/2])
Rp=sym('Rq',[N N/2])

for i=1:N
    for j=1:(N/2)
        Rq(i,j)=0;
        Rp(i,j)=0;
    end
end

c=0;
d=0;
for i=1:N
    if rem(i-1,2) ==0
        c=c+1;
        Rq(i,c)=1;
    else
        d=d+1;
        Rp(i,d)=1;
    end
end


J=Rq*Rp'-Rp*Rq';
  