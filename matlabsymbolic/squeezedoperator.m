clearvars
syms r theta UR UI WR WI a real

N=2
M=sym('M', N)

[Rq,Rp,J]=RqRpJ(N)

UR=cosh(r)
%syms UI
UI=0
WR=-cos(theta)*sinh(r)
WI=-sin(theta)*sinh(r)

M1= Rq*(UR+WR)*Rq'+Rp*(UR-WR)*Rp';
M2= Rp*(UI+WI)*Rq'-Rq*(UI-WI)*Rp';

M=M1+M2

%% verify that M is symplectic
simplify(M'*J*M)
