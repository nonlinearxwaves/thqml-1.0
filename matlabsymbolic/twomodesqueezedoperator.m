clearvars
syms r theta UR UI WR WI a real


N=4
M=sym('M', N)

[Rq,Rp,J]=RqRpJ(N)

UR=[cosh(r), 0;0, cosh(r)]
UI=[0, 0;0, 0]
WR=[0, -cos(theta)*sinh(r);-cos(theta)*sinh(r), 0]
WI=[0, -sin(theta)*sinh(r);-sin(theta)*sinh(r), 0]

%%
M1= Rq*(UR+WR)*Rq'+Rp*(UR-WR)*Rp';
M2= Rp*(UI+WI)*Rq'-Rq*(UI-WI)*Rp';

M=M1+M2

%% verify that M is symplectic
simplify(M'*J*M)

simplify(M*J'*M'*J)

