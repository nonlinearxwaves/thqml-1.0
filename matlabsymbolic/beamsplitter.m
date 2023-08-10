clearvars
syms r theta UR UI WR WI a phi0 phi1 real

% theta=pi/4
%theta=0
% phi0=0
% phi1=0
N=4
M=sym('M', N)

[Rq,Rp,J]=RqRpJ(N)

UR=[cos(theta)*cos(phi0), -sin(theta)*cos(phi1);
    sin(theta)*cos(phi1), cos(theta)*cos(phi0)]
UI=[-cos(theta)*sin(phi0), sin(theta)*sin(phi1);
    sin(theta)*sin(phi1), cos(theta)*sin(phi0)]
WR=[0, 0;0, 0]
WI=[0, 0;0, 0]

%%
MR= Rq*(UR+WR)*Rq'+Rp*(UR-WR)*Rp';
MI= Rp*(UI+WI)*Rq'-Rq*(UI-WI)*Rp';

M=MR+MI

%% verify that M is symplectic
simplify(M'*J*M)

simplify(M*J'*M'*J)

