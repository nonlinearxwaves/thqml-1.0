%% test identities

N=4

[Rq,Rp,J]=RqRpJ(N)

  
Rq'*Rq 
Rq'*Rp

Rp'*Rp 
Rp'*Rq

Rq*Rq'+Rp*Rp'


Rq*Rp'-Rp*Rq'

J*Rq
J*Rp



J
inv(J)
J'
J*J
J'*J
%%
A=sym('A',[N/2 N/2],'real')
%%
A1= A-A';
Rq*A1*Rp'-Rp*A1*Rq'


