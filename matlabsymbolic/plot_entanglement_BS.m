% plot entanglement 2 mode BS versus nu
nu = linspace(0,100);

E= log(2./(nu+1))-0.5*(nu-1).*log((nu-1)./(nu+1))

plot(nu, E)