function ff=fcfun_stribeck(uu)

theta=uu(2);
dtheta=uu(1);
%u=uu(3);

Mc=0.95;%%库仑摩擦力矩(Nm)
Mv=5;%%粘滞摩擦系数(Nms/rad)
Ms=2;%3;%静摩擦力矩(Nm)
ws=0.1;%Stribeck特征速度(rad/s)

ff=(Ms-Mc)*exp(-(dtheta/ws)^2)*sign(dtheta)+Mc*sign(dtheta);%+Mv*w(i);

