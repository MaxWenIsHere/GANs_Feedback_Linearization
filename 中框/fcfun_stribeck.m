function ff=fcfun_stribeck(uu)

theta=uu(2);
dtheta=uu(1);
%u=uu(3);

Mc=0.95;%%����Ħ������(Nm)
Mv=5;%%ճ��Ħ��ϵ��(Nms/rad)
Ms=2;%3;%��Ħ������(Nm)
ws=0.1;%Stribeck�����ٶ�(rad/s)

ff=(Ms-Mc)*exp(-(dtheta/ws)^2)*sign(dtheta)+Mc*sign(dtheta);%+Mv*w(i);

