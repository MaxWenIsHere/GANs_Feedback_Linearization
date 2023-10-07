function ff=fmdl1(dtheta)
friction_n=-0.18;
friction_p=0.15;
if dtheta<0
   ff=friction_n;%-0.18;
elseif dtheta>0
   ff=friction_p;%0.15;%
else
   ff=0;
end
