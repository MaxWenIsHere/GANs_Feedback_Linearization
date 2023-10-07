%clear all;
%close all;
load f.dat;
load time.dat;
load input.dat;
load output.dat;

tSamp=0.001;
%persistent k;
%global T;
l=1;
m=size(f);
for k=1:1:m-1
    f1=f(k);
    f2=f(k+1);
   if f2==f1
      fout(l)=f1;
  else l=l+1;
  end
end
% result[2]=f;
ai11=0; ai12=0; ai21=0; ai22=0; bi1=0; bi2=0; ci1=0; ci2=0;
ai11_1=0; ai12_1=0; ai21_1=0; ai22_1=0; bi1_1=0; bi2_1=0;
for m=1:l
    w(m)=2.0*pi*fout(m);
    point=20000;
    for i=10:(point-10)
        
			     ai11=ai11_1+cos(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
				 ai12=ai12_1+cos(w(m)*i*tSamp)*sin(w(m)*i*tSamp);
			     ai21=ai21_1+sin(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
			     ai22=ai22_1+sin(w(m)*i*tSamp)*sin(w(m)*i*tSamp);
			     bi1=bi1_1+cos(w(m)*i*tSamp)*input(20000*(l-1)+i);
			     bi2=bi2_1+sin(w(m)*i*tSamp)*input(20000*(l-1)+i);
                 ai11_1=ai11;
                 ai12_1=ai12;
                 ai21_1=ai21;
                 ai22_1=ai22;
                 bi1_1=bi1;
                 bi2_1=bi2;
                 
     end
        if(abs(ai11*ai22-ai12*ai21)>1e-6)
	     
	      ci1=(ai22*bi1-ai12*bi2)/(ai11*ai22-ai12*ai21);
	      ci2=(ai11*bi2-ai21*bi1)/(ai11*ai22-ai12*ai21);
	      Pin=atan2(ci2,ci1);
	      Ain=sqrt(ci1*ci1+ci2*ci2);
          Angle_in(m)=Pin;
          Amp_in(m)=Ain;
          
        else
                Angle_in(m)=0;
                Amp_in(m)=0;
                
       end
end
  
ao11=0; ao12=0; ao21=0; ao22=0; bo1=0; bo2=0; co1=0; co2=0;
ao11_1=0; ao12_1=0; ao21_1=0; ao22_1=0; bo1_1=0; bo2_1=0;

for m=1:l
    w(m)=2.0*pi*fout(m);
    point=20000;
    for i=10:(point-10)
 	   ao11=ao11_1+cos(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
 	   ao12=ao12_1+cos(w(m)*i*tSamp)*sin(w(m)*i*tSamp);
 	   ao21=ao21_1+sin(w(m)*i*tSamp)*cos(w(m)*i*tSamp);
 	   ao22=ao22_1+sin(w(m)*i*tSamp)*sin(w(m)*i*tSamp);
 	   bo1=bo1_1+cos(w(m)*i*tSamp)*output(20000*(l-1)+i);
 	   bo2=bo2_1+sin(w(m)*i*tSamp)*output(20000*(l-1)+i);
       ao11_1=ao11;
       ao12_1=ao12;
       ao21_1=ao21;
       ao22_1=ao22;
       bo1_1=bo1;
       bo2_1=bo2;
     end
     if (abs(ao11*ao22-ao12*ao21)>1e-6)
  	      co1=(ao22*bo1-ao12*bo2)/(ao11*ao22-ao12*ao21);
   	      co2=(ao11*bo2-ao21*bo1)/(ao11*ao22-ao12*ao21);
   	      Angle_out(m)=atan2(co2,co1);
   	      Amp_out(m)=sqrt(co1*co1+co2*co2);
     else
 	           Angle_out(m)=0.0;
 	           Amp_out(m)=0.0;
     end
 end
for n=1:l
 if (abs(Amp_in)>1e-6)
	    Amp(n)=20*log10(abs(Amp_out(n)/Amp_in(n)));
 else
	       Freq(n)=0.0;
	       Angle(n)=0.0;
	       Amp(n)=0.0;
 end 
Freq(n)=fout(n);
Angle(n)=(Angle_out(n)-Angle_in(n))*180/pi;
end

fidf=fopen('freq.dat','w');
for j=1:l
fprintf(fidf,'%f\t%f\t%f\n',Freq(j),Amp(j),Angle(j));
end
fclose(fid);

