function [ mu ] = MF(x,fuzzyset)%%%%% x is value of any feature and fuzzyset is the number of Fuzzy set

      %%%%max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    if(x<=0)
        mu=1;
    end
    if(x>=1)
        mu=1;
    end
    
    
 %%%%%%%%%%%%%%%%%%% Fuzzy Set1     
    if(fuzzyset==1)
         a=-1;
         b=0;
         c=1;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end    
 %%%%%%%%%%%%%%%%%%% Fuzzy Set2     
    if(fuzzyset==2)
        a=0;
        b=1;
        c=2;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end     
 %%%%%%%%%%%%%%%%%%% Fuzzy Set3     
    if(fuzzyset==3)
        a=-0.5 ;
        b=0;
        c=0.5;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%%%%% Fuzzy Set4     
     if(fuzzyset==4)
         a=0;
         b=0.5;
         c=1;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
     end     
%%%%%%%%%%%%%%%%%%%%%% Fuzzy Set5
    if(fuzzyset==5)
        a=0.5;
        b=1;
        c=1.5;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%%%%%% Fuzzy Set6
    if(fuzzyset==6)
        a=-0.3333333;
        b=0;
        c=0.3333333;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%%%%%% Fuzzy Set7
    if(fuzzyset==7)
        a=0;
        b=0.3333333;
        c=0.6666667;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%%%%%% Fuzzy Set8
     if(fuzzyset==8)
        a=0.3333333;
        b=0.6666667;
        c=1;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%%%%% Fuzzy Set9
    if(fuzzyset==9)
        a=0.6666667;
        b=1;
        c=1.3333333;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%%% Fuzzy Set10
    if(fuzzyset==10)
        a=-0.25;
        b=0;
        c=0.25;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%%% Fuzzy Set11
    if(fuzzyset==11)
        a=0;
        b=0.25;
        c=0.5;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%%% Fuzzy Set12
    if(fuzzyset==12)
        a=0.25;
        b=0.5;
        c=0.75;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%% Fuzzy Set13
    if(fuzzyset==13)
        a=0.5;
        b=0.75;
        c=1;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
%%%%%%%%%%%%%%%%%%% Fuzzy Set14
    if(fuzzyset==14)
        a=0.75;
        b=1;
        c=1.25;
        mu = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
    end 
    
 


end

