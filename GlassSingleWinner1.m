   


   %%%%%%%%%%%%%%  @copy by sobhan siamak %%%%%%%%%%

%% 
% In this project we implement following paper
% paper with title of "A proposed method for learning rule weights in fuzzy rule based classification system"
%  Tnorm = product
% Datasets are Wine, Glass and Sonar from UCI repository
%% 
clc;
clear;
close all;

%% First Step is Reading Datasets and Normalized them


glass=load('glass.data');%%%%% Glass have 9 features, 214 pattern and 6 classes but there is not class4

%%%%%%%%%%   Normalize glass
normalglass=glass;
normalglass=normalglass(:,2:end);
[m2,n2]=size(normalglass);
minmax2=zeros(1,n2-1);%%%%%first row is min and second row is max
for i=1:n2-1
    minmax2(1,i)=min(normalglass(:,i));%%%%% min row
    minmax2(2,i)=max(normalglass(:,i));%%%%% max row
end
for i=1:n2-1
    for j=1:m2
      normalglass(j,i)=(normalglass(j,i)-minmax2(1,i))/(minmax2(2,i)-minmax2(1,i));          
    end
end
glassDs=normalglass(:,1:end-1);
glassLabel=normalglass(:,end);
gcn=53;

%% Step2 ============= Rule Generation

%% #1:Rule Generation for Glass Dataset 

 %% %%%%%%% Rule Generator with one antecedente or length=1
     %%%% the number of these rules is 9*14=126
     glassRB1=zeros(9*14,3);
     
     %# Feature Fuzzyset Class#
     ind=1;
     for i=1:9
         for j=1:14
             glassRB1(ind,1)=i;
             glassRB1(ind,2)=j;
             ind=ind+1;
         end
     end
     
     
      %%%%%%% Define Class Label for every Rule
%      confidence1=zeros(9,14);
     mufriend=zeros(126,1);%%%%%  for first term of 18 formula in paper
     muenemy=zeros(126,1);%%%%%   for second term of 18 formula in paper
     index=1;
     
    for i=1:9   %%%%% i is the number of features
        for j=1:14 %%%%%% j is the number of fuzzysets
             conf=zeros(214,1);
             for k=1:size(glassDs,1)
               conf(k)=MF(glassDs(k,i),j);
             end
             classtotal=sum(conf);
             class1=sum(conf(1:70));
             class2=sum(conf(71:146));
             class3=sum(conf(147:163));
             class5=sum(conf(164:176));
             class6=sum(conf(177:185));
             class7=sum(conf(186:214));
             
             max1=class1/classtotal;
             max2=class2/classtotal;
             max3=class3/classtotal;
             max5=class5/classtotal;
             max6=class6/classtotal;
             max7=class7/classtotal;
             
             maxarry=[max1 max2 max3 max5 max6 max7];
             [maxm,indmx]=max(maxarry);
             if(indmx==1)
                 glassRB1(index,3)=1;                 
                 mufriend(index)=class1;
                 muenemy(index)=class2+class3+class5+class6+class7;
             end
             if(indmx==2)
                 glassRB1(index,3)=2;
                 mufriend(index)=class2;
                 muenemy(index)=class1+class3+class5+class6+class7;
             end
             if(indmx==3)
                 glassRB1(index,3)=3;
                 mufriend(index)=class3;
                 muenemy(index)=class1+class2+class5+class6+class7;
            end
             if(indmx==4)%%%%%% there is not class 4 in glass dataset
                 glassRB1(index,3)=5;
                 mufriend(index)=class5;
                 muenemy(index)=class1+class2+class3+class6+class7;
             end
             if(indmx==5)%%%%%% there is not class 4 in glass dataset
                 glassRB1(index,3)=6;
                 mufriend(index)=class6;
                 muenemy(index)=class1+class2+class3+class5+class7;
             end
             if(indmx==6)%%%%%% there is not class 4 in glass dataset
                 glassRB1(index,3)=7;
                 mufriend(index)=class7;
                 muenemy(index)=class1+class2+class3+class5+class6;
             end
             
           index=index+1;  
       end %%%% end of for 9
   end%%%%%% end of for 14
     
     
     
 %% %%%%%%% Rule Generator with two antecedente or length=2
     %%%% the number of these rules is comb(9,2)*14*14=7056

 glassRB2=zeros(7056,5);%%%%first and second ellements are features third and fourth ellements are fuzzysets and last is class
 index2=1;
 for i=1:9
         for j=i+1:9
             for k=1:14
                 for m=1:14
                     glassRB2(index2,1)=i;
                     glassRB2(index2,2)=j;
                     glassRB2(index2,3)=k;
                     glassRB2(index2,4)=m; 
                     index2=index2+1;
                 end
             end
         end
 end
 
 
   %%%%%%% Define Class Label for every Rule
   
   mu2friend=zeros(7056,1);%%%%%  for first term of 18 formula in paper
   mu2enemy=zeros(7056,1);%%%%%   for second term of 18 formula in paper
          
              %%%%%% Every vector Rule is= [feature1,feature2,fuzzyset1,fuzzyset2,class]
   
   index3=1;
     for i=1:9
         for j=i+1:9
             for k=1:14
                 for m=1:14
                     conf2=zeros(214,3);%%%% 214 is the num of pattern in glass ds
                     for k1=1:size(glassDs,1)
                          conf2(k1,1)=MF(glassDs(k1,i),k);%%%%%feature i is with fuzzyset k
                          conf2(k1,2)=MF(glassDs(k1,j),m);%%%%%%feature j is with fuzzyset m
                          conf2(k1,3)=conf2(k1,1)*conf2(k1,2);
                     end
                     gclasstotal=sum(conf2(:,3));
                     gclass1=sum(conf2(1:70,3));
                     gclass2=sum(conf2(71:146,3));
                     gclass3=sum(conf2(147:163,3));
                     gclass5=sum(conf2(164:176,3));
                     gclass6=sum(conf2(177:185,3));
                     gclass7=sum(conf2(186:214,3));
                     mx1=gclass1/gclasstotal;
                     mx2=gclass2/gclasstotal;
                     mx3=gclass3/gclasstotal;
                     mx5=gclass5/gclasstotal;
                     mx6=gclass6/gclasstotal;
                     mx7=gclass7/gclasstotal;
                     mxarry=[mx1 mx2 mx3 mx5 mx6 mx7];
                     [maxm2,indmx2]=max(mxarry);
                     if(indmx2==1)
                          glassRB2(index3,5)=1;                 
                          mu2friend(index3)=class1;
                          mu2enemy(index3)=class2+class3+class5+class6+class7;
                     end
                     if(indmx2==2)
                          glassRB2(index3,5)=2;
                          mu2friend(index3)=class2;
                          mu2enemy(index3)=class1+class3+class5+class6+class7;
                     end
                     if(indmx2==3)
                          glassRB2(index3,5)=3;
                          mu2friend(index3)=class3;
                          mu2enemy(index3)=class1+class2+class5+class6+class7;
                     end
                     if(indmx2==4)%%%%%% there is not class 4 in glass dataset
                           glassRB2(index3,5)=5;
                           mu2friend(index3)=class5;
                           mu2enemy(index3)=class1+class2+class3+class6+class7;
                     end
                     if(indmx2==5)%%%%%% there is not class 4 in glass dataset
                           glassRB2(index3,5)=6;
                           mu2friend(index3)=class6;
                           mu2enemy(index3)=class1+class2+class3+class5+class7;
                     end
                     if(indmx2==6)%%%%%% there is not class 4 in glass dataset
                           glassRB2(index3,5)=7;
                           mu2friend(index3)=class7;
                           mu2enemy(index3)=class1+class2+class3+class5+class6;
                     end
             
                     index3=index3+1;          
                     

                 end
             end
         end
     end
   
     
     %% Step3         Select Rules from Candidate Rules according to the paper  Q=40 from each class 
     
     
     
   m1=size(glassRB1,1);
   m2=size(glassRB2,1);
   m=m1+m2;
   n=size(glassRB2,2); 
   glassRB=zeros(m,n+1);%%%% the last ellement is e(Rj) from 18 formula in paper
   Acnt=gcn;
   glassRB(1:m2,1:end-1)=glassRB2;
   glassRB(m2+1:end,3:5)=glassRB1;
   
     %%%%%  calculate points for each Rule
   e1=mufriend-muenemy;  %%%%%%%%%% the 18 formula in paper
   e2=mu2friend-mu2enemy; %%%%%%%%%% the 18 formula in paper
   
   glassRB(1:7056,6)=e2;
   glassRB(7057:end,6)=e1;
 
 
%% sorting and select Q=40 Rules from each class
   glassclass1=-1000*ones(40,6);
   glassclass2=-1000*ones(40,6);
   glassclass3=-1000*ones(40,6);
   glassclass5=-1000*ones(40,6);
   glassclass6=-1000*ones(40,6);
   glassclass7=-1000*ones(40,6);
  
   [m,n]=size(glassRB);

   
   for i=1:m
       [minclass1,ind1]=min(glassclass1(:,6));
       [minclass2,ind2]=min(glassclass2(:,6));
       [minclass3,ind3]=min(glassclass3(:,6));
       [minclass5,ind5]=min(glassclass5(:,6));
       [minclass6,ind6]=min(glassclass6(:,6));
       [minclass7,ind7]=min(glassclass7(:,6));
       
       if((glassRB(i,5)==1) && (glassRB(i,6)>=minclass1))
             glassclass1(ind1,:)= glassRB(i,:);
       end
       
       if((glassRB(i,5)==2) && (glassRB(i,6)>=minclass2))
           glassclass2(ind2,:)= glassRB(i,:);
       end
       
       if((glassRB(i,5)==3) && (glassRB(i,6)>=minclass3))
           glassclass3(ind3,:)= glassRB(i,:);
       end
       if((glassRB(i,5)==5) && (glassRB(i,6)>=minclass5))
           glassclass5(ind5,:)= glassRB(i,:);
       end
       if((glassRB(i,5)==6) && (glassRB(i,6)>=minclass6))
           glassclass6(ind6,:)= glassRB(i,:);
       end
       if((glassRB(i,5)==7) && (glassRB(i,6)>=minclass7))
           glassclass7(ind7,:)= glassRB(i,:);
       end      
       
       
   end
   
   
   %%%%%% Final 240 RuleBase from glass Dataset is glassRBFinal
   glassRBFinal=zeros(240,5);
   glassRBFinal(1:40,:)=glassclass1(:,1:5);
   glassRBFinal(41:80,:)=glassclass2(:,1:5);
   glassRBFinal(81:120,:)=glassclass3(:,1:5);
   glassRBFinal(121:160,:)=glassclass5(:,1:5);
   glassRBFinal(161:200,:)=glassclass6(:,1:5);
   glassRBFinal(201:240,:)=glassclass7(:,1:5);
   
   
   %% Step4             Rule weigting and Rule Reduction in Single Winner Method
   
   CF=ones(240,1);  %%%% the final weigth(certainty factor) of selected rules
   CFtemp=ones(240,1);
   
   %%%%%%% calculate Mu for 240 Rules and 214 patterns
   
   MuRules=zeros(240,214);%%% 240 rules and 214 patterns
   
   for i=1:240
       for j=1:214
           
            if((glassRBFinal(i,1)~=0) && (glassRBFinal(i,2)~=0) )%%%%%  that means two antecedent Rules
                 mu1=MF(glassDs(j,glassRBFinal(i,1)),glassRBFinal(i,3));
                 mu2=MF(glassDs(j,glassRBFinal(i,2)),glassRBFinal(i,4));
                 MuRules(i,j)=mu1*mu2;  %%%% Tnorm is product
        
            end
           if((glassRBFinal(i,1)==0) && (glassRBFinal(i,2)==0) )%%%%%  that means one antecedent Rules
                 MuRules(i,j)=MF(glassDs(j,glassRBFinal(i,3)),glassRBFinal(i,4));        
           end
           
       end
   end


   %%%%%% start 3 iteration for glass dataset
   
for i1=1:3
  
%      AccT=38*ones(3*240,1);%%%%%%% Accuracy in sigle winner=1
     AccT=zeros(3*240,1);%%%%%%% Accuracy in sigle winner=1
%      ErrateT=38*ones(3*240,1);%%%%%%% Error rate in single winner=1
     ErrateT=zeros(3*240,1);%%%%%%% Error rate in single winner=1

     indexacc=1;
  

    AccTotal1=zeros(240,1);%%%%%%% Accuracy 
    ErrateTotal1=zeros(240,1);%%%%%%% Error rate 


    for i=1:240

        CF(i)=0;
        infinit=zeros(214,3);%%%%% for inf weigth 
        infinit(:,2)=glassLabel;
        z=zeros(214,3);%%%%% for 0 weigth
        z(:,2)=glassLabel;
        Itemp=zeros(214,1);%%%% for I set in Paper
        %%%%% weghting with infinit weight
        CFtemp(i)=1000;  %%%%%  
        for j=1:214
            temp=zeros(214,1);
            temp=MuRules(:,j).*CFtemp;
            [m1,ind1]=max(temp);%%%%%% single winner
            infinit(j,1)=glassRBFinal(ind1,5);%%%% the 5th ellement is class of each rule      
            if(infinit(j,1)==infinit(j,2))
                infinit(j,3)=1;%%%%%%%% 1 means correct classification
            else
                infinit(j,3)=0;%%%%%%%% 0 means miss classification
            end
        end


        %%%%% weghting with 0 weight
        CFtemp(i)=0;
        for j=1:214
            temp1=zeros(214,1);
            temp1=MuRules(:,j).*CFtemp;
            [m2,ind2]=max(temp1);
            z(j,1)=glassRBFinal(ind2,5);%%%% the 5th ellement is class of each rule      
            if(z(j,1)==z(j,2))
                   z(j,3)=1;%%%%%%%% 1 means correct classification
            else
                   z(j,3)=0;%%%%%%%% 0 means miss classification
            end
        end


        %%%%%%%%% pick TF and FT from 0 and infinit 

        for j=1:214
            if(infinit(j,3) ~= z(j,3))%%%% Means we keep TF and FT classification
                Itemp(j)=1;
            else
                Itemp(j)=0;
            end
        end


        m=sum(Itemp);
        if(m==0)
            CF(i)=0;
            CFtemp(i)=0;
            continue;
        end
        I=zeros(m,1);%%%%% I is the index of patterns that paper algorithm selected ---- TF and FT 
        ind=1;
        for j=1:214
            if(Itemp(j)==1)
                I(ind)=j; %%%%% I is the index of pattern
                ind=ind+1;
            end
        end


        %%%%%%%% calculate score for each ellement of I set or selected rule 
        m=sum(Itemp);
        Iscore=zeros(m,2);
        Iscore(:,1)=I;
        mx=0;
        for k=1:m
            for j=1:240
                if(glassRBFinal(j,5)~=glassRBFinal(i,5)) 
                    phi=CF(j)*MuRules(j,Iscore(k,1));
                    if(mx>phi)
                        phi=mx;
                    end
                    mx=phi;
                end            
            end
            Iscore(k,2)=(phi/MuRules(i,Iscore(k,1)));%%%%  1 in term1 murules ---->i
        end

       Iscore=sortrows(Iscore,2);%%%%% sort of Scores   
       %%%%%%%%%%%%%%% end of score for Single winner

       %%%%%% implementation Algorithm for finding the best threshold   

       epsilon=0.01;   
       m=size(Iscore,1);
       threshold=zeros(m+1,1);%%%%%%threshold array
       threshold(1)=0;
       threshold(m+1)=Iscore(m,2)+epsilon;

       for j=2:m
            threshold(j)=(Iscore(j-1,2)+Iscore(j,2))/2;
       end

       Acc=zeros(m+1,1);%%%%%% Accuracy in one iteration
       errate=zeros(m+1,1);%%%%%%%% Error rate in one iteration
       errate2=zeros(m+1,1);%%%%%%%% Error rate in one iteration

       TP=0;
       TN=0;
       FP=0;
       FN=0;


       for j=1:m+1  %%%%loop in threshold
            for k=1:m  %%%%%%loop in Score(Iscore)
                if((Iscore(k,2))<threshold(j))%%%% patterns are top of threshold
                    if((glassLabel(Iscore(k,1))==glassRBFinal(i,5)))
                          TP=TP+1;
                         % FN=FN+1;
                         %TN=TN+1;

                    end
                    if(glassLabel(Iscore(k,1))~=glassRBFinal(i,5))
                         FN=FN+1;
                         %TP=TP+1;
                         %FP=FP+1;
                    end

                end
                if((Iscore(k,2))>threshold(j))%%%% patterns are bottom up threshold
                     if(glassLabel(Iscore(k,1))~=glassRBFinal(i,5))
                          TN=TN+1;
                          %FP=FP+1;
                         % TP=TP+1;
                     end
                     if(glassLabel(Iscore(k,1))==glassRBFinal(i,5))
                         FP=FP+1;
                         %TN=TN+1;
                         %FN=FN+1;
                     end
                end
            end
            Acc(j)=(TP+TN)/m;
            errate(j)=(FP+FN)/m;
            errate2(j)=1-Acc(j);     
            TP=0;
            TN=0;
            FP=0;
            FN=0;
        end


       [Accmax,Accind]=max(Acc);    
       CF(i)=threshold(Accind);

        AccTotal1(i)=Accmax;%%%%% best Acc in one iteration
        ErrateTotal1(i)=errate(Accind);%%%%% best Error rate in one iteration

        CFtemp(i)=CF(i);  %%%%%    

        
         AccT(indexacc)= AccTotal1(i)  ;%%%%%%% Accuracy in sigle winner=1
         ErrateT(indexacc)=  ErrateTotal1(i) ;%%%%%%% Error rate in single winner=1
         indexacc=indexacc+1;
        

    end %%%%% end of 240 pattern in one iteration

 




       
       
end %%%%% end of 3 iteration
   

   
   
cnt=0;
for j=1:240
   if(CF(j)~=0)
       cnt=cnt+1;
   end
end   
   
%    if(cnt<glr)
%        for j1=1:240
%           if(CF(j1)==g && cnt<glr )
%                 CF(j1*8)=rand ;
%                 cnt=cnt+1;
%            end
%        end
%    end

glassSWRB=zeros(cnt,6);%%%% Final Rule Base in Single Winner
ErrateT=sort(ErrateT,'descend');
CFSW=zeros(cnt,1);%%%% that is CF final = Weigths in Single winner
index=1;
for j=1:240
    if(CF(j)~=0)
       glassSWRB(index,end)=j;
       glassSWRB(index,1:end-1)=glassRBFinal(j,:);
       CFSW(index)=CF(j);%%%% that is CF final = Weigths in Single winner
       index=index+1;
    end
end
   
   

%%%%%%%%%% Display Final Rules
disp('***********************************');
disp('***********************************');
disp('************** Final Rule Base in Single Winner without leave one out is *****************');
disp('*******************************************************************************************');
disp('structure of each Rule in two antecedent is:');
disp('***** Feature Feature Fuzzyset Fuzzyset Class *****');
disp('***********************************');
disp('structure of each Rule in one antecedent is:');
disp('***** 0 0 Feature Fuzzyset Class*****');
disp('***********************************');

disp('Feature Feature Fset Fset Class');
glassSWrb=glassSWRB(:,1:end-1);
disp(glassSWrb);

   



%% Step5  Test all training data by selected rules from befor step
r=size(CFSW,1);
MuSW=zeros(r,214);
for i=1:r
    for j=1:214%%%%num of pattern
        
         if((glassSWRB(i,1)~=0) && (glassSWRB(i,2)~=0) )%%%%%  that means two antecedent Rules
             mu1=MF(glassDs(j,glassSWRB(i,1)),glassSWRB(i,3));
             mu2=MF(glassDs(j,glassSWRB(i,2)),glassSWRB(i,4));
             MuSW(i,j)=mu1*mu2;  %%%% Tnorm is product
        
         end
         if((glassSWRB(i,1)==0) && (glassSWRB(i,2)==0) )%%%%%  that means one antecedent Rules
             MuSW(i,j)=MF(glassDs(j,glassSWRB(i,3)),glassSWRB(i,4));        
         end
        
    end
end
classSW=zeros(214,3);
classSW(:,2)=glassLabel;
for i=1:214
    temp=zeros(214,1);
    temp=MuSW(:,i).*CFSW;
    [mx1,index1]=max(temp);%%%%%% single winner
    classSW(i,1)=glassSWRB(index1,5);%%%% the 5th ellement is class of each rule   
    if(classSW(i,1)==classSW(i,2))
            classSW(i,3)=1;%%%%%%%% 1 means correct classification
        else
            classSW(i,3)=0;%%%%%%%% 0 means miss classification
    end    
    
end
    

for i=1:214
    if(classSW(i,3)==1)
        Acnt=Acnt+1;
    end
end

AccFinalSW=Acnt/214;

disp('************************************************');
disp('************************************************');
disp('************************************************');
disp('Accuracy in Glass dataset on all training data with Single Winner without leave one out is');
disp('***********************************');
disp('***********************************');

disp(['Accuracy in Glass dataset Single Winner without leave one out is: ' num2str(AccFinalSW)]);

   
   

%% Plot


figure(1)
plot(ErrateT,'b');
% xlim([0 300])
xlabel('Iteration Number');
ylabel('Error Rate');
legend('Error Rate=Blue');
title('Glass Dataset single winner');
   
   
   
   

