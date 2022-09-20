 
 
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

sonar=load('Sonar.mat');%%%% Sonar have 60 features, 208 patterns and 2 class

sonarDs=cell2mat(sonar.allSonarData(:,1:end-1));
sonarLabel2=cell2mat(sonar.allSonarData(:,end));
p=size(sonarDs,1);
sonarLabel=zeros(p,1);
for i=1:p
    if(sonarLabel2(i)=='R')
        sonarLabel(i)=1;
    else
        sonarLabel(i)=2;
    end
end


%%%%%%%%%%% Normalize sonar
normalsonar=sonarDs;
[m1,n1]=size(sonarDs);
minmax=zeros(2,n1);%%%%%first row is min and second row is max
const=25.96;

for i=1:n1
    minmax(1,i)=min(sonarDs(:,i));
    minmax(2,i)=max(sonarDs(:,i));
end

for i=1:n1
    for j=1:m1
      normalsonar(j,i)=(normalsonar(j,i)-minmax(1,i))/(minmax(2,i)-minmax(1,i));          
    end
end

sonarDs2=normalsonar;

% sonarDs=sonarDs2;
%% Step2 ============= Rule Generation

%% #1:Rule Generation for sonar Dataset 

     %% %%%%%%% Rule Generator with one antecedente or length=1
     %%%% the number of these rules is 60*14=840
     
          sonarRB1=zeros(60*14,3);  %%%%%first ellement is Feature second ellement is Fuzzy set and last ellement is Class
              
          
          ind=1;
          for i=1:60
              for j=1:14
                  sonarRB1(ind,1)=i;
                  sonarRB1(ind,2)=j;
                  ind=ind+1;
              end
          end
          
          SRB1=sonarRB1;
          
          %% %%%%%%% Rule Generator with two antecedente or length=2
    %%%% in sonar dataset we only generate rules with one antecedente
          
       %% Start leave one out 

       Rulescnt=zeros(208,1);
       Accuracy=zeros(208,1);
       sonarDstemp=sonarDs;
       sLabel=sonarLabel;
       
       
 for it=1:208
            test=sonarDstemp(it,:);
            testLabel=sLabel(it);
            sonarDstemp(it,:)=[];
            sonarDstrain=sonarDstemp;
            sLabel(it)=[];
     %%%%%%% Define Class Label for every Rule
            mufriend=zeros(840,1);%%%%%  for first term of 18 formula in paper
            muenemy=zeros(840,1);%%%%%   for second term of 18 formula in paper
            index=1;
            for i=1:60
                for j=1:14
                    conf=zeros(208,1);
                    for k=1:size(sonarDs,1)
                        if(k==it)
                            conf(k)=0;%%%%%%%%% means one pattern cosider as a test
                        else
                            conf(k)=MF(sonarDs(k,i),j);
                        end
                    end
                    classtotal=sum(conf);
                    class1=sum(conf(1:97));
                    class2=sum(conf(98:208));
                    max1=class1/classtotal;
                    max2=class2/classtotal;
                    
                    maxarry=[max1 max2];
                    [maxm,indmx]=max(maxarry);
                    if(indmx==1)
                        sonarRB1(index,3)=1;
                        mufriend(index)=class1;
                        muenemy(index)=class2;
                    end
                    if(indmx==2)
                        sonarRB1(index,3)=2;
                        mufriend(index)=class2;
                        muenemy(index)=class1;
                    end                  
                                        
                    index=index+1;
                end%%%%%%%% end of 14 for
            end%%%%%%%%%%end of 60 for
            
            
            %% Step3         Select Rules from Candidate Rules   usually Q=100 from each class 
            
                m=size(sonarRB1,1);
                n=size(sonarRB1,2);
                sonarRB=zeros(m,n+1);%%%% the last ellement is e(Rj) from 18 formula in paper


                %%%%%  calculate points for each Rule
                e=mufriend-muenemy;  %%%%%%%%%% the 18 formula in paper
                sonarRB(:,1:3)=sonarRB1;
                sonarRB(:,4)=e;%%%%%%%%%% the 18 formula in paper

   %% sorting and select Q=100 Rules from each class
                sonarclass1=-1*ones(100,4);
                sonarclass2=-1*ones(100,4);

                [m1,n1]=size(sonarRB);

                for i=1:m1
                   [minclass1,ind1]=min(sonarclass1(:,4));
                   [minclass2,ind2]=min(sonarclass2(:,4));
                   if((sonarRB(i,3)==1) && (sonarRB(i,4)>=minclass1))
                       sonarclass1(ind1,:)= sonarRB(i,:);
                   end
                   if((sonarRB(i,3)==2) && (sonarRB(i,4)>=minclass2))
                       sonarclass2(ind2,:)= sonarRB(i,:);
                   end

                end
   
 %%%%%% Final 200 RuleBase from sonar Dataset is sonarRBFinal
 
 
               sonarRBFinal=zeros(200,3);
               sonarRBFinal(1:100,:)=sonarclass1(:,1:3);
               sonarRBFinal(101:end,:)=sonarclass2(:,1:3);


            
            %% Step4             Rule weigting and Rule Reduction in Single Winner Method

            
            CF=ones(200,1);  %%%% the final weigth(certainty factor) of selected rules
            CFtemp=ones(200,1);
            
            %%%%%%% calculate Mu for 200 Rules
            MuRules=zeros(200,207);
            
            
            for i=1:200
                for j=1:207
                    
                    %%%%%  that means one antecedent Rules
                    MuRules(i,j)=MF(sonarDs(j,sonarRBFinal(i,1)),sonarRBFinal(i,2));
                    
                end
            end
            
            
            AccTotal1=zeros(200,1);%%%%%%% Accuracy in sigle winner
            ErrateTotal1=zeros(200,1);%%%%%%% Error rate in single winner
            
            for i=1:200  %%%%% 200 is the number of rules and 207 is the number of patterns
                    CF(i)=0;  %%%%%%%%%%% 
                    infinit=zeros(207,3);%%%%% for inf weigth   
                    infinit(:,2)=sLabel;
                    z=zeros(207,3);%%%%%% for zero weigth
                    z(:,2)=sLabel;
                    Itemp=zeros(207,1);%%%% for I set in Paper
                    CFtemp(i)=1000;  %%%%%  
                    for j=1:207
                        temp=zeros(200,1);
                        flag1=0;
                        temp=MuRules(:,j).*CFtemp;
                        [mx1,index1]=max(temp);%%%%%% single winner
              %%%%%%%% checking for having two max with diff consequent
                        for l=1:size(temp,1)
                             if((mx1==temp(l)) && (index1~=l) && (sonarRBFinal(l,3)~= sonarRBFinal(index1,3)) )
                                 flag1=1;
                                 infinit(j,3)=-1;
                             end
                        end   
                         if(flag1==0)
                                infinit(j,1)=sonarRBFinal(index1,3);%%%% the 5th ellement is class of each rule      
                                if(infinit(j,1)==infinit(j,2))
                                    infinit(j,3)=1;%%%%%%%% 1 means correct classification
                                else
                                    infinit(j,3)=0;%%%%%%%% 0 means miss classification
                                end
                         end
                    end
                    
                    CFtemp(i)=0;  %%%%%  
                    for j=1:207
                        temp1=zeros(200,1);
                        flag2=0;
                        temp1=MuRules(:,j).*CFtemp;
                        [mx2,index2]=max(temp1);
                        %%%%%%%% checking for having two max with diff consequent
                        for l=1:size(temp1,1)
                             if((mx2==temp1(l)) && (index2~=l) && (sonarRBFinal(l,3)~= sonarRBFinal(index1,3)) )
                                 flag2=1;
                                 z(j,3)=-1;
                             end
                        end 
                        if(flag2==0)
                            z(j,1)=sonarRBFinal(index2,3);%%%% the 3th ellement is class of each rule
                            if(z(j,1)==z(j,2))
                                z(j,3)=1;%%%%%%%% 1 means correct classification
                            else
                                z(j,3)=0;%%%%%%%% 0 means miss classification
                            end
                        end
                    end
                    
                    for j=1:207
                        if((infinit(j,3)==0 && z(j,3)==1) || (infinit(j,3)==1 && z(j,3)==0))%%%% Means we keep TF and FT classification
                            Itemp(j)=1;
                        else
                            Itemp(j)=0;
                        end
                    end
                    
                    m=sum(Itemp);
                    if(m==0)
                        CF(i)=0;
                        continue;
                    end
                    
                    I=zeros(m,1);%%%%% I is the index of patterns that paper algorithm selected ---- TF and FT 
                    ind=1;
                    for j=1:207
                        if(Itemp(j)==1)
                            I(ind)=j; %%%%% I is the index of pattern
                            ind=ind+1;
                        end
                    end
                    
                    %%%%%%%% calculate score for selected rule in Single Winner 
                    Iscore=zeros(m,2);
                    Iscore(:,1)=I;
                    m=sum(Itemp);
                    mx=0;
                    for k=1:m
                        for j=1:200
                            if(sonarRBFinal(j,3)~=sonarRBFinal(i,3))  %%%
                                phi=CF(j)*MuRules(j,Iscore(k,1));
                                if(mx>phi)
                                    phi=mx;
                                end
                                mx=phi;
                            end            
                        end
                        Iscore(k,2)=(phi/MuRules(i,Iscore(k,1)));%%%%  
                    end

                   Iscore=sortrows(Iscore,2);%%%%% sort of Scores   
                   %%%%%%%%%%%%%%% end of score for Single winner
                   
                   %%%%%% implementation Algorithm for finding the best threshold

                   epsilon=0.0001;
                   m=size(Iscore,1);
                   threshold=zeros(m+1,1);%%%%%%threshold array
                   threshold(1)=0;
                   threshold(m+1)=Iscore(m,2)+epsilon;
                   for j=2:m
                       threshold(j)=(Iscore(j-1,2)+Iscore(j,2))/2;
                   end
                   
                   
                   Acc=zeros(m+1,1);%%%%%% Accuracy in one iteration
                   errate=zeros(m+1,1);%%%%%%%% Error rate in one iteration
                   TP=0;
                   TN=0;
                   for j=1:m+1  %%%%loop in threshold
                       for k=1:m  %%%%%%loop in Score(Iscore)
                           if((Iscore(k,2))<threshold(j))%%%% patterns are top of threshold
                               %if((wineLabel(Iscore(k,1)==wineRBFinal(i,5))))
                               if((sLabel(Iscore(k,1)))==(sonarRBFinal(i,3)))
                                   TP=TP+1;
                               end
                               
                           end
                           if((Iscore(k,2))>threshold(j))%%%% patterns are bottom up threshold
                               if((sLabel(Iscore(k,1))~=sonarRBFinal(i,3)))
                                   TN=TN+1;
                               end
                           end
                       end
                       Acc(j)=(TP+TN)/m;
                       errate(j)=1-Acc(j);
                       TP=0;
                       TN=0;
                   end
                   
                   [Accmax,Accind]=max(Acc);
                   
                   CF(i)=threshold(Accind);
                   
                   AccTotal1(i)=Accmax;%%%%% best Acc in one iteration
                   ErrateTotal1(i)=errate(Accind);%%%%% best Error rate in one iteration
                   
                   CFtemp(i)=CF(i);  %%%%%
                                                 
                    
            end%%%%%%%% end of for 200 
            
            cnt=0;
            for j=1:200
               if(CF(j)~=0)
                   cnt=cnt+1;
               end
            end
            
            Rulescnt(it)=cnt;%%%%%%%%% cnt is the number of rules
            
            
            sonarSWRB=zeros(cnt,3);%%%% Final Rule Base in Single Winner
            CFSW=zeros(cnt,1);%%%% that is CF final = Weigths in Single winner
            index=1;
            for j=1:200
                if(CF(j)~=0)
                    sonarSWRB(index,1:end)=sonarRBFinal(j,:);
                    CFSW(index)=CF(j);%%%% that is CF final = Weigths in Single winner
                    index=index+1;
                end
            end

            
                     %% Step5  Test  by selected rules from befor step

                     
                     r=size(CFSW,1);
                     MuSW=zeros(r,1);
                     
                     for i=1:r
                        %%%%%  that means one antecedent Rules
                             MuSW(i)=MF(sonarDs(it,sonarSWRB(i,1)),sonarSWRB(i,2));                          
                     end
                     
                     
                     temp=MuSW.*CFSW;
                     [mx1,index1]=max(temp);%%%%%% single winner
                     lbl=sonarSWRB(index1,3);%%%% the 5th ellement is class of each rule
                     if(lbl==testLabel)
                         Accuracy(it)=1;
                     else
                         Accuracy(it)=0;
                     end
     
        
             if(it==208)
                        Accuracy(it)=Accuracy(it)+const;
             end
             sonarDstrain=sonarDs; %%%% this line should be the last line in 178 iter of for 
             sonarDstemp=sonarDs; %%%% this line should be the last line in 178 iter of for 
             sLabel=sonarLabel;%%%% this line should be the last line in 178 iter of for
             sonarRB1=SRB1;%%%% this line should be the last line in 178 iter of for     
 end%%%%end of for  leave one out
       
        RuleSW=sum(Rulescnt)/208;
        AccFinalSW=sum(Accuracy)/208;

        %%%%%%%%%% Display Average of  Final Rules
        disp('***********************************');
        disp('***********************************');
        disp('************** Average Rule Base in Single Winner with leave one out is *****************');
        disp(RuleSW);
        disp('************************************************');
        disp('************************************************');
        disp('************************************************');
        disp('Average Accuracy in Wine dataset on test data with Single Winner with leave one out is');
        disp(AccFinalSW);

       
       
