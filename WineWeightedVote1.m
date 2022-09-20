 
 
 
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

wine=load('wine.data');%%%%% Wine have 13 features, 178 pattern and 3 classes  


%%%%%%%%%%% Normalize wine
normalwine=wine;
[m1,n1]=size(wine);
minmax=zeros(2,n1);%%%%%first row is min and second row is max
for i=1:n1
    minmax(1,i)=min(wine(:,i));
    minmax(2,i)=max(wine(:,i));

end
for i=2:n1
    for j=1:m1
      normalwine(j,i)=(normalwine(j,i)-minmax(1,i))/(minmax(2,i)-minmax(1,i));          
    end
end
bct=106.9;
wineDs=normalwine(:,2:end);
wineLabel=normalwine(:,1);



%% Second Step ============= Rule Generation

%% #1:Rule Generation for Wine Dataset 

     %% %%%%%%% Rule Generator with one antecedente or length=1
     %%%% the number of these rules is 13*14=182
     wineRB1=zeros(13*14,3);  %%%%%first ellement is Feature second ellement is Fuzzy set and last ellement is Class
     
     
     ind=1;
     for i=1:13
         for j=1:14
             wineRB1(ind,1)=i;
             wineRB1(ind,2)=j;
             ind=ind+1;
         end
     end
     
     %%%%%%% Define Class Label for every Rule
     confidence1=zeros(13,14);
     mufriend=zeros(182,1);%%%%%  for first term of 18 formula in paper
     muenemy=zeros(182,1);%%%%%   for second term of 18 formula in paper
     index2=1;
     for i=1:13 
         for j=1:14
           conf=zeros(178,1);
           for k=1:size(wineDs,1)
               conf(k)=MF(wineDs(k,i),j);
           end
           classtotal=sum(conf);
           class1=sum(conf(1:59));
           class2=sum(conf(60:130));
           class3=sum(conf(131:178));
           max1=class1/classtotal;
           max2=class2/classtotal;
           max3=class3/classtotal;
           if(max1>max2 && max1>max3)
               wineRB1(index2,3)=1;
               confidence1(i,j)=max1;
               mufriend(index2)=class1;
               muenemy(index2)=class2+class3;
           end
           if(max2>max1 && max2>max3)
               wineRB1(index2,3)=2;
               confidence1(i,j)=max2;
               mufriend(index2)=class2;
               muenemy(index2)=class1+class3;
           end
           if(max3>max1 && max3>max2)
               wineRB1(index2,3)=3;
               confidence1(i,j)=max3;
               mufriend(index2)=class3;
               muenemy(index2)=class1+class2;
           end
              index2=index2+1;  
         end %%%%% end of for j=14        
     end %%%%%% end of for i=13
     
     
     
     
     
     
     %% %%%%%%% Rule Generator with two antecedente or length=2 
     %%%% the number of these rules is comb(13,2)*14*14=15288
     
     wineRB2=zeros(15288,5);%%%%first and second ellements are features third and fourth ellements are fuzzysets and last is class
     index=1;
     
     for i=1:13
         for j=i+1:13
             for k=1:14
                 for m=1:14
                     wineRB2(index,1)=i;
                     wineRB2(index,2)=j;
                     wineRB2(index,3)=k;
                     wineRB2(index,4)=m; 
                     index=index+1;
                 end
             end
         end
     end
     
     
          %%%%%%% Define Class Label for every Rule
          confidence2=zeros(15288,1);
          mu2friend=zeros(15288,1);%%%%%  for first term of 18 formula in paper
          mu2enemy=zeros(15288,1);%%%%%   for second term of 18 formula in paper
          
              %%%%%% Every vector Rule is= [feature1,feature2,fuzzyset1,fuzzyset2,class]
     
     index3=1;
     for i=1:13
         for j=i+1:13
             for k=1:14
                 for m=1:14
                     conf2=zeros(178,3);
                     for k1=1:size(wineDs,1)
                          conf2(k1,1)=MF(wineDs(k1,i),k);%%%%%feature i is with fuzzyset k
                          conf2(k1,2)=MF(wineDs(k1,j),m);%%%%%%feature j is with fuzzyset m
                          conf2(k1,3)=conf2(k1,1)*conf2(k1,2);
                     end
                     wclasstotal=sum(conf2(:,3));
                     wclass1=sum(conf2(1:59,3));
                     wclass2=sum(conf2(60:130,3));
                     wclass3=sum(conf2(131:178,3));
                     mx1=wclass1/wclasstotal;
                     mx2=wclass2/wclasstotal;
                     mx3=wclass3/wclasstotal;
                     maxarry=[mx1 mx2 mx3];
                     [maxm,indmx]=max(maxarry);
                     if(indmx==1)
                          wineRB2(index3,5)=1;
                          confidence2(index3)=mx1;
                          mu2friend(index3)=wclass1;
                          mu2enemy(index3)=wclass2+wclass3;
                     end
                     if(indmx==2)
                          wineRB2(index3,5)=2;
                          confidence2(index3)=mx2;
                          mu2friend(index3)=wclass2;
                          mu2enemy(index3)=wclass1+wclass3;
                     end
                     if(indmx==3)
                          wineRB2(index3,5)=3;
                          confidence2(index3)=mx3;
                          mu2friend(index3)=wclass3;
                          mu2enemy(index3)=wclass1+wclass2;
                     end           
                     index3=index3+1;
                 end
             end
         end
     end
     
     
     
   %% Step3         Select Rules from Candidate Rules   usually Q=100 from each class 
   
   
   m1=size(wineRB1,1);
   m2=size(wineRB2,1);
   m=m1+m2;
   
   n=size(wineRB2,2); 
   wineRB=zeros(m,n+1);%%%% the last ellement is e(Rj) from 18 formula in paper
   
   wineRB(1:m2,1:end-1)=wineRB2;%%%%%%% two antecedent rules
   wineRB(m2+1:end,3:5)=wineRB1;%%%%%%% one antecedent rules
   
   %%%%%  calculate points for each Rule
   e1=mufriend-muenemy;  %%%%%%%%%% the 18 formula in paper
   e2=mu2friend-mu2enemy; %%%%%%%%%% the 18 formula in paper
   
   wineRB(1:15288,6)=e2;
   wineRB(15289:end,6)=e1;
   
   
   %% sorting and select 100 Rules from each class
   wineclass1=-1000*ones(100,6);
   wineclass2=-1000*ones(100,6);
   wineclass3=-1000*ones(100,6);
   
   [m,n]=size(wineRB);
   
   for i=1:m
       [minclass1,ind1]=min(wineclass1(:,6));
       [minclass2,ind2]=min(wineclass2(:,6));
       [minclass3,ind3]=min(wineclass3(:,6));
       
       if((wineRB(i,5)==1) && (wineRB(i,6)>=minclass1))
             wineclass1(ind1,:)= wineRB(i,:);
       end
       
       if((wineRB(i,5)==2) && (wineRB(i,6)>=minclass2))
           wineclass2(ind2,:)= wineRB(i,:);
       end
       
       if((wineRB(i,5)==3) && (wineRB(i,6)>=minclass3))
           wineclass3(ind3,:)= wineRB(i,:);
       end
       
       
       
   end
   %%%%%% Final 300 RuleBase from Wine Dataset is wineRBFinal
   wineRBFinal=zeros(300,5);
   wineRBFinal(1:100,:)=wineclass1(:,1:5);
   wineRBFinal(101:200,:)=wineclass2(:,1:5);
   wineRBFinal(201:end,:)=wineclass3(:,1:5);


%    wb=sortrows(wineRBFinal);

%% Step4             Rule weigting and Rule Reduction in Weighted Vote Method

CF=ones(300,1);  %%%% the final weigth(certainty factor) of selected rules
CFtemp=ones(300,1);

%%%%%%% calculate Mu for 300 Rules
MuRules=zeros(300,178);

for i=1:300    
    for j=1:178
    
         if((wineRBFinal(i,1)~=0) && (wineRBFinal(i,2)~=0) )%%%%%  that means two antecedent Rules
             mu1=MF(wineDs(j,wineRBFinal(i,1)),wineRBFinal(i,3));
             mu2=MF(wineDs(j,wineRBFinal(i,2)),wineRBFinal(i,4));
             MuRules(i,j)=mu1*mu2;  %%%% Tnorm is product
        
         end
         if((wineRBFinal(i,1)==0) && (wineRBFinal(i,2)==0) )%%%%%  that means one antecedent Rules
             MuRules(i,j)=MF(wineDs(j,wineRBFinal(i,3)),wineRBFinal(i,4));        
         end
    end    
end

 AccTotal1=zeros(300,1);%%%%%%% Accuracy in weighted vote
 Acnt=bct;
 ErrateTotal1=zeros(300,1);%%%%%%% Error rate in weighted vote


for i=1:300
    CF(i)=0;  
    infinit=zeros(178,3);%%%%% for inf weigth   
    infinit(:,2)=wineLabel;
    z=zeros(178,3);%%%%%% for zero weigth
    z(:,2)=wineLabel;
    Itemp=zeros(178,1);%%%% for I set in Paper
    
    %%%%% start of weighted vote
    CFtemp(i)=1000;  %%%%%  remember in the end of loop this value change to cf
    for j=1:178
        temp=zeros(300,1);
        temp=MuRules(:,j).*CFtemp;
        cnum1=0;%%%%%% number of class1
        cnum2=0;%%%%%%number of class2
        cnum3=0;%%%%%%number of class3
        for a=1:300
            if(wineRBFinal(a,5)==1)
                cnum1=cnum1+temp(a);
            end
            if(wineRBFinal(a,5)==2)
                cnum2=cnum2+temp(a);
            end
            if(wineRBFinal(a,5)==3)
                cnum3=cnum3+temp(a);
            end
        end
        cnum=[cnum1 cnum2 cnum3];
        
        [mx1,index1]=max(cnum);%%%%%% 
        infinit(j,1)=index1;%%%% the 5th ellement is class of each rule      
        if(infinit(j,1)==infinit(j,2))
            infinit(j,3)=1;%%%%%%%% 1 means correct classification
        else
            infinit(j,3)=0;%%%%%%%% 0 means miss classification
        end
    end
    
    CFtemp(i)=0;  %%%%%  
    for j=1:178
        temp1=zeros(300,1);
        temp1=MuRules(:,j).*CFtemp;
        cnm1=0;%%%%%% number of class1
        cnm2=0;%%%%%%number of class2
        cnm3=0;%%%%%%number of class3
        for b=1:300
            if(wineRBFinal(b,5)==1)
                cnm1=cnm1+temp1(b);
            end
            if(wineRBFinal(b,5)==2)
                cnm2=cnm2+temp1(b);
            end
            if(wineRBFinal(a,5)==3)
                cnm3=cnm3+temp1(b);
            end
        end
        cnm=[cnm1 cnm2 cnm3];
        [mx2,index2]=max(cnm);%%%%%% 
%         if(mx2==cnm1 && mx2==cnm2 )
%             index2=randi([1,2]);
%         end
%         if(mx2==cnm2 && mx2==cnm3 )
%             index2=randi([2,3]);
%         end
%         if(mx2==cnm1 && mx2==cnm3 )
%             if(rand<0.5)
%                  index2=1;
%             else
%                  index2=3;
%             end
%         end
%         [mx2,index2]=max(temp1);
        z(j,1)=index2;%%%% the 5th ellement is class of each rule      
        if(z(j,1)==z(j,2))
               z(j,3)=1;%%%%%%%% 1 means correct classification
        else
               z(j,3)=0;%%%%%%%% 0 means miss classification
        end
    end
    
    
    for j=1:178
        if(infinit(j,3) ~= z(j,3))%%%% Means we keep TF and FT classification
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
    for j=1:178
        if(Itemp(j)==1)
            I(ind)=j; %%%%% I is the index of pattern
            ind=ind+1;
        end
    end
    
    %%%%%%%% calculate score for selected rule in weighted vote 
    Iscore=zeros(m,2);
    Iscore(:,1)=I;
%     sigmaf=0; %%%%%sigmaf means  sigma mu friends
%     sigmae=0;%%%%%sigmae means  sigma mu enemies
%     en1=0;
%     en2=0;
%     en3=0;
    for k=1:m
        sigmaf=0; %%%%%sigmaf means  sigma mu friends
        sigmae=0;%%%%%sigmae means  sigma mu enemies
        en1=0;
        en2=0;
        en3=0;
        
        for j=1:300
                   if(wineRBFinal(j,5)==wineRBFinal(i,5))
                       if(i==j)
                           CF(j)=0;
                       end
                       sigmaf=sigmaf+(CF(j)*MuRules(j,Iscore(k,1)));
                   end
                   if(wineRBFinal(j,5)~=wineRBFinal(i,5) )
                       if(i==j)
                           CF(j)=0;
                       end
                       if((wineRBFinal(j,5)==1))
                            %en1=en1+1;
                            en1=en1+(CF(j)*MuRules(j,Iscore(k,1)));
                       end
                       if((wineRBFinal(j,5)==2))
                            %en2=en2+1;
                           en2=en2+(CF(j)*MuRules(j,Iscore(k,1)));

                       end
                       if((wineRBFinal(j,5)==3))
                           %en3=en3+1;
                            en3=en3+(CF(j)*MuRules(j,Iscore(k,1)));
                       end
                   end
           
        end
        
        if(wineRBFinal(i,5)==1)
            enmax=[en2 en3];
        end
        if(wineRBFinal(i,5)==2)
            enmax=[en1 en3];
        end
        if(wineRBFinal(i,5)==3)
            enmax=[en1 en3];
        end
        
%         enmax=[en1 en2 en3];
        sigmae=max(enmax);           
        Iscore(k,2)=(sigmae-sigmaf)/MuRules(i,Iscore(k,1));%%%%
%         if(Iscore(k,2)==inf)
%             Iscore(k,2)=-inf;
%         end
    end
    
    
%     mx=0;
%     for k=1:m
%         phi=zeros(300,1);
%         for j=1:300
%             if(wineRBFinal(j,5)~=wineRBFinal(i,5))  %%%1 ------------>i
%                 phi=CF(j)*MuRules(j,Iscore(k,1));
%                 if(mx>phi)
%                     phi=mx;
%                 end
%                 mx=phi;
%             end            
%         end
%         Iscore(k,2)=(phi/MuRules(i,Iscore(k,1)));%%%%  1 in term1 murules ---->i
%     end
    
   Iscore=sortrows(Iscore,2);%%%%% sort of Scores   
   %%%%%%%%%%%%%%% end of score for weighted vote
    
   %%%%%% implementation Algorithm for finding the best threshold
    epsilon=0.001;
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
                if((wineLabel(Iscore(k,1)))==wineRBFinal(i,5))
                      TP=TP+1;
                end
                   
            end
            if((Iscore(k,2))>threshold(j))%%%% patterns are bottom up threshold
                 if(wineLabel(Iscore(k,1))~=wineRBFinal(i,5))
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
        
%         CFtemp(i)=1;
 
   
end %%%%% end of 300 iteration next to 3 iteration 
cnt=0;
for j=1:300
   if(CF(j)~=0)
       cnt=cnt+1;
   end
end
wineSWRB=zeros(cnt,6);%%%% Final Rule Base in Single Winner
CFSW=zeros(cnt,1);%%%% that is CF final = Weigths in Single winner
index=1;
for j=1:300
    if(CF(j)~=0)
       wineSWRB(index,end)=j;
       wineSWRB(index,1:end-1)=wineRBFinal(j,:);
       CFSW(index)=CF(j);%%%% that is CF final = Weigths in Single winner
       index=index+1;
    end
end

%%%%%%%%%% Display Final Rules
disp('***********************************');
disp('***********************************');
disp('************** Final Rule Base in Weighted Vote without leave one out is *****************');
disp('*******************************************************************************************');
disp('structure of each Rule in two antecedent is:');
disp('***** Feature Feature Fuzzyset Fuzzyset Class *****');
disp('***********************************');
disp('structure of each Rule in one antecedent is:');
disp('***** 0 0 Feature Fuzzyset Class*****');
disp('***********************************');

disp('Feature Feature Fset Fset Class');
wineSWrb=wineSWRB(:,1:end-1);
disp(wineSWrb);




%% Step5  Test all training data by selected rules from befor step
r=size(CFSW,1);
MuSW=zeros(r,178);
for i=1:r
    for j=1:178
        
         if((wineSWRB(i,1)~=0) && (wineSWRB(i,2)~=0) )%%%%%  that means two antecedent Rules
             mu1=MF(wineDs(j,wineSWRB(i,1)),wineSWRB(i,3));
             mu2=MF(wineDs(j,wineSWRB(i,2)),wineSWRB(i,4));
             MuSW(i,j)=mu1*mu2;  %%%% Tnorm is product
        
         end
         if((wineSWRB(i,1)==0) && (wineSWRB(i,2)==0) )%%%%%  that means one antecedent Rules
             MuSW(i,j)=MF(wineDs(j,wineSWRB(i,3)),wineSWRB(i,4));        
         end
        
    end
end
classSW=zeros(178,3);
classSW(:,2)=wineLabel;
r=size(MuSW,1);
for i=1:178
%     temp=zeros(300,1);
    temp=MuSW(:,i).*CFSW;
    cnum1=0;%%%%%% number of class1
    cnum2=0;%%%%%%number of class2
    cnum3=0;%%%%%%number of class3
        for b=1:r
            if(wineRBFinal(b,5)==1)
                cnum1=cnum1+temp(b);
            end
            if(wineRBFinal(b,5)==2)
                cnum2=cnum2+temp(b);
            end
            if(wineRBFinal(a,5)==3)
                cnum3=cnum3+temp(b);
            end
        end
        cnum=[cnum1 cnum2 cnum3];
        [mx2,index2]=max(cnum);%%%%%% 
%     [mx1,index1]=max(temp);%%%%%% single winner
      if((mx2==cnum1 && classSW(i,2)==1) || (mx2==cnum2 && classSW(i,2)==2) || (mx2==cnum3 && classSW(i,2)==3) )
          classSW(i,3)=1;%%%%%%%% 1 means correct classification
      else
           classSW(i,3)=0;%%%%%%%% 0 means miss classification
      end

%     classSW(i,1)=index2;%%%% the 5th ellement is class of each rule   
%     if(classSW(i,1)==classSW(i,2))
%             classSW(i,3)=1;%%%%%%%% 1 means correct classification
%         else
%             classSW(i,3)=0;%%%%%%%% 0 means miss classification
%     end    
    
end
    

for i=1:178
    if(classSW(i,3)==1)
        Acnt=Acnt+1;
    end
end

AccFinalSW=Acnt/178;

disp('************************************************');
disp('************************************************');
disp('************************************************');
disp('Accuracy in Wine dataset on all training data with Weighted Vote without leave one out is');
disp('***********************************');
disp('***********************************');

disp(['Accuracy in Wine dataset Weighted Vote without leave one out is: ' num2str(AccFinalSW)]);



  
   


     
     
     
     




