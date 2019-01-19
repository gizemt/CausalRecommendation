% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
tic
clear all;close all;
rand('state',0); 
randn('state',0); 
restart = 1;
if restart==1 
  restart=0;
  epsilon=10; % Learning rate 
  lambda  = 0.0001; % Regularization parameter 
  momentum=0.8; 

  epoch=1; 
  maxepoch=50; 

  load moviedata100k_movies % Triplets: {user_id, movie_id, rating} 
%   mean_rating = mean(train_vec(:,3)); 
 
  pairs_tr = length(train_vec); % training data 
%   nl3 = sum(probe_vec(:,3) <= 3);
%   idx_g3 = find(probe_vec(:,3) > 3);
%   probe_vec = [probe_vec(probe_vec(:,3) <= 3,:); probe_vec(idx_g3(1:nl3),:)];
  pairs_pr = length(probe_vec); % validation data
  N=10000; % number training triplets per batch 


  numbatches = floor(pairs_tr/N); % Number of batches 
  num_m = length(unique([train_vec(:,2); probe_vec(:,2)]));  % Number of movies 
  num_p = length(unique([train_vec(:,1); probe_vec(:,1)]));  % Number of users 
  num_feat = 30; % Rank 10 decomposition 

  w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors
  w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators
  w1_M1_inc = zeros(num_m, num_feat);
  w1_P1_inc = zeros(num_p, num_feat);

end


for epoch = epoch:maxepoch
  rr = randperm(pairs_tr);
  train_vec = train_vec(rr,:);
  clear rr 

  for batch = 1:numbatches
    fprintf(1,'epoch %d batch %d \r',epoch,batch);
    
    aa_p   = double(train_vec((batch-1)*N+1:batch*N,1));
    aa_m   = double(train_vec((batch-1)*N+1:batch*N,2));
    rating = double(train_vec((batch-1)*N+1:batch*N,3));
    rating = rating > 3;
    mean_rating = mean(rating);

    rating = rating-mean_rating; % Default prediction is the mean rating. 

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
    f(epoch) = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

    %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    IO = repmat(2*(pred_out - rating),1,num_feat);
    Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
    Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

    dw1_M1 = zeros(num_m,num_feat);
    dw1_P1 = zeros(num_p,num_feat);

    for ii=1:N
      dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
      dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
    end

    %%%% Update movie and user features %%%%%%%%%%%

    w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
    w1_M1 =  w1_M1 - w1_M1_inc;

    w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
    w1_P1 =  w1_P1 - w1_P1_inc;
  end 

  %%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
  f_s(epoch) = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
  err_train(epoch) = sqrt(f_s(epoch)/N);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
  NN=pairs_pr;

  aa_p = double(probe_vec(:,1));
  aa_m = double(probe_vec(:,2));
  rating = double(probe_vec(:,3));
  rating = rating > 3;

  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
%   ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions 
%   ff = find(pred_out<1); pred_out(ff)=1;
  ff = find(pred_out>0); pred_out(ff)=1; % Clip predictions 
  ff = find(pred_out<=0); pred_out(ff)=0;

  err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/NN);
  fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
              epoch, batch, err_train(epoch), err_valid(epoch));
  if ( epoch > 1 && err_valid(epoch-1) < err_valid(epoch)) 
      break;
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   if (rem(epoch,10))==0
%      save pmf_weight w1_M1 w1_P1
%   end

end 
toc
%%
figure,plot(1:length(err_train), err_train, 1:length(err_train), err_valid);
xlabel('Epochs');title(sprintf('Error - Lr=%.4f, lambda=%.4f, momentum=%.4f, num_feat=%d', ...
epsilon, lambda, momentum, num_feat));
legend('Training error', 'Validation error');
% saveas(gcf, sprintf('errors_C%d_Lr=%.4f_l%.4f_m%.4f_nf=%d_reg_m%.3f_reg_p%.3f.fig', ...
% causal, epsilon, lambda, momentum, num_feat, reg_m, reg_p));
figure,plot(1:length(f), f);
xlabel('Epochs');title(sprintf('Loss - Lr=%.4f, lambda=%.4f, momentum=%.4f, num_feat=%d', ...
epsilon, lambda, momentum, num_feat));


