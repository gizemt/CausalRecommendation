clear all;
tic
for num_feat = 30;%[10, 30, 50]
        for lambda = 100;%[0.1, 1, 10, 100]
                for causal = 1%0:1
                    
%                     try

  epoch=1; 
  maxepoch=20; 
  load moviedata100k_movies % Triplets: {user_id, movie_id, rating} 
  mean_rating = mean(train_vec(:,3)); 
  pairs_tr = length(train_vec); % training data
   
%   nl3 = sum(probe_vec(:,3) <= 3);
%   idx_g3 = find(probe_vec(:,3) > 3);
%   probe_vec = [probe_vec(probe_vec(:,3) <= 3,:); probe_vec(idx_g3(1:nl3),:)];
  pairs_pr = length(probe_vec); % validation data
  
  N=50000; % number of training triplets per batch
  numbatches = floor(pairs_tr/N); % Number of batches  
  num_m = length(unique([train_vec(:,2); probe_vec(:,2)]));  % Number of movies 
  num_p = length(unique([train_vec(:,1); probe_vec(:,1)]));  % Number of users 
  
  
  reg_m = 10;
  reg_p = 10;

  w1_M1     = randn(num_m, num_feat+1); % Movie feature vectors
  w1_P1     = randn(num_p, num_feat+1); % User feature vecators
  if causal == 1
      w1_M1 = w1_M1*sqrt(1/lambda);
      w1_P1 = w1_P1*sqrt(1/lambda);
  end
  w1_M1_inc = zeros(num_m, num_feat+1);
  w1_P1_inc = zeros(num_p, num_feat+1);
  
  p_score = zeros(num_m,num_p);
  items = unique(train_vec(:,2));
  p_score(items,:) = repmat(arrayfun(@(i)sum(train_vec(:,2) == i), items), 1, num_p);
  p_score = p_score/num_p;

%   fprintf('Causal=%d, Lr=%.4f, lambda=%.4f, momentum=%.4f, num_feat=%d, reg_m=%.3f, reg_p=%.3f \n', ...
%       causal, epsilon, lambda, momentum, num_feat, reg_m, reg_p);


epoch = 1;
flag = true;
f = [];
err_train_batch = [];
while flag
  rr = randperm(pairs_tr);
  train_vec = train_vec(rr,:);
  clear rr 

  for batch = 1:numbatches

    aa_p   = double(train_vec((batch-1)*N+1:batch*N,1));
    aa_m   = double(train_vec((batch-1)*N+1:batch*N,2));
    rating = double(train_vec((batch-1)*N+1:batch*N,3));
    rating = (rating > 3);

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
    vec_ind = sub2ind(size(p_score), aa_m, aa_p);
    p_score_vec = p_score(vec_ind);
    w1_P1_old = w1_P1;
    for u = 1:num_p
        if sum(aa_p == u) > 0
        idx_mu = aa_m(aa_p == u);
        out_P1 = bsxfun(@times, permute(w1_M1(idx_mu,:), [2 3 1]), permute(w1_M1(idx_mu,:), [3 2 1 ]));
        V_P1 = sum(bsxfun(@times, out_P1, reshape(1./p_score(idx_mu, u),1,1,numel(p_score(idx_mu, u)))), 3) + lambda*eye(num_feat+1);
        Y_P1 = sum(((rating(aa_p == u)./p_score(idx_mu, u))*ones(1,num_feat+1)).*w1_M1(idx_mu,:), 1)';
        w1_P1(u, :) = (inv(V_P1)*Y_P1)';
        end
    end
    for i = 1:num_m
        if sum(aa_m == i) > 0
        idx_pm = aa_p(aa_m == i);
        out_M1 = bsxfun(@times, permute(w1_P1_old(idx_pm,:), [2 3 1]), permute(w1_P1_old(idx_pm,:), [3 2 1 ]));
        V_M1 = sum(bsxfun(@times, out_M1, reshape(1./p_score(i, idx_pm),1,1,numel(p_score(i, idx_pm)))), 3) + lambda*eye(num_feat+1);
        Y_M1 = sum(((rating(aa_m == i)./p_score(i, idx_pm)')*ones(1,num_feat+1)).*w1_P1_old(idx_pm,:), 1)';
        w1_M1(i, :) = (inv(V_M1)*Y_M1)';
        end
    end
    
    
        if causal == 1
        f(end+1) = sum( (1./p_score_vec).*(pred_out - rating).^2 + ...
            0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
        else
            f(end+1) = sum( (pred_out - rating).^2 + ...
            0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
        end
  err_train_batch = [err_train_batch sqrt(sum((pred_out- rating).^2)/length(pred_out));];
  end 

  %%%%%%%%%%%%%% Compute Predictions after Parameter Updates %%%%%%%%%%%%%%%%%
  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
  sum(pred_out>0 == rating)/length(pred_out)
  if causal == 1
  f_s = sum( (1./p_score_vec).*(pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
  else
  f_s = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));    
  end
  err_train_loss(epoch) = sqrt(f_s/N);
  err_train(epoch) = sqrt(sum((pred_out- rating).^2)/length(pred_out));

  %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
  NN=pairs_pr;

  aa_p = double(probe_vec(:,1));
  aa_m = double(probe_vec(:,2));
  rating = double(probe_vec(:,3));
  rating = (rating > 3);

  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
  err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/NN);
  ff = find(pred_out>0); pred_out(ff)=1; % Clip predictions 
  ff = find(pred_out<=0); pred_out(ff)=0;          
  if ( epoch > 1 && err_valid(epoch-1) < err_valid(epoch)) || epoch > maxepoch || isnan(err_valid(end))
      %  
      flag = false;
  else
      epoch = epoch + 1;
  end
          
% if epoch <= maxepoch && epoch > 1
%     fprintf('epoch %4f TrainingRMSE %6.4f %6.4f %d \n', ...
%                   epoch-1, err_train(epoch-1), err_valid(epoch-1), f(epoch-1));
% else
%     fprintf('epoch %4f TrainingRMSE %6.4f %6.4f %d \n', ...
%                   epoch, err_train(epoch), err_valid(epoch), f(epoch));
% end

end

% if epoch <= maxepoch && epoch > 1
%     fprintf('epoch %4f TrainingRMSE %6.4f %6.4f %d \n', ...
%                   epoch-1, err_train(epoch-1), err_valid(epoch-1), f(epoch-1));
% else
%     fprintf('epoch %4f TrainingRMSE %6.4f %6.4f %d \n', ...
%                   epoch, err_train(epoch), err_valid(epoch), f(epoch));
% end  
%                 end
%             end
        end
    end
end
toc
% fclose(fileID);
%% 
%%%%%%%%%%%%%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% movie_mat = zeros(num_m, num_p);
% pred_ind = sub2ind(size(p_score), aa_m, aa_p);
% movie_mat(pred_ind) = pred_out;
% gem_score = pred_out.*(1-p_score(pred_ind));
% pop_score = pred_out.*p_score(pred_ind);
% reccommended = [];
% for user = unique(aa_p)'
%   [~, idx_maxpop] = max(pop_score(aa_p == user));
%   [~, idx_maxgem] = max(gem_score(aa_p == user));
%   [~, idx_maxrat] = max(pred_out(aa_p == user));
%   reccommended = [reccommended; user, aa_m(idx_maxpop), aa_m(idx_maxgem), aa_m(idx_maxrat)];
% end
%%
figure,plot(1:length(err_train), err_train, 1:length(err_train), err_valid);
xlabel('Epochs');
% title(sprintf('Error - Lr=%.4f, lambda=%.4f, momentum=%.4f, num_feat=%d, reg_m=%.3f, reg_p=%.3f', ...
% epsilon, lambda, momentum, num_feat, reg_m, reg_p));
legend('Training error', 'Validation error');
% saveas(gcf, sprintf('errors_C%d_Lr=%.4f_l%.4f_m%.4f_nf=%d_reg_m%.3f_reg_p%.3f.fig', ...
% causal, epsilon, lambda, momentum, num_feat, reg_m, reg_p));
figure,plot(1:length(f), f);
xlabel('Epochs');
% title(sprintf('Loss - Lr=%.4f, lambda=%.4f, momentum=%.4f, num_feat=%d, reg_m=%.3f, reg_p=%.3f', ...
% epsilon, lambda, momentum, num_feat, reg_m, reg_p));
% saveas(gcf, sprintf('loss_C%d_Lr=%.4f_l%.4f_m%.4f_nf=%d_reg_m%.3f_reg_p%.3f.fig', ...
% causal, epsilon, lambda, momentum, num_feat, reg_m, reg_p));
%%
% %%%%%% Recommend one movie to users (pairs that are not present in train&valid dataset) %%%%%%%%%%%%%%%%%%%%%%
% present_m = [train_vec(:,2); probe_vec(:,2)];
% present_p = [train_vec(:,1); probe_vec(:,1)];
% present_lin = sub2ind(size(p_score), present_m, present_p);
% missing_lin = 1:(size(p_score,1)*size(p_score,2));
% missing_lin(present_lin) = [];
% [missing_m, missing_p] = ind2sub(size(p_score), missing_lin);
% rec_user = [];
% for random_user = randi(num_p) % 1:num_p
% missing_m_user = missing_m(missing_p == random_user);
% pred_user = w1_M1(missing_m_user,:)*w1_P1(random_user,:)' + mean_rating;
% pred_user(pred_user>5)=5; % Clip predictions 
% pred_user(pred_user<1)=1;
% idx_user = sub2ind(size(p_score), missing_m_user, random_user*ones(size(missing_m_user)));
% [~, idx_gem_user] = max(pred_user.*(1./p_score(idx_user))');
% [~, idx_pop_user] = max(pred_user.*p_score(idx_user)');
% [~, idx_rat_user] = max(pred_user);
% rec_user = [rec_user; random_user, missing_m_user(idx_pop_user), missing_m_user(idx_gem_user), missing_m_user(idx_rat_user)];
% end
% 
%%
%%%%%% Recommend one movie to users (pairs that are present in valid dataset) %%%%%%%%%%%%%%%%%%%%%%
% present_m = [train_vec(:,2); probe_vec(:,2)];
% present_p = [train_vec(:,1); probe_vec(:,1)];
% present_lin = sub2ind(size(p_score), probe_vec(:,2), probe_vec(:,1));
% missing_lin = 1:(size(p_score,1)*size(p_score,2));
% missing_lin(present_lin) = [];
% [present_m, present_p] = ind2sub(size(p_score), present_lin);
present_m = probe_vec(:,2);
present_p = probe_vec(:,1);
rec_user = [];
for random_user = randi(num_p) % 1:num_p
missing_m_user = present_m(present_p == random_user);
pred_user = w1_M1(missing_m_user,:)*w1_P1(random_user,:)' + mean_rating;
pred_user(pred_user>5)=5; % Clip predictions 
pred_user(pred_user<1)=1;
idx_user = sub2ind(size(p_score), missing_m_user, random_user*ones(size(missing_m_user)));
[~, idx_gem_user] = max(pred_user.*(1./p_score(idx_user)));
[~, idx_pop_user] = max(pred_user.*p_score(idx_user));
[~, idx_rat_user] = max(pred_user);
rec_user = [rec_user; random_user, missing_m_user(idx_pop_user), missing_m_user(idx_gem_user), missing_m_user(idx_rat_user)];
end
%% %% Display random recommendations   %%%%%%%
movies = readtable('movies.csv');
randu = random_user;%randi(num_p);
% fprintf('For user %d \n Popular recommendation: %s (%.2f)\nHidden gem recommendation: %s (%.2f)\nHighest rating recommendation: %s (%.2f)\n',...
%     randu, movies{rec_user(randu, 2), 2}{1}, w1_M1(rec_user(randu, 2),:)*w1_P1(random_user,:)' + mean_rating,...
%            movies{rec_user(randu, 3), 2}{1}, w1_M1(rec_user(randu, 3),:)*w1_P1(random_user,:)' + mean_rating,...
%            movies{rec_user(randu, 4), 2}{1}, w1_M1(rec_user(randu, 4),:)*w1_P1(random_user,:)' + mean_rating);
fprintf('For user %d \n Popular recommendation: %s (%.2f) %.1f \nHidden gem recommendation: %s (%.2f) %.1f\nHighest rating recommendation: %s (%.2f) %.1f\n',...
    randu, movies{rec_user(1, 2), 2}{1}, w1_M1(rec_user(1, 2),:)*w1_P1(random_user,:)' + mean_rating, probe_vec(probe_vec(:,1) == randu & probe_vec(:,2) == rec_user(1,2), 3),...
           movies{rec_user(1, 3), 2}{1}, w1_M1(rec_user(1, 3),:)*w1_P1(random_user,:)' + mean_rating, probe_vec(probe_vec(:,1) == randu & probe_vec(:,2) == rec_user(1,3), 3),...
           movies{rec_user(1, 4), 2}{1}, w1_M1(rec_user(1, 4),:)*w1_P1(random_user,:)' + mean_rating, probe_vec(probe_vec(:,1) == randu & probe_vec(:,2) == rec_user(1,4), 3));
    



