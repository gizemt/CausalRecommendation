% Gizem Tabak

function divide_dataset()
data = csvread('ratings1M.csv', 1, 0);
users = unique(data(:,1));
movies = unique(data(:,2));
ratings = data(:,3);
data(:,3) = ratings;
train_vec = [];
probe_vec = [];
m_idx = 1;
% for m = movies'
%     data(data(:,2) == m, 2) = m_idx;
%     m_idx = m_idx+1;
% end
%%% USERS
% for u = users'
%     mr_u = data(data(:,1) == u, 2:3); % movie, rating pairs of user u
%     nm_u = size(mr_u,1); % number of movies user u rated
%     idx = floor(nm_u/10);
%     train_vec = [train_vec; [u*ones(nm_u-idx,1) mr_u(1:nm_u-idx, :)]];
%     probe_vec = [probe_vec; [u*ones(idx,1) mr_u(nm_u-idx+1:nm_u, :)]];
% end
%%% RATINGS
% for r = 0.5:0.5:5
%     um_r = data(data(:,3) == r, 1:2); % movie, rating pairs of user u
%     nm_r = size(um_r,1); % number of movies user u rated
%     idx = 1000;%floor(nm_r/10);
%     train_vec = [train_vec; [um_r(1:nm_r-idx, :) r*ones(nm_r-idx,1)]];
%     probe_vec = [probe_vec; [um_r(nm_r-idx+1:nm_r, :) r*ones(idx,1)]];
% end
%%% MOVIES
%%% movies
for m = movies'
    if sum(data(:,2) == m) > 1
    data(data(:,2) == m, 2) = m_idx;
    m_idx = m_idx+1;
    else
    data(data(:,2) == m, :) = [];    
    end
end
users = unique(data(:,1));
movies = unique(data(:,2));
ratings = data(:,3);
for m = movies'
    n_m = sum(data(:,2) == m);
    if n_m < 10
        n_val_m = floor(n_m / 2);
    else
        n_val_m = ceil(n_m / 10);
    end
    idx_m = find(data(:,2) == m);
    [val_idx_m, val_idx_m_idx] = datasample(idx_m, n_val_m, 'Replace', false);
    train_idx_m = ones(n_m,1);
    train_idx_m(val_idx_m_idx) = 0;
    probe_vec = [probe_vec; [data(val_idx_m, 1:3)]];
    train_vec = [train_vec; [data(idx_m(train_idx_m==1), 1:3)]];
end
save('moviedata1M_movies.mat', 'train_vec', 'probe_vec');