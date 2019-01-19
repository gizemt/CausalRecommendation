load moviedata100k_movies
num_m = length(unique([train_vec(:,2); probe_vec(:,2)]));  % Number of movies 
num_p = length(unique([train_vec(:,1); probe_vec(:,1)]));  % Number of users 
% figure,histogram([train_vec(:,1); probe_vec(:,1)], num_p, 'EdgeColor', 'none', 'FaceAlpha', 0.9);axis tight;title('Number of movies a user rated');xlabel('UserID');
% figure,histogram([train_vec(:,2); probe_vec(:,2)], num_m, 'EdgeColor', 'none', 'FaceAlpha', 0.9);axis tight;title('Number of users rated a movie');xlabel('MovieID');

figure,histogram(train_vec(:,2), num_m, 'EdgeColor', 'none', 'FaceAlpha', 0.9);hold on;histogram(probe_vec(:,2), num_m, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
title('Number of ratings for a movie');xlabel('Movie ID');legend('Training', 'Validation');axis tight;
figure,histogram(train_vec(:,1), num_p, 'EdgeColor', 'none', 'FaceAlpha', 0.9);hold on;histogram(probe_vec(:,1), num_p, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
title('Number of movies a user rated');xlabel('User ID');legend('Training', 'Validation');axis tight;
% n_movies = hist([train_vec(:,2); probe_vec(:,2)], num_m);
% sum(n_movies <2)
% n_users = hist([train_vec(:,1); probe_vec(:,1)], num_p);
% sum(n_users <2)

% figure,histogram([train_vec(:,3); probe_vec(:,3)], 0.5:1:5.5, 'FaceAlpha', 0.9);axis tight;ylabel('Number of movies');xlabel('Rating');
% figure,histogram(([train_vec(:,3); probe_vec(:,3)] > 3)*1, -0.25:0.5:1.75, 'FaceAlpha', 0.9);axis tight;ylabel('Number of movies');xlabel('Rating');
