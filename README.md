# Causal Inference for Recommender Systems
### Summary
<p align="justify"> In this project, I implemented a causal correction method for recommender systems in order to compensate for the user exposure bias. I combined a Bayesian weighting method [2] with probabilistic matrix factorization for collaborative filtering [1]. I applied the proposed method on MovieLens 100k and 1M datasets to develop a movie recommendation system. The developed system also had 3 modes and can make *popular*, *best match* or *hidden gem* recommendations per user's request. </p>

### Details
<p align="justify"> Collaborative filtering is one of the two fundamental ideas used in recommender systems. In this method, recommendations are made based on user preferences and similarities in their taste, as opposed to item content or user demographics. A collaborative-filtering based recommender system can be thought of as a matrix completion problem, such that each row represents the users and each column represents the items in the dataset, and each element of the matrix represents a user’s reaction for an item, which can be their rating for movie recommendation, or whether or not they click on a link for an advertisement system, and so on. This matrix completion problem is addressed by Salakhutdinov et. al with probabilistic matrix factorization (PMF) [1], which assumes a latent Gaussian model on the user preferences and item properties, and learns those latent model parameters with gradient methods in order to reconstruct the matrix and fill up the unknown elements.</p>

<p align="justify"> One important drawback of probabilistic matrix factorization, also encountered in many other data science applications, is being a data-driven approach, and hence, reflecting the bias of the input data in the resulting recommendations. While it is easy to obtain unbiased data in some applications, it might not naturally be possible in some others. For example, in a movie recommendation problem, it is obvious that ”popular” movies are being rated more than the ”unpopular” ones, and hence they are encountered more frequently in the dataset. As a result, more popular or more exposed items have bigger influence on the learning process than the unpopular or less exposed ones, and hence the resulting recommender system obtains a bias in favor of those.</p>

<p align="justify"> One way of correcting such bias is to resample the input dataset such that the resulting dataset would be as if it is obtained by a randomized experiment. In order to implement that correction, I used Liang et. al. [2]'s Bayesian exposure model, measured the exposure score of each user to an item empirically and weighted the samples based on those exposure scores. </p>

<p align="justify"> The above-mentioned weighting process yields an unbiased recommender. Hence, it is expected the system to not favor popular or more exposed items, but recommend unpopular items just as much for the requesting users. I exploited this advantage to generate 3 different modes so that the system can make popular, best match (most likable) and hidden gem recommendations by weighting the movies by their popularity scores appropriately.</p>

#### References
[1] R. Salakhutdinov and A. Mnih, “Probabilistic matrix factorization,” in Proceedings of the 20th International Conference on Neural Information Processing Systems, USA, 2007, NIPS’07, pp. 1257–1264, Curran Associates Inc.

[2] D. Liang, L. Charlin, and D. M. Blei, “Causal inference for recommendation,” in Workshop on Causation: Foundation to Application, 32nd Conference on Uncertainty in Artificial Intelligence, 2016.

[3] T. Schnabel, A. Swaminathan, A. Singh, N. Chandak, and T. Joachims, “Recommendations as treatments: Debiasing learning and evaluation,” in Proceedings of The 33rd International Conference on Machine Learning, Maria Florina Balcan and Kilian Q. Weinberger, Eds., New York, New York, USA, 20–22 Jun 2016, vol. 48 of Proceedings of Machine Learning Research, pp. 1670– 1679, PMLR.

[4] F. Maxwell Harper and Joseph A. Konstan, “The MovieLens datasets: History and context,” ACM Trans. Interact. Intell. Syst., vol. 5, no. 4, pp. 19:1–19:19, Dec. 2015.
