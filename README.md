# Research Implementation Series

## 📚 About This Repository

This repository is a personal and public log of my journey to master **classical machine learning algorithms** from the ground up, as part of my 6-month research-focused AI roadmap.  
The aim is not only to implement each algorithm from scratch, but also to **understand the theory, reproduce foundational results, run modern experiments, and document my learnings and insights**.  
Each algorithm comes with links to the original research paper and a curated open-source implementation for hands-on learning.

**Motivation:**  
- Bridge the gap between theory and practical application.
- Gain deep intuition for strengths, limitations, and implementation nuances of each method.
- Build a resource for others following a similar path.

---

## 📆 Month-1: Algorithm List

For each algorithm, you’ll find:
- **Foundational Paper**: The original academic source (PDF or stable link)
- **GitHub Implementation**: Clean “from-scratch” code to study and adapt

| #  | Algorithm                 | Foundational Paper                                                                                                                  | GitHub Implementation                                                                                    |
|----|---------------------------|------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 1  | Logistic Regression       | [Cox, 1958](https://www.jstor.org/stable/2983890)                                                                                  | [PierreExeter/logistic-regression-from-scratch](https://github.com/PierreExeter/logistic-regression-from-scratch)       |
| 2  | Ridge Regression          | [Hoerl & Kennard, 1970](https://homepages.math.uic.edu/~lreyzin/papers/ridge.pdf)                                                  | [akaashagarwal/ridge-regression](https://github.com/akaashagarwal/ridge-regression)                     |
| 3  | Lasso                     | [Tibshirani, 1996](https://web.stanford.edu/~hastie/Papers/lars.pdf)                                                               | [mihirchakradeo/LassoRegression](https://github.com/mihirchakradeo/LassoRegression)                     |
| 4  | Elastic Net               | [Zou & Hastie, 2005](https://hastie.su.domains/Papers/ElasticNet.pdf)                                                              | [geekquad/Lasso-Ridge-and-ElasticNet-from-Scratch](https://github.com/geekquad/Lasso-Ridge-and-ElasticNet-from-Scratch) |
| 5  | Perceptron                | [Rosenblatt, 1958](https://homes.cs.washington.edu/~pedrod/560A16/papers/perceptron.pdf)                                           | [valeriopaolicelli/Perceptron-algorithm-from-scratch](https://github.com/valeriopaolicelli/Perceptron-algorithm-from-scratch)  |
| 6  | k-Nearest Neighbours      | [Cover & Hart, 1967](https://isl.stanford.edu/~cover/papers/neighb.pdf)                                                            | [sagarmk/Knn-from-scratch](https://github.com/sagarmk/Knn-from-scratch)                                |
| 7  | Support Vector Machines   | [Cortes & Vapnik, 1995](https://www.csie.ntu.edu.tw/~cjlin/courses/libsvm/cortes_vapnik95.pdf)                                     | [Sohaib1424/SVM-from-scratch](https://github.com/Sohaib1424/SVM-from-scratch)                          |
| 8  | Decision Tree (C4.5/CART) | [Quinlan, 1993](https://cs.toronto.edu/~quinlan/c4.5.pdf)                                                                          | [Valdecy/C4.5](https://github.com/Valdecy/C4.5)                                                        |
| 9  | Random Forests            | [Breiman, 2001](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)                                                       | [SebastianMantey/Random-Forest-from-Scratch](https://github.com/SebastianMantey/Random-Forest-from-Scratch)    |
| 10 | Bagging                   | [Breiman, 1996](https://www.stat.berkeley.edu/~breiman/bagging.pdf)                                                                | [Sarthak-10/Bagging-Classifier-from-scratch](https://github.com/Sarthak-10/Bagging-Classifier-from-scratch)   |
| 11 | Extra Trees               | [Geurts et al., 2006](https://cs.nyu.edu/~bengio/classify/raw_data/extra_trees.pdf)                                                | [jwmng/extratrees](https://github.com/jwmng/extratrees)                                                |
| 12 | AdaBoost                  | [Freund & Schapire, 1997](https://cseweb.ucsd.edu/~yfreund/papers/boosting.pdf)                                                    | [jaimeps/adaboost-implementation](https://github.com/jaimeps/adaboost-implementation)                  |
| 13 | Gradient Boosting (MART)  | [Friedman, 2001](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)                                                                 | [eriklindernoren/ML-from-Scratch (gradient_boosting.py)](https://github.com/eriklindernoren/ML-from-Scratch/blob/master/mlfromscratch/supervised/gradient_boosting.py) |
| 14 | XGBoost                   | [Chen & Guestrin, 2016](https://arxiv.org/pdf/1603.02754.pdf)                                                                      | [Ekeany/XGBoost-From-Scratch](https://github.com/Ekeany/XGBoost-From-Scratch)                          |
| 15 | LightGBM                  | [Ke et al., 2017](https://arxiv.org/pdf/1712.01012.pdf)                                                                            | [microsoft/LightGBM](https://github.com/microsoft/LightGBM)                                             |
| 16 | CatBoost                  | [Prokhorenkova et al., 2018](https://arxiv.org/pdf/1706.09516.pdf)                                                                 | [catboost/catboost](https://github.com/catboost/catboost)                                               |
| 17 | Stacked Generalization    | [Wolpert, 1992](https://citeseerx.ist.psu.edu/doc_view/10.1.1.32.1900)                                                             | [dustinstansbury/stacked_generalization](https://github.com/dustinstansbury/stacked_generalization)     |
| 18 | Naive Bayes               | [McCallum & Nigam, 1998](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)                                            | [gbroques/naive-bayes](https://github.com/gbroques/naive-bayes)                                        |
| 19 | k-Means                   | [MacQueen, 1967](https://people.eecs.berkeley.edu/~jordan/kmeans.pdf)                                                              | [tugot17/K-Means-Algorithm-From-Scratch](https://github.com/tugot17/K-Means-Algorithm-From-Scratch)    |
| 20 | Hierarchical Clustering   | [Ward, 1963](https://web.as.uky.edu/statistics/users/pbreheny/701/F17/notes/10-05.pdf)                                             | [hhundiwala/hierarchical-clustering](https://github.com/hhundiwala/hierarchical-clustering)             |
| 21 | DBSCAN                    | [Ester et al., 1996](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)                                                           | [aravindsairam/DBSCAN_from_-scratch](https://github.com/aravindsairam/DBSCAN_from_-scratch)             |

---

## 📝 What’s Inside Each Directory

- **/algorithm_name/**  
  - Paper summary & insights  
  - Implementation (Jupyter notebooks, scripts)  
  - Experiments (reproducibility, modern tweaks, extra tests)
  - Plots, logs, and personal reflections

---

