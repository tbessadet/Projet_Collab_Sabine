import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class AdaBoost_maison:
    """AdaBoost multiclasse (SAMME), labels dans {0, ..., K-1}, arbres de decision faibles."""

    def __init__(self, n_estimators=50, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []   # weak learners h_t
        self.alphas = []  # poids alpha_t de chaque learner
        self.train_errors = []
        self.classes_ = None

    def fit(self, X, y):
        n = X.shape[0]
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        w = np.full(n, 1 / n)

        for t in range(self.n_estimators):
            # max_depth=1 est trop faible pour K=10 : l'erreur depasse 1 - 1/K, prendre 2-4
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=w)
            pred = tree.predict(X)

            err = np.sum(w[pred != y]) / np.sum(w)

            # learner pire que le hasard (1 - 1/K au lieu de 1/2 en binaire) -> on arrete
            if err >= 1 - 1 / K:
                break

            # SAMME : le terme log(K-1) garantit alpha > 0 tant que err < 1 - 1/K
            alpha = np.log((1 - err) / (err + 1e-10)) + np.log(K - 1)

            # on augmente seulement le poids des exemples mal classes
            w *= np.exp(alpha * (pred != y))
            w /= np.sum(w)

            self.trees.append(tree)
            self.alphas.append(alpha)
            self.train_errors.append(err)
        return self

    def decision_function(self, X):
        """Matrice (n, K) : score[i, k] = sum_t alpha_t * 1[h_t(x_i) == k]."""
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for tree, alpha in zip(self.trees, self.alphas):
            scores += alpha * (tree.predict(X)[:, None] == self.classes_[None, :])
        return scores

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def staged_score(self, X, y):
        """Accuracy apres 1, 2, ..., T learners (pour tracer la courbe)."""
        # TODO: cumuler les scores (n, K) learner par learner, argmax, accuracy a chaque etape
        pass


def main():
    X, y = load_digits(return_X_y=True)  # 10 classes : 0..9
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    model = AdaBoost_maison(n_estimators=50, max_depth=3).fit(X_train, y_train)
    print("Accuracy test :", accuracy_score(y_test, model.predict(X_test)))

    # TODO: comparer avec un seul arbre, et avec sklearn.ensemble.AdaBoostClassifier
    # TODO: tracer train/test accuracy en fonction du nombre d'estimateurs


if __name__ == "__main__":
    main()
