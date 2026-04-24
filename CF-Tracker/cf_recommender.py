"""
cf_recommender.py
=================
Recommendation system for Codeforces problems, implementing the three
algorithms from the Mangaki paper (Vie et al., 2017):
 
  1. CFKnn   — collaborative filtering via cosine similarity (KNN)
  2. CFSGD   — matrix factorisation via stochastic gradient descent
  3. CFLasso — tag-based linear regression (LASSO)
 
Input data shape
----------------
A DataFrame (or list of dicts) with columns:
    USER         — Codeforces handle (str)
    USER_ELO     — user's CF rating (int)
    PROBLEM      — problem identifier, e.g. "1234A" (str)
    PROBLEM_ELO  — problem rating (int)
    PROBLEM_TAGS — list of tag strings, e.g. ["dp", "greedy"]
    VERDICT      — "OK" for a solve, anything else for a failure
 
Missing-data strategies  (controlled by CFRecommender.missing_strategy)
------------------------------------------------------------------------
"zero"      — [default, original behaviour] missing (user, problem) pairs
              are treated as 0 in the rating matrix. Dense and simple but
              conflates "never attempted" with "failed".
 
"ignore"    — missing pairs are simply excluded from training. The sparse
              matrix only contains observed verdicts. Removes the zero-bias
              but reduces the amount of signal for KNN similarity.
 
"elo_fill"  — missing pairs are imputed with the Elo-based win probability:
                  P(win) = 1 / (1 + 10^((problem_elo - user_elo) / 400))
              Problems without a rating get P=0.5. This gives every
              (user, problem) pair a meaningful prior.
 
LASSO tag printing  (controlled by CFRecommender.lasso_print_tags)
-------------------------------------------------------------------
False  — [default] silent.
True   — after fitting LASSO, print the non-zero tag coefficients for
         every user, sorted by descending absolute value.
 
All three options are independent and fully backward-compatible:
the defaults reproduce the original behaviour exactly.

"""
 
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.special import expit          # sigmoid: maps any real → (0, 1)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
 
# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_tags(raw) -> list:
    """
    Parse PROBLEM_TAGS in any of the supported formats:
      - Already a Python list:         ['dp', 'greedy']
      - Semicolon-separated string:    'dp;greedy;math'   ← CSV format
      - JSON-encoded list string:      '["dp","greedy"]'
      - Comma-separated string:        'dp,greedy'        ← fallback
    """
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    # Semicolon-separated (primary CSV format)
    if '|' in raw:
        return [t.strip() for t in raw.split('|') if t.strip()]

    if ';' in raw:
        return [t.strip() for t in raw.split(';') if t.strip()]
    # JSON list
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    if not isinstance(raw, str) or not raw.strip():
        return []
    # Comma-separated fallback
    if ',' in raw:
        return [t.strip() for t in raw.split(',') if t.strip()]
    # Single tag
    return [raw.strip()]


def _encode_ids(series: pd.Series):
    """
    Map a string Series to contiguous integer IDs.
    Returns (encoded_array, id_to_label_dict, label_to_id_dict).
    """
    labels = series.unique()
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}
    return series.map(label_to_id).values, id_to_label, label_to_id
 
 
def _elo_win_probability(user_elo: float, problem_elo: float) -> float:
    """
    Standard Elo expected score formula.
    Returns the probability that a player rated user_elo
    solves a problem rated problem_elo.
    """
    return 1.0 / (1.0 + 10.0 ** ((problem_elo - user_elo) / 400.0))

def jaccard_similarity_csr(X: csr_matrix) -> csr_matrix:
    # Ensure CSR
    X = X.tocsr()

    # Force binary (in case values are not exactly 0/1)
    X_bin = X.copy()
    X_bin.data = np.ones_like(X_bin.data)

    # Intersection: |A ∩ B|
    intersection = (X_bin @ X_bin.T).tocsr()

    # Row sums: |A|
    row_sums = np.array(X_bin.sum(axis=1)).ravel()

    # Compute Jaccard only on nonzero intersections
    rows, cols = intersection.nonzero()
    inter_data = intersection.data

    union_data = row_sums[rows] + row_sums[cols] - inter_data

    # Avoid division by zero
    union_data[union_data == 0] = 1

    jaccard_data = inter_data / union_data

    # Build sparse result
    J = csr_matrix((jaccard_data, (rows, cols)), shape=X.shape)

    return J

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 1 — KNN (k-nearest neighbours
# ─────────────────────────────────────────────────────────────────────────────
 
class CFKnn:
    """
    Collaborative filtering via cosine user similarity.
 
    For a given (user, problem) pair, finds the k users most similar to
    the target user who have attempted that problem, and averages their
    verdicts (observed or imputed) as the predicted solve probability.
    """
 
    def __init__(self, nb_neighbors: int = 20):
        self.nb_neighbors = nb_neighbors
 
    def fit(self, user_ids, problem_ids, verdicts, nb_users, nb_problems):
        """
        Build the user-problem matrix and precompute user similarities.
 
        Parameters
        ----------
        user_ids, problem_ids : integer-encoded arrays (0-based)
        verdicts              : float array — observed or imputed values
        nb_users, nb_problems : total number of unique users / problems
        """
        self.nb_users    = nb_users
        self.nb_problems = nb_problems
 
        self.ratings = coo_matrix(
            (verdicts, (user_ids, problem_ids)),
            shape=(nb_users, nb_problems),
        )
        self.ratings_by_user    = self.ratings.tocsr()
        self.ratings_by_problem = self.ratings.tocsc()
        self.user_similarity    = cosine_similarity(self.ratings_by_user)  # cosine_similarity(self.ratings_by_user)   # #instead of cosine_similarity 
 
    def predict_one(self, user_id: int, problem_id: int) -> float:
        """
        Predict solve probability for a single (user_id, problem_id) pair.
        Falls back to 0.5 if no neighbours have a value for this problem.
        """
        attempted_col = self.ratings_by_problem[:, problem_id]
        raters = list(attempted_col.indices)
 
        if not raters:
            return 0.5
 
        raters.sort(
            key=lambda r: self.user_similarity[user_id, r],
            reverse=True,
        )
        neighbours        = raters[:self.nb_neighbors]
        neighbour_ratings = self.ratings_by_problem[neighbours, problem_id]
        return float(neighbour_ratings.mean()) if neighbour_ratings.nnz else 0.5
 
    def predict(self, user_ids, problem_ids) -> np.ndarray:
        return np.array([
            self.predict_one(u, p)
            for u, p in zip(user_ids, problem_ids)
        ])
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 2 — SGD matrix factorisation (§"Une technique plus performante")
# ─────────────────────────────────────────────────────────────────────────────
 
class CFSGD:
    """
    Matrix factorisation via stochastic gradient descent with L2 regularisation.
 
    Learns latent vectors U (users) and V (problems) such that
    sigmoid(U[i] · V[j]) ≈ probability that user i solves problem j.

    """
 
    def __init__(
        self,
        nb_components: int = 20,
        nb_iterations: int = 15,
        gamma: float = 0.01,
        lambda_: float = 0.1,
    ):
        self.nb_components = nb_components
        self.nb_iterations = nb_iterations
        self.gamma         = gamma
        self.lambda_       = lambda_
 
    def fit(self, user_ids, problem_ids, verdicts, nb_users, nb_problems):
        self.nb_users    = nb_users
        self.nb_problems = nb_problems
 
        rng    = np.random.default_rng(seed=42)
        self.U = rng.standard_normal((nb_users,    self.nb_components)) * 0.01
        self.V = rng.standard_normal((nb_problems, self.nb_components)) * 0.01
 
        n = len(user_ids)
 
        for epoch in range(self.nb_iterations):
            indices = rng.permutation(n)
 
            for idx in indices:
                i     = user_ids[idx]
                j     = problem_ids[idx]
                rij   = verdicts[idx]
                p_hat = expit(self.U[i] @ self.V[j])
                error = p_hat - rij
 
                self.U[i] -= self.gamma * (error * self.V[j] + self.lambda_ * self.U[i])
                self.V[j] -= self.gamma * (error * self.U[i] + self.lambda_ * self.V[j])
 
            if (epoch + 1) % 5 == 0:
                preds = self.predict(user_ids, problem_ids)
                eps   = 1e-7
                bce   = -np.mean(
                    verdicts * np.log(preds + eps)
                    + (1 - verdicts) * np.log(1 - preds + eps)
                )
                print(f"  [SGD] epoch {epoch + 1}/{self.nb_iterations}  BCE loss: {bce:.4f}")
 
    def predict_one(self, user_id: int, problem_id: int) -> float:
        return float(expit(self.U[user_id] @ self.V[problem_id]))

    def predict(self, user_ids, problem_ids) -> np.ndarray:
        dots = np.einsum('ij,ij->i', self.U[user_ids], self.V[problem_ids])
        return expit(dots)
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 3 — LASSO tag regression (§"Utiliser les posters")
# ─────────────────────────────────────────────────────────────────────────────
 
class CFLasso:
    """
    Tag-based sparse linear regression (LASSO), adapted from MangakiLASSO.
 
    Uses problem tag vectors as feature matrix T (instead of poster embeddings).
    For each user, a LASSO model learns a sparse weight vector P_i over tags.
 
    Predicted solve probability = clip(P_i · T_j, 0, 1).
 
    Setting print_tags=True will print each user's non-zero tag coefficients
    after training, sorted by descending absolute value.
    """
 
    def __init__(self, alpha: float = 0.01, print_tags: bool = False):
        self.alpha      = alpha
        self.print_tags = print_tags
 
    def fit(self, user_ids, problem_ids, verdicts, nb_users,
            problem_tag_matrix, tag_names=None, id_to_user=None):
        """
        Train one LASSO model per user.
 
        Parameters
        ----------
        user_ids, problem_ids  : integer-encoded arrays
        verdicts               : float array
        nb_users               : total number of users
        problem_tag_matrix     : np.ndarray (nb_problems × nb_tags)
        tag_names              : list of tag name strings (for printing)
        id_to_user             : dict mapping user_id → handle (for printing)
        """
        from collections import defaultdict
 
        self.nb_users           = nb_users
        self.problem_tag_matrix = problem_tag_matrix
        self.tag_names          = tag_names or []
        self.id_to_user         = id_to_user or {}
        self.lasso              = {}
 
        user_problems = defaultdict(list)
        user_verdicts = defaultdict(list)
        for u, p, v in zip(user_ids, problem_ids, verdicts):
            user_problems[u].append(p)
            user_verdicts[u].append(v)
 
        for user_id in range(nb_users):
            pids = user_problems[user_id]
            verd = user_verdicts[user_id]
            if not pids:
                continue
            X_user = problem_tag_matrix[pids]
            y_user = np.array(verd)
            model  = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=2000)
            model.fit(X_user, y_user)
            self.lasso[user_id] = model
 
        if self.print_tags:
            self._print_all_tag_weights()
 
    def _print_all_tag_weights(self):
        """
        Print non-zero tag coefficients for every trained user,
        sorted by descending absolute value.
        Called automatically after fit() when print_tags=True.
        """
        print("\n" + "=" * 60)
        print("LASSO tag coefficients per user")
        print("=" * 60)
        for user_id, model in self.lasso.items():
            handle = self.id_to_user.get(user_id, f"user_{user_id}")
            coef   = model.coef_
            nonzero = [
                (self.tag_names[k] if k < len(self.tag_names) else f"tag_{k}", float(w))
                for k, w in enumerate(coef) if w != 0.0
            ]
            if not nonzero:
                continue
            nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n  {handle}:")
            for tag, w in nonzero:
                sign = "+" if w > 0 else ""
                print(f"    {tag:<30} {sign}{w:.4f}")
        print("=" * 60 + "\n")
 
    def predict_one(self, user_id: int, problem_id: int) -> float:
        if user_id not in self.lasso:
            return 0.5
        tag_vec = self.problem_tag_matrix[problem_id].reshape(1, -1)
        raw     = float(self.lasso[user_id].predict(tag_vec)[0])
        return float(np.clip(raw, 0.0, 1.0))
 
    def predict(self, user_ids, problem_ids) -> np.ndarray:
        return np.array([
            self.predict_one(u, p)
            for u, p in zip(user_ids, problem_ids)
        ])
 
    def explain(self, user_id: int, tag_names: list = None) -> dict:
        """
    Return the non-zero feature weights for a user, split into:
    - tag weights  (key = tag name)
    - elo weight   (key = '__elo__')

    Positive tag weight  = tends to solve problems with that tag.
    Negative tag weight  = tends to fail problems with that tag.
    Positive elo weight  = tends to solve harder problems (benefits from difficulty).
    Negative elo weight  = tends to solve easier problems (penalised by difficulty).
    """ 
        names = tag_names or self.tag_names
        if user_id not in self.lasso:
            return {}
        coef = self.lasso[user_id].coef_
        return {
        name: round(float(w), 4)
        for name, w in zip(names, coef)
        if w != 0.0
    }
 
# ─────────────────────────────────────────────────────────────────────────────
#  Ensemble — blend models (§"Combiner des modèles")
# ─────────────────────────────────────────────────────────────────────────────
 
class CFEnsemble:
    """
    Weighted average of KNN, SGD, and LASSO predictions (blending).
    
    """
 
    def __init__(self, knn: CFKnn, sgd: CFSGD, lasso: CFLasso,
                 weights=(0.2, 0.5, 0.3)):
        self.models  = [knn, sgd, lasso]
        self.weights = np.array(weights) / sum(weights)
 
    def predict_one(self, user_id: int, problem_id: int) -> float:
        preds = np.array([m.predict_one(user_id, problem_id) for m in self.models])
        return float(preds @ self.weights)
 
    def predict(self, user_ids, problem_ids) -> np.ndarray:
        stacked = np.stack([m.predict(user_ids, problem_ids) for m in self.models])
        return stacked.T @ self.weights
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  High-level facade
# ─────────────────────────────────────────────────────────────────────────────
 
class CFRecommender:
    """
    High-level recommender that preprocesses data, trains all models,
    and exposes rank_candidates() / recommend() for any user.
 
    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: USER, USER_ELO, PROBLEM, PROBLEM_ELO,
        PROBLEM_TAGS, VERDICT.
 
    target_prob_lo, target_prob_hi : float
        Desired solve-probability window (default 0.50–0.65).
 
    missing_strategy : str  — how to handle unobserved (user, problem) pairs
        "zero"      Original behaviour. Missing pairs contribute 0 to the
                    rating matrix (implicit negative signal).
        "ignore"    Missing pairs are excluded entirely. Only observed
                    verdicts are used for training.
        "elo_fill"  Missing pairs are imputed with the Elo-based win
                    probability P = 1/(1+10^((prob_elo-user_elo)/400)).
                    Problems without a rating use P=0.5.
 
    lasso_print_tags : bool
        If True, print each user's LASSO tag coefficients to stdout
        after fitting. Default False.
    """
 
    MISSING_STRATEGIES = ("zero", "ignore", "elo_fill")
 
    def __init__(
        self,
        df: pd.DataFrame,
        target_prob_lo: float = 0.50,
        target_prob_hi: float = 0.65,
        missing_strategy: str = "zero",
        lasso_print_tags: bool = True,
    ):
        if missing_strategy not in self.MISSING_STRATEGIES:
            raise ValueError(
                f"missing_strategy must be one of {self.MISSING_STRATEGIES}, "
                f"got '{missing_strategy}'"
            )
        self.target_prob_lo   = target_prob_lo
        self.target_prob_hi   = target_prob_hi
        self.missing_strategy = missing_strategy
        self.lasso_print_tags = lasso_print_tags
        self._preprocess(df)
 
    # ── Preprocessing ────────────────────────────────────────────────────────
 
    def _preprocess(self, df: pd.DataFrame):
        df = df.copy()
        print(df['PROBLEM_TAGS'])
        df['PROBLEM_TAGS'] = df['PROBLEM_TAGS'].apply(_parse_tags)
 
        # Binary observed verdict
        df['verdict_binary'] = (df['VERDICT'] == 'OK').astype(float).replace(0,0.01) #Differentiates Wrong Asnwers from a lack of submissions
       


        # Integer-encode users and problems
        df['user_id'],    self.id_to_user,    self.user_to_id    = _encode_ids(df['USER'])
        df['problem_id'], self.id_to_problem, self.problem_to_id = _encode_ids(df['PROBLEM'])
 
        self.nb_users    = len(self.id_to_user)
        self.nb_problems = len(self.id_to_problem)
 
        # Per-problem Elo lookup (for elo_fill strategy)
        self._problem_elo = (
            df.drop_duplicates('problem_id')
              .set_index('problem_id')['PROBLEM_ELO']
              .to_dict()
        )
 
        # Per-user Elo lookup (for elo_fill strategy)
        self._user_elo = (
            df.drop_duplicates('user_id')
              .set_index('user_id')['USER_ELO']
              .to_dict()
        )
 
        # Tag feature matrix T: shape (nb_problems, nb_tags)
        problem_tags = (
            df.drop_duplicates('problem_id')
              .sort_values('problem_id')[['problem_id', 'PROBLEM_TAGS']]
        )
        
        mlb = MultiLabelBinarizer()
        tag_matrix              = mlb.fit_transform(problem_tags['PROBLEM_TAGS'])
        self.tag_names          = list(mlb.classes_)
       
        self.problem_tag_matrix = tag_matrix.astype(float)
 
        self.df = df

        problem_meta_sorted = (
            df.drop_duplicates('problem_id')
              .sort_values('problem_id')
        )
        
        # ── Binary tag matrix ─────────────────────────────────────────────────
        mlb = MultiLabelBinarizer()
        self.tag_names  = list(mlb.fit(problem_meta_sorted['PROBLEM_TAGS']).classes_)
        raw_tag_matrix  = mlb.transform(problem_meta_sorted['PROBLEM_TAGS']).astype(float)
 
        # L2-normalise each row so multi-tag problems don't dominate the
        # LASSO dot product over single-tag problems.
        norms = np.linalg.norm(raw_tag_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised_tags = raw_tag_matrix / norms
 
        # ── Elo feature column ────────────────────────────────────────────────

        elos = problem_meta_sorted['PROBLEM_ELO'].fillna(0).values.astype(float)
        rated_mask = elos > 0
        if rated_mask.sum() > 1:
            elo_min =  elos[rated_mask].min() 
            elo_max =  elos[rated_mask].max()
            elo_range = elo_max - elo_min if elo_max > elo_min else 1.0
            elo_norm = np.where(rated_mask, (elos - elo_min) / elo_range, 0.5)
        else:
            elo_norm = np.full(len(elos), 0.5)
 
        # Append as the last column — tag_names gets a sentinel entry so
        # explain() can label the Elo coefficient correctly.
        self.tag_names.append('__elo__')
        self.problem_tag_matrix = np.column_stack([normalised_tags, elo_norm])
 
        # Store normalisation params so predict_one can apply the same
        # transformation to unseen problems at inference time.
        self._elo_min   = elos[rated_mask].min() if rated_mask.sum() > 0 else 0.0
        self._elo_max   = elos[rated_mask].max() if rated_mask.sum() > 0 else 3500.0
        self._elo_range = self._elo_max - self._elo_min if self._elo_max > self._elo_min else 1.0
 



        # Problem metadata for display
        self.problem_meta = (
            df.drop_duplicates('PROBLEM')
              .set_index('problem_id')[['PROBLEM', 'PROBLEM_ELO', 'PROBLEM_TAGS']]
              .to_dict('index')
        )
 
    def _build_training_arrays(self):
        """
        Build the (user_ids, problem_ids, verdicts) training arrays
        according to the chosen missing_strategy.
 
        "zero"     — return observed rows only; the sparse matrix fills
                     unobserved cells with 0 implicitly.
        "ignore"   — identical to "zero" at the array level; the distinction
                     is that we never add synthetic rows for unseen pairs.
        "elo_fill" — add one row for every (user, problem) pair that has
                     NOT been observed, using the Elo-based prior as the
                     imputed verdict.
        """
        observed_u = self.df['user_id'].values
        observed_p = self.df['problem_id'].values
        observed_v = self.df['verdict_binary'].values
 
        if self.missing_strategy in ("zero", "ignore"):
            return observed_u, observed_p, observed_v
 
        # elo_fill: impute all unobserved (user, problem) pairs
        observed_set = set(zip(observed_u.tolist(), observed_p.tolist()))
 
        synthetic_u, synthetic_p, synthetic_v = [], [], []
 
        for uid in range(self.nb_users):
            user_elo = self._user_elo.get(uid, 1200)
            for pid in range(self.nb_problems):
                if (uid, pid) in observed_set:
                    continue
                prob_elo = self._problem_elo.get(pid)
                if prob_elo is not None and prob_elo > 0:
                    imputed = _elo_win_probability(user_elo, prob_elo)
                else:
                    imputed = 0.5   # no rating available — neutral prior
                synthetic_u.append(uid)
                synthetic_p.append(pid)
                synthetic_v.append(imputed)
 
        if not synthetic_u:
            return observed_u, observed_p, observed_v
 
        all_u = np.concatenate([observed_u, np.array(synthetic_u)])
        all_p = np.concatenate([observed_p, np.array(synthetic_p)])
        all_v = np.concatenate([observed_v, np.array(synthetic_v)])
        return all_u, all_p, all_v
 
    # ── Training ─────────────────────────────────────────────────────────────
 
    def fit(
        self, 
        nb_neighbors: int = 20, 
        nb_components: int = 20,
        nb_iterations: int = 15,
        train_knn: bool = True,
        train_sgd: bool = True,
        train_lasso: bool = True,
        ):
        """Train all three models (and the ensemble) on the dataset."""
        print(f"Missing strategy: '{self.missing_strategy}'")
        u, p, v = self._build_training_arrays()
        print(f"Training on {len(u):,} (user, problem) pairs ")
              
        if(train_knn):
            print("Training KNN...")
            self.knn = CFKnn(nb_neighbors=nb_neighbors)
            self.knn.fit(u, p, v, self.nb_users, self.nb_problems)
        if(train_sgd):
            print("Training SGD...")
            self.sgd = CFSGD(nb_components=nb_components, nb_iterations=nb_iterations)
            self.sgd.fit(u, p, v, self.nb_users, self.nb_problems)
        if(train_lasso):
            print("Training LASSO...")
            self.lasso = CFLasso(print_tags=self.lasso_print_tags)
            self.lasso.fit(
                u, p, v, self.nb_users,
                self.problem_tag_matrix,
                tag_names=self.tag_names,
                id_to_user=self.id_to_user,
            )
        if train_knn and train_sgd and train_lasso:
            self.ensemble = CFEnsemble(self.knn, self.sgd, self.lasso)
        
        trained = [m for m, flag in [("KNN", train_knn), ("SGD", train_sgd), ("LASSO", train_lasso)] if flag]
        print(f"Trained: {', '.join(trained) if trained else 'none'}.")


    # ── Inference ────────────────────────────────────────────────────────────
 
    def _get_model(self, model: str):
        models = {
            'knn':      self.knn,
            'sgd':      self.sgd,
            'lasso':    self.lasso,
            'ensemble': self.ensemble,
        }
        if model not in models:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(models)}")
        return models[model]
 
    def rank_candidates(
        self,
        handle: str,
        model: str = 'ensemble',
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Return the top_k unseen problems whose predicted solve probability
        is closest to the centre of [target_prob_lo, target_prob_hi].
        Already-attempted problems are always excluded.
        """
        if handle not in self.user_to_id:
            raise ValueError(f"User '{handle}' not found in the dataset.")
 
        user_id = self.user_to_id[handle]
        m       = self._get_model(model)
 
        attempted = set(
            self.df[self.df['user_id'] == user_id]['problem_id'].unique()
        )
 
        candidate_ids = [
            pid for pid in range(self.nb_problems)
            if pid not in attempted
        ]
 
        if not candidate_ids:
            return pd.DataFrame()
 
        uid_arr = np.full(len(candidate_ids), user_id)
        pid_arr = np.array(candidate_ids)
        probs   = m.predict(uid_arr, pid_arr)
 
        target_centre = (self.target_prob_lo + self.target_prob_hi) / 2
        distances     = np.abs(probs - target_centre)
        order         = np.argsort(distances)[:top_k]
 
        rows = []
        for i in order:
            pid  = candidate_ids[i]
            meta = self.problem_meta.get(pid, {})
            rows.append({
                'problem':     meta.get('PROBLEM', str(pid)),
                'problem_elo': meta.get('PROBLEM_ELO'),
                'tags':        meta.get('PROBLEM_TAGS', []),
                'solve_prob':  round(float(probs[i]), 3),
                'in_target':   self.target_prob_lo <= probs[i] <= self.target_prob_hi,
            })
 
        return pd.DataFrame(rows)
 
    def recommend(self, handle: str, model: str = 'ensemble') -> dict | None:
        """
        Return the single best problem for the user.
        Returns None if no candidates are available.
        """
        ranked = self.rank_candidates(handle, model=model, top_k=1)
        if ranked.empty:
            return None
        return ranked.iloc[0].to_dict()
 
    def explain(self, handle: str) -> dict:
        """
        Return the LASSO tag preference weights for a user.
        Positive = strength, negative = weakness.
        """
        if handle not in self.user_to_id:
            raise ValueError(f"User '{handle}' not found.")
        return self.lasso.explain(self.user_to_id[handle], self.tag_names)
 
# ─────────────────────────────────────────────────────────────────────────────
#  Statistical analysis functions
# ─────────────────────────────────────────────────────────────────────────────
def lasso_evaluation(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    alpha: float = 0.001,
    min_train_samples: int = 5,
    n_users: int = None,
    seed: int = 42,
) -> dict:
    """
    Evaluate CFLasso accuracy using per-user train/test splitting.
 
    Strategy
    --------
    For each user who has enough data, we:
      1. Hold out a random fraction of their submissions as the test set.
      2. Train a LASSO model on the remaining (train) submissions.
      3. Predict solve probability on the test set.
      4. Compute accuracy metrics by comparing predictions to true verdicts.
 
    We do this per-user (not globally) because LASSO fits one model per user —
    a global split would leak test users into other users' training data anyway.
 
    Metrics reported
    ----------------
    Per-user:
      - MAE      : mean absolute error  |predicted_prob - true_verdict|
      - RMSE     : root mean squared error
      - AUC-ROC  : ranking quality — how well the model separates solves from failures
                   (only computed when the test set contains both classes)
      - accuracy : fraction of correct binary predictions  (threshold = 0.5)
      - calibration_bias : mean(predicted) - mean(true)  — positive = over-confident
 
    Aggregate (over all evaluated users):
      - mean / std / median of each per-user metric
      - fraction of users where LASSO beats the naive baseline (always predict 0.5)
      - users_evaluated  : how many users had enough data to evaluate
      - users_skipped    : how many were skipped (too few samples or no tag variance)
 
    Parameters
    ----------
    df               : full submission DataFrame
    test_fraction    : fraction of each user's submissions held out for testing
    alpha            : LASSO regularisation strength
    min_train_samples: minimum training samples required to fit a user's model
    n_users          : if set, only evaluate the first n_users users
    seed             : random seed for the train/test split
 
    Returns
    -------
    dict with keys: per_user (list of dicts), aggregate (dict), tag_names (list)
    """
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
 
    rng = np.random.default_rng(seed)
 
    # ── Preprocessing (mirrors CFRecommender._preprocess) ────────────────────
    df = df.copy()
    df['PROBLEM_TAGS']   = df['PROBLEM_TAGS'].apply(_parse_tags)
    df['verdict_binary'] = (df['VERDICT'] == 'OK').astype(float)
 
    df['user_id'],    id_to_user,    user_to_id    = _encode_ids(df['USER'])
    df['problem_id'], id_to_problem, problem_to_id = _encode_ids(df['PROBLEM'])
 
    nb_users    = len(id_to_user)
    nb_problems = len(id_to_problem)
 
    # Tag + Elo feature matrix (same pipeline as CFRecommender._preprocess)
    problem_meta = (
        df.drop_duplicates('problem_id').sort_values('problem_id')
    )
    mlb       = MultiLabelBinarizer()
    tag_names = list(mlb.fit(problem_meta['PROBLEM_TAGS']).classes_)
    raw_tags  = mlb.transform(problem_meta['PROBLEM_TAGS']).astype(float)
 
    norms = np.linalg.norm(raw_tags, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_tags = raw_tags / norms
 
    elos      = problem_meta['PROBLEM_ELO'].fillna(0).values.astype(float)
    rated     = elos > 0
    elo_min   = elos[rated].min() if rated.sum() > 0 else 0.0
    elo_max   = elos[rated].max() if rated.sum() > 0 else 3500.0
    elo_range = elo_max - elo_min if elo_max > elo_min else 1.0
    elo_norm  = np.where(rated, (elos - elo_min) / elo_range, 0.5)
 
    tag_names.append('__elo__')
    T = np.column_stack([norm_tags, elo_norm])   # (nb_problems, nb_tags+1)
 
    # ── Per-user evaluation ───────────────────────────────────────────────────
    # Build user_id -> list of (problem_id, verdict) from observed data
    user_data = defaultdict(list)
    for _, row in df.iterrows():
        user_data[int(row['user_id'])].append(
            (int(row['problem_id']), float(row['verdict_binary']))
        )
 
    user_ids_to_eval = list(range(nb_users))
    if n_users is not None:
        user_ids_to_eval = user_ids_to_eval[:n_users]
 
    per_user      = []
    skipped       = 0
    baseline_mae  = []   # naive always-0.5 baseline per user
 
    for uid in user_ids_to_eval:
        observations = user_data.get(uid, [])
        if len(observations) < min_train_samples + 1:
            skipped += 1
            continue
 
        # Deduplicate on problem_id (keep last verdict)
        seen = {}
        for pid, v in observations:
            seen[pid] = v
        obs = list(seen.items())   # [(problem_id, verdict), ...]
 
        # Shuffle and split
        indices = list(range(len(obs)))
        rng.shuffle(indices)
        n_test  = max(1, int(len(obs) * test_fraction))
        n_train = len(obs) - n_test
 
        if n_train < min_train_samples:
            skipped += 1
            continue
 
        train_idx = indices[:n_train]
        test_idx  = indices[n_train:]
 
        train_pids = [obs[i][0] for i in train_idx]
        train_verd = np.array([obs[i][1] for i in train_idx])
        test_pids  = [obs[i][0] for i in test_idx]
        test_verd  = np.array([obs[i][1] for i in test_idx])
 
        X_train = T[train_pids]
        X_test  = T[test_pids]
 
        # Skip users with no tag variance in training data
        if X_train.std() == 0:
            skipped += 1
            continue
 
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
        model.fit(X_train, train_verd)
 
        raw_preds  = model.predict(X_test)
        preds      = np.clip(raw_preds, 0.0, 1.0)
        errors     = preds - test_verd
 
        mae  = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(preds) - np.mean(test_verd))
 
        # Binary accuracy at threshold 0.5
        pred_binary = (preds >= 0.5).astype(int)
        accuracy    = float(np.mean(pred_binary == test_verd.astype(int)))
 
        # Naive baseline: always predict 0.5
        baseline_err = np.abs(0.5 - test_verd)
        b_mae        = float(np.mean(baseline_err))
        baseline_mae.append(b_mae)
 
        # AUC-ROC only when both classes present in test set.
        # test_verd must be cast to int — sklearn rejects float 0.0/1.0
        # as "continuous format" in some versions.
        test_verd_int = test_verd.astype(int)
        unique_labels = np.unique(test_verd_int)
        if len(unique_labels) == 2:
            auc = float(roc_auc_score(test_verd_int, preds))
        else:
            auc = None
 
        handle = id_to_user.get(uid, f"user_{uid}")
        n_nonzero_coef = int(np.sum(model.coef_ != 0))
 
        per_user.append({
            'handle':        handle,
            'n_train':       n_train,
            'n_test':        n_test,
            'mae':           round(mae,  4),
            'rmse':          round(rmse, 4),
            'accuracy':      round(accuracy, 4),
            'auc_roc':       round(auc, 4) if auc is not None else None,
            'calib_bias':    round(bias, 4),
            'baseline_mae':  round(b_mae, 4),
            'beats_baseline': mae < b_mae,
            'n_nonzero_coef': n_nonzero_coef,
            'solve_rate':    round(float(np.mean(test_verd)), 3),
        })
 
    if not per_user:
        print("No users had enough data to evaluate.")
        return {'per_user': [], 'aggregate': {}, 'tag_names': tag_names}
 
    # ── Aggregate stats ───────────────────────────────────────────────────────
    def _stats(key):
        vals = np.array([r[key] for r in per_user if r[key] is not None])
        if len(vals) == 0:
            return None
        return {
            'mean':   round(float(vals.mean()),   4),
            'std':    round(float(vals.std()),    4),
            'median': round(float(np.median(vals)), 4),
            'min':    round(float(vals.min()),    4),
            'max':    round(float(vals.max()),    4),
        }
 
    beats_baseline = sum(r['beats_baseline'] for r in per_user)
    auc_users      = [r for r in per_user if r['auc_roc'] is not None]
 
    aggregate = {
        'users_evaluated':       len(per_user),
        'users_skipped':         skipped,
        'beats_baseline_count':  beats_baseline,
        'beats_baseline_pct':    round(beats_baseline / len(per_user) * 100, 1),
        'mae':                   _stats('mae'),
        'rmse':                  _stats('rmse'),
        'accuracy':              _stats('accuracy'),
        'auc_roc':               _stats('auc_roc') if auc_users else None,
        'calibration_bias':      _stats('calib_bias'),
        'n_nonzero_coef':        _stats('n_nonzero_coef'),
        'baseline_mae_mean':     round(float(np.mean(baseline_mae)), 4) if baseline_mae else None,
    }
 
    # ── Print report ──────────────────────────────────────────────────────────
    w = 60
    print("\n" + "=" * w)
    print("LASSO EVALUATION REPORT")
    print("=" * w)
    print(f"  Users evaluated  : {aggregate['users_evaluated']}")
    print(f"  Users skipped    : {aggregate['users_skipped']}  (too few samples or no tag variance)")
    print(f"  Test fraction    : {test_fraction:.0%}  |  alpha = {alpha}")
    print()
 
    print("  ── Prediction error ─────────────────────────────────────")
    for metric in ('mae', 'rmse'):
        s = aggregate[metric]
        print(f"  {metric.upper():<12}  mean={s['mean']:.4f}  std={s['std']:.4f}"
              f"  median={s['median']:.4f}  [{s['min']:.4f} – {s['max']:.4f}]")
    print(f"  Baseline MAE     mean={aggregate['baseline_mae_mean']:.4f}  "
          f"(naive always-0.5 predictor)")
    print(f"  Beats baseline   {aggregate['beats_baseline_count']} / "
          f"{aggregate['users_evaluated']} users  "
          f"({aggregate['beats_baseline_pct']}%)")
    print()
 
    print("  ── Classification quality ───────────────────────────────")
    s = aggregate['accuracy']
    print(f"  Accuracy (≥0.5)  mean={s['mean']:.4f}  std={s['std']:.4f}"
          f"  median={s['median']:.4f}")
    if aggregate['auc_roc']:
        s = aggregate['auc_roc']
        print(f"  AUC-ROC          mean={s['mean']:.4f}  std={s['std']:.4f}"
              f"  median={s['median']:.4f}"
              f"  ({len(auc_users)} users with both classes in test set)")
    else:
        print("  AUC-ROC          n/a  (no user had both classes in test set)")
    print()
 
    print("  ── Calibration ──────────────────────────────────────────")
    s = aggregate['calibration_bias']
    direction = "over-confident" if s['mean'] > 0 else "under-confident"
    print(f"  Bias             mean={s['mean']:+.4f}  std={s['std']:.4f}  ({direction})")
    print()
 
    print("  ── Model sparsity ───────────────────────────────────────")
    s = aggregate['n_nonzero_coef']
    print(f"  Non-zero coef    mean={s['mean']:.1f}  std={s['std']:.1f}"
          f"  median={s['median']:.1f}  [{int(s['min'])} – {int(s['max'])}]"
          f"  (out of {len(tag_names)} features incl. __elo__)")
    print()
    print("=" * w)
 
    return {
        'per_user':  per_user,
        'aggregate': aggregate,
        'tag_names': tag_names,
    }
 


def sgd_evaluation(
    df: pd.DataFrame,
    components_range: list = None,
    nb_iterations: int = 20,
    gamma: float = 0.01,
    lambda_: float = 0.1,
    test_fraction: float = 0.2,
    missing_strategy: str = "ignore",
    cutoff_k: int = 200,
    seed: int = 42,
) -> dict:
    """
    Evaluate CFSGD accuracy and show the impact of nb_components.
 
    Strategy
    --------
    We split the full dataset into a global train set and test set
    (by row, not by user) — this mirrors matrix completion: the model
    sees most of the matrix and must predict the held-out cells.
 
    For each value of nb_components we:
      1. Train CFSGD on the train set.
      2. Compute BCE loss, MAE, RMSE, accuracy, AUC-ROC on train AND test.
      3. Record the train/test gap — a widening gap signals overfitting.
 
    Metrics reported per component count
    -------------------------------------
      BCE loss   : binary cross-entropy (the SGD training objective)
      MAE        : mean absolute error |predicted_prob - true_verdict|
      RMSE       : root mean squared error
      Accuracy   : fraction correctly classified at threshold 0.5
      AUC-ROC    : ranking quality (0.5 = random, 1.0 = perfect)
      overfit_gap: test_bce - train_bce  (positive = overfitting)
 
    Parameters
    ----------
    df                : full submission DataFrame
    components_range  : list of D values to try (default [1,2,4,8,16,32,64])
    nb_iterations     : SGD epochs for each run
    gamma             : learning rate
    lambda_           : L2 regularisation strength
    test_fraction     : fraction of rows held out for testing
    missing_strategy  : passed to CFRecommender for imputation
    cutoff_k          : cutoff threshold if strategy is "cutoff"
    seed              : random seed for reproducibility
 
    Returns
    -------
    dict with keys: results (list of per-D dicts), best_d (int)
    """
    from sklearn.metrics import roc_auc_score
 
    if components_range is None:
        components_range = [1, 2, 4, 8, 16, 32, 64]
 
    rng = np.random.default_rng(seed)
 
    # ── Preprocessing ─────────────────────────────────────────────────────────
    df = df.copy()
    df['PROBLEM_TAGS']   = df['PROBLEM_TAGS'].apply(_parse_tags)
    df['verdict_binary'] = (df['VERDICT'] == 'OK').astype(float)
 
    df['user_id'],    id_to_user,    user_to_id    = _encode_ids(df['USER'])
    df['problem_id'], id_to_problem, problem_to_id = _encode_ids(df['PROBLEM'])
 
    nb_users    = len(id_to_user)
    nb_problems = len(id_to_problem)
 
    # Apply missing strategy imputation (mirrors CFRecommender._build_training_arrays)
    rec_tmp = CFRecommender(
        df,
        missing_strategy=missing_strategy,
        
    )
    all_u, all_p, all_v = rec_tmp._build_training_arrays()
    n_total = len(all_u)
 
    # Global train/test split on rows
    indices  = rng.permutation(n_total)
    n_test   = max(1, int(n_total * test_fraction))
    n_train  = n_total - n_test
 
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]
 
    tr_u = all_u[train_idx];  tr_p = all_p[train_idx];  tr_v = all_v[train_idx]
    te_u = all_u[test_idx];   te_p = all_p[test_idx];   te_v = all_v[test_idx]
 
    # For AUC we need binary labels — cast test verdicts to int
    te_v_int = te_v.astype(int)
    both_classes_in_test = len(np.unique(te_v_int)) == 2
 
    eps = 1e-7
 
    def _bce(y_true, y_pred):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ))
 
    def _metrics(y_true, y_pred, binary_labels=None):
        mae      = float(np.mean(np.abs(y_pred - y_true)))
        rmse     = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        bce      = _bce(y_true, y_pred)
        pred_bin = (y_pred >= 0.5).astype(int)
        acc      = float(np.mean(pred_bin == y_true.astype(int)))
        if binary_labels is not None and len(np.unique(binary_labels)) == 2:
            auc = float(roc_auc_score(binary_labels, y_pred))
        else:
            auc = None
        return dict(bce=bce, mae=mae, rmse=rmse, accuracy=acc, auc_roc=auc)
 
    # ── Sweep over nb_components ──────────────────────────────────────────────
    results = []
    print(f"\nSGD EVALUATION  (strategy='{missing_strategy}', "
          f"epochs={nb_iterations}, γ={gamma}, λ={lambda_})")
    print(f"Train rows: {n_train:,}  |  Test rows: {n_test:,}")
    print(f"Users: {nb_users}  |  Problems: {nb_problems}")
    print("-" * 78)
    header = (f"  {'D':>4}  {'train_bce':>10}  {'test_bce':>10}  "
              f"{'gap':>8}  {'test_mae':>9}  {'test_acc':>9}  {'test_auc':>9}")
    print(header)
    print("-" * 78)
 
    for D in components_range:
        model = CFSGD(
            nb_components=D,
            nb_iterations=nb_iterations,
            gamma=gamma,
            lambda_=lambda_,
        )
        # Silence epoch prints during sweep
        import io, sys
        _old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        model.fit(tr_u, tr_p, tr_v, nb_users, nb_problems)
        sys.stdout = _old_stdout
 
        train_preds = model.predict(tr_u, tr_p)
        test_preds  = model.predict(te_u, te_p)
 
        tr_met = _metrics(tr_v, train_preds)
        te_met = _metrics(te_v, test_preds,
                          binary_labels=te_v_int if both_classes_in_test else None)
 
        gap = te_met['bce'] - tr_met['bce']
 
        auc_str = f"{te_met['auc_roc']:.4f}" if te_met['auc_roc'] else "   n/a"
        print(f"  {D:>4}  {tr_met['bce']:>10.4f}  {te_met['bce']:>10.4f}  "
              f"{gap:>+8.4f}  {te_met['mae']:>9.4f}  {te_met['accuracy']:>9.4f}  {auc_str:>9}")
 
        results.append({
            'nb_components':  D,
            'train_bce':      round(tr_met['bce'],      4),
            'test_bce':       round(te_met['bce'],      4),
            'overfit_gap':    round(gap,                4),
            'train_mae':      round(tr_met['mae'],      4),
            'test_mae':       round(te_met['mae'],      4),
            'train_accuracy': round(tr_met['accuracy'], 4),
            'test_accuracy':  round(te_met['accuracy'], 4),
            'test_auc_roc':   round(te_met['auc_roc'],  4) if te_met['auc_roc'] else None,
        })
 
    print("-" * 78)
 
    # ── Best D by test BCE ────────────────────────────────────────────────────
    best = min(results, key=lambda r: r['test_bce'])
    best_d = best['nb_components']
 
    print(f"\n  Best nb_components by test BCE : {best_d}")
    print(f"    train_bce={best['train_bce']:.4f}  "
          f"test_bce={best['test_bce']:.4f}  "
          f"gap={best['overfit_gap']:+.4f}  "
          f"test_auc={best['test_auc_roc'] if best['test_auc_roc'] else 'n/a'}")
 
    # ── Overfitting commentary ────────────────────────────────────────────────
    print()
    gaps = [r['overfit_gap'] for r in results]
    if gaps[-1] > gaps[0] * 1.5:
        print("  ⚠  Overfitting detected: gap widens as D increases.")
        print("     Consider reducing nb_components or increasing lambda_.")
    elif all(g < 0.01 for g in gaps):
        print("  ℹ  No overfitting detected. The model may still be underfitting.")
        print("     Try increasing nb_components or nb_iterations.")
    else:
        print("  ✓  Moderate overfitting pattern. The sweet spot is around D =", best_d)
 
    print()
 
    return {'results': results, 'best_d': best_d}




def knn_elo_analysis(
    df: pd.DataFrame,
    target_handle: str,
    k: int = 20,
    n_sample_users: int = 200,
    missing_strategy: str = "ignore",

):
    """
    Analyse the Elo distribution of KNN neighbourhoods.
 
    Part 1 — Single user
    --------------------
    For `target_handle`, find its k nearest neighbours, retrieve their
    Elo ratings, and print:
      - the sorted list of neighbour Elos
      - a text histogram of the distribution
      - mean and standard deviation
 
    Part 2 — Population study (n_sample_users users)
    --------------------------------------------------
    For each sampled user compute:
      - neighbourhood mean Elo
      - neighbourhood std Elo
      - gap = neighbourhood_mean - user_elo  (positive = neighbours rated higher)
    Then print aggregate statistics over the sample:
      - mean and std of per-user neighbourhood spread (std)
      - mean and std of the gap between neighbourhood mean and user Elo
 
    Parameters
    ----------
    df               : full submission DataFrame
    target_handle    : handle to inspect in Part 1
    k                : number of neighbours (must match the KNN used)
    n_sample_users   : how many users to include in Part 2
    missing_strategy : passed to CFRecommender
  
    """
    print(f"Building recommender  (strategy='{missing_strategy}', k={k})...")
    rec = CFRecommender(df, missing_strategy=missing_strategy)
    rec.fit(nb_neighbors=k, train_sgd=False, train_lasso=False)
 
    knn_model  = rec.knn
    user_elos  = df.drop_duplicates('USER').set_index('USER')['USER_ELO'].to_dict()
 
    # ── helper: get the k nearest neighbours for a given handle ──────────────
    def get_neighbour_elos(handle: str) -> np.ndarray:
        """
        Return an array of Elo ratings for the up-to-k nearest neighbours
        that have STRICTLY POSITIVE cosine similarity with this user.
 
        Neighbours with similarity = 0 share no problems with the target
        user — including them would be meaningless and inflate the sample
        with random Elos. They are excluded regardless of k.
        """
        if handle not in rec.user_to_id:
            return np.array([])
        uid  = rec.user_to_id[handle]
 
        # Copy to avoid mutating the shared similarity matrix across calls
        sims = knn_model.user_similarity[uid].copy()
        
 
        # Only consider neighbours with a genuine shared-problem overlap
        positive_mask = sims > 0
        if not positive_mask.any():
            return np.array([])
 
        # Among positive-similarity neighbours, take the top k
        candidate_ids = np.where(positive_mask)[0]
        candidate_ids = candidate_ids[np.argsort(sims[candidate_ids])[::-1]][:k]
 
        elos = []
        for nid in candidate_ids:
            neighbour_handle = rec.id_to_user.get(int(nid))
            if neighbour_handle is None:
                continue
            elo = user_elos.get(neighbour_handle)
            if elo is not None and elo > 0:   # skip placeholder 0-elo rows
                elos.append(float(elo))
        return np.array(elos)
 
    def text_histogram(values: np.ndarray, n_bins: int = 10, width: int = 40) -> str:
        """Return a compact ASCII histogram string."""
        if len(values) == 0:
            return "  (no data)"
        lo, hi  = values.min(), values.max()
        if lo == hi:
            return f"  all values = {lo:.0f}"
        bins    = np.linspace(lo, hi, n_bins + 1)
        counts, edges = np.histogram(values, bins=bins)
        max_count = counts.max() or 1
        lines = []
        for i, c in enumerate(counts):
            bar   = '█' * int(c / max_count * width)
            label = f"  [{edges[i]:6.0f} – {edges[i+1]:6.0f})  {bar}  {c}"
            lines.append(label)
        return '\n'.join(lines)
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Part 1 — single user
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PART 1 — KNN neighbourhood for '{target_handle}'  (k={k})")
    print('='*60)
 
    neighbour_elos = get_neighbour_elos(target_handle)
    user_elo       = user_elos.get(target_handle, None)
 
    if len(neighbour_elos) == 0:
        print("  No neighbours found.")
    else:
        print(f"  User Elo       : {user_elo}")
        print(f"  Neighbour Elos : {sorted(neighbour_elos.astype(int).tolist())}")
        print(f"\n  Distribution:")
        print(text_histogram(neighbour_elos))
        print(f"\n  Mean           : {neighbour_elos.mean():.1f}")
        print(f"  Std dev        : {neighbour_elos.std():.1f}")
        if user_elo is not None:
            gap = neighbour_elos.mean() - user_elo
            print(f"  Gap (mean - user Elo) : {gap:+.1f}")
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Part 2 — population study
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PART 2 — Population study  ({n_sample_users} users, k={k})")
    print('='*60)
 
    all_handles  = list(rec.user_to_id.keys())
    sample       = all_handles[:n_sample_users]   # take first N (deterministic)
 
    spreads, gaps = [], []
    for handle in sample:
        elos = get_neighbour_elos(handle)
        if len(elos) == 0:
            continue
        elo = user_elos.get(handle)
        if elo is None or elo == 0:  # skip placeholder 0-elo rows
            continue
        # std is 0 for a single neighbour — still valid for gap analysis
        spreads.append(elos.std() if len(elos) > 1 else 0.0)
        gaps.append(elos.mean() - elo)
 
    spreads = np.array(spreads)
    gaps    = np.array(gaps)
 
    if len(spreads) == 0:
        print("  Not enough data.")
        return
 
    print(f"  Users analysed : {len(spreads)}")
    print()
    print("  Neighbourhood SPREAD (std of neighbour Elos per user):")
    print(f"    mean spread : {spreads.mean():.1f}")
    print(f"    std  spread : {spreads.std():.1f}")
    print(f"    min  spread : {spreads.min():.1f}")
    print(f"    max  spread : {spreads.max():.1f}")
    print()
    print("  GAP = neighbourhood_mean_elo − user_elo:")
    print(f"    mean gap    : {gaps.mean():+.1f}  "
          f"({'neighbours rated higher on avg' if gaps.mean() > 0 else 'neighbours rated lower on avg'})")
    print(f"    std  gap    : {gaps.std():.1f}")
    print(f"    min  gap    : {gaps.min():+.1f}")
    print(f"    max  gap    : {gaps.max():+.1f}")
    print()
    print("  Gap distribution:")
    print(text_histogram(gaps, n_bins=10, width=35))
 


if __name__ == '__main__':
    
    file_path ="codeforces_submissions.csv"

    target_user="" #Target user's name - ensure he is present in the dataframe
    
    
    df = pd.read_csv(file_path)
    
    # Standardize column names (optional but recommended)
    df.columns = [
        "USER",
        "USER_ELO",
        "PROBLEM",
        "PROBLEM_TAGS",
        "PROBLEM_ELO",        
        "VERDICT"
    ]
    
    # Convert tags from string to list (if stored like "dp,greedy,math")
    df["PROBLEM_TAGS"] = df["PROBLEM_TAGS"].apply(
        lambda x: x.split("|") if isinstance(x, str) else []
    )
    
    print(f"Dataset: {len(df)} submissions, {df['USER'].nunique()} users, "
          f"{df['PROBLEM'].nunique()} problems\n")
    

    print("\n\nRunning SGD evaluation...")
    sgd_evaluation(
        df,
        components_range=[64],
        nb_iterations=50,
        missing_strategy='ignore',
    )
    
    """
    knn_elo_analysis(
        df,
        target_handle=target_user,
        k=20,
        n_sample_users=200,
        missing_strategy='ignore',
    )
    
    rec = CFRecommender(df, target_prob_lo=0.50, target_prob_hi=0.65,missing_strategy="ignore",lasso_print_tags=True) # Strategies : "zero", "ignore", "elo_fill"
    rec.fit(nb_neighbors=10, nb_components=10, nb_iterations=10)

    print(f"\nTop 5 recommendations for '{target_user}' (ensemble):")
    print(rec.rank_candidates(target_user, model='ensemble', top_k=5).to_string(index=False))

    print(f"\nSingle best recommendation (SGD):")
    print(rec.recommend(target_user, model='sgd'))

    print(f"\nTag preferences (LASSO) for '{target_user}':")
    print(rec.explain(target_user))
    """