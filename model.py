"""Recommender system models."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from .util import prepare_data

class RecommenderBase:
    """Recommender base class based on PMF."""

    def __init__(self,
                 dataset,
                 num_users,
                 num_items,
                 rank=15,
                 optimizer='AdamOptimizer',
                 learning_rate=0.001,
                 lambda_users=0.1,
                 lambda_items=0.1,
                 batch_size=500,
                 item_nn_module=None,
                 text_descriptions=None,
                 lambda_weights=None,
                 validation_set=None):
        """
        :param dataset: TF dataset
        :param num_users: number of users
        :param num_items: number of items
        :param rank: rank of user-item matrix
        :param optimizer: TF optmizer
        :param learning_rate: learning rate
        :param lambda_users: regularisation of user params
        :param lambda_items: regularisation of item params
        :param batch_size: batch size
        :param item_nn_module: Keras model for neural network item module
        :param text_descriptions: textual description of items
        :param lambda_weights: regularisation of NN weights
        :param validation_set: TF dataset
        """

        self.num_users = num_users
        self.num_items = num_items
        self.rank = rank

        # User Matrix
        self.user_matrix = tfe.Variable(
            initial_value=tf.random_uniform([self.num_users, self.rank]),
            name="users"
        )
        # Item Matrix
        self.item_matrix = tfe.Variable(
            initial_value=tf.random_uniform([self.num_items, self.rank]),
            name="items"
        )
        # Item NN Module
        self.item_module = item_nn_module

        # Item text descriptions
        self.texts = text_descriptions
        # Dataset
        self.dataset = dataset
        self.validation_set = validation_set
        self.batch_size = batch_size
        # Optimizer
        self.optimizer = tf.train.__dict__[optimizer](learning_rate)
        # Regularisation
        self.lambda_users = lambda_users
        self.lambda_items = lambda_items
        self.lambda_weights = lambda_weights or 0.1

    def _predict(self, users, items):
        """Predict for a set of users and items.

        :param users: int tensor of size N x 1
        :param items: int tensor of size N x 1

        """
        U = tf.gather(self.user_matrix, users)
        V = tf.gather(self.item_matrix, items)
        # Matrix multiplication step
        result = tf.reduce_sum(tf.multiply(U, V), axis=1)
        return result

    def predict(self, dataset):
        """Predict on dataset.

        :param dataset: TF dataset

        """
        data = dataset.batch(self.batch_size)
        iterator = tfe.Iterator(data)
        result = []
        for batch in iterator:
            (users, items), ratings = batch
            result.append(self._predict(users, items))
        return tf.concat(result, axis=0)

    def loss(self, dataset):
        """Loss for a whole dataset without regularisation.
        
        :param dataset: TF dataset.
        
        """
        data = dataset.batch(self.batch_size)
        iterator = tfe.Iterator(data)
        loss = 0.0
        for batch in iterator:
            loss += self.batch_loss(batch)
        return loss

    def batch_loss(self, batch):
        """PMF loss without regularisation

        :param batch: batch size

        """
        (users, items), ratings = batch
        N = ratings.get_shape().as_list()[0]
        # Predict on training set
        values = self._predict(users, items)
        # Calculate Squared Error
        cost = tf.reduce_sum((values-ratings)**2)
        cost = cost/N
        return cost


class ImplicitMixin:
    """Mixin to make a recommender implicit."""

    def __init__(self, alpha=1., eps=1.):
        self.alpha = alpha
        self.eps = eps

    def _regulariser(self, batch):
        """Regulariser for loss function
        
        :param batch: batch size
        
        """
        (users, items), (inv_variances, p) = batch
        N = p.get_shape().as_list()[0]
        r = 0
        if self.item_module:
            idx = np.random.permutation(range(self.num_items))[:self.batch_size]
            item_reg = self.item_module.call_features(tf.gather(self.texts, idx))
            batch_item_matrix = tf.gather(self.item_matrix, idx)
            for w in self.item_module.trainable_weights:
                r += self.lambda_weights*tf.reduce_sum(w**2)
        else:
            item_reg = tf.constant(tf.zeros_like(self.item_matrix))
            batch_item_matrix = self.item_matrix
        r += self.lambda_users*tf.reduce_sum(self.user_matrix**2)
        r += (N / self.batch_size) * self.lambda_items * tf.reduce_sum((batch_item_matrix - item_reg) ** 2)
        return r/N

    def _mutate_dataset(self, dataset):
        """Transform the dataset as required for implicit computations."""

        (users, items), confidences = dataset._tensors

        def _create_implicit_data(confidence):
            inv_variance = 1.0 + self.alpha * tf.log(
                1.0 + confidence / self.eps
            )
            p = tf.cast(
                tf.greater(confidence, 0.0),
                dtype=tf.float32
            )
            return inv_variance, p

        inv_variances, p = _create_implicit_data(confidences)
        transformed_data = prepare_data(
            ((users, items), (inv_variances, p)),
            ((tf.int32, tf.int32), (tf.float32, tf.float32))
        )
        return tf.data.Dataset.from_tensor_slices(transformed_data)

    
    def batch_loss(self, batch):
        """PMF loss without regularisation
        
        :param batch: batch size
        
        """
        (users, items), (inv_variances, ps) = batch
        N = inv_variances.get_shape().as_list()[0]
        # Predict on training set
        values = self._predict(users, items)
        # Calculate Squared Error
        cost = tf.reduce_sum(inv_variances * (values - ps) ** 2)
        cost = cost/N
        return cost

   

class SGDRecommender(RecommenderBase):
    """Recommender class trained with SGD."""

    def _regulariser(self, batch):
        """Regulariser for loss function

        :param batch: batch size

        """
        (users, items), ratings = batch
        N = ratings.get_shape().as_list()[0]
        r = 0
        if self.item_module:
            idx = np.random.permutation(range(self.num_items))[:self.batch_size]
            item_reg = self.item_module.call_features(tf.gather(self.texts, idx))
            batch_item_matrix = tf.gather(self.item_matrix, idx)
            for w in self.item_module.trainable_weights:
                r += self.lambda_weights*tf.reduce_sum(w**2)
        else:
            item_reg = tf.constant(tf.zeros_like(self.item_matrix))
            batch_item_matrix = self.item_matrix
        r += self.lambda_users*tf.reduce_sum(self.user_matrix**2)
        r += (N / self.batch_size) * self.lambda_items*tf.reduce_sum((batch_item_matrix-item_reg)**2)
        return r/N

    def training_loss(self, batch):
        """Training Loss

        :param batch: batch size

        """
        return self.batch_loss(batch) + self._regulariser(batch)

    def train(self,
              n_iter,
              verbose=True,
              tol=1e-3,
              stop_iter=3,
              seed=42):
        """Method to perform training.

        :param n_iter: number of iterations
        :param verbose: print progress (Default value = True)
        :param tol: early stopping criterion - if the validation loss does not
            decrease by more than this amount, start counting down to stopping
            threshold
        :param stop_iter: number of epochs to wait for loss to start
            decreasing adequately again before stopping
        :param seed: random seed for reproducibility.
        """
        data = self.dataset.shuffle(100, seed=seed).batch(self.batch_size)
        
        train_size = tf.shape(self.dataset._tensors[0]).numpy()[1]
        
        iterator = tfe.Iterator(data)
        until_stop = stop_iter
        train_loss = []
        val_loss = []
        best_val_loss = None
        
        best_user_matrix = self.user_matrix.numpy().copy()
        best_item_matrix = self.item_matrix.numpy().copy()
        
        for i in range(n_iter):
            try:
                batch = iterator.next()
            except StopIteration:
                iterator = tfe.Iterator(data)
                batch = iterator.next()

            self.optimizer.minimize(
                lambda: self.training_loss(batch),
                global_step=tf.train.get_or_create_global_step()
            )

            if i % (train_size // self.batch_size) == 0:
                v = 0
                for j, b in enumerate(
                    tfe.Iterator(
                        self.validation_set.batch(self.batch_size)
                    )
                ):
                    v += self.batch_loss(b)
                t = 0
                for k, b in enumerate(
                    tfe.Iterator(
                        self.dataset.batch(self.batch_size)
                    )
                ):
                    t += self.batch_loss(b)
                new_val_loss = v / (j + 1)
                val_loss.append(new_val_loss)
                new_train_loss = t / (k + 1)
                train_loss.append(new_train_loss)
                if i == 0:
                    best_val_loss = new_val_loss
                    
                    best_user_matrix = self.user_matrix.numpy().copy()
                    best_item_matrix = self.item_matrix.numpy().copy()
                    
                elif i > 1 and best_val_loss - new_val_loss < tol:
                    until_stop -= 1
                    if until_stop == 0:
                        print(
                            ("Stopping early after {} epochs.\n"
                             "Previous loss = {},\n"
                             "current loss = {},\n"
                             "difference = {}").format(
                                int(i / (train_size // self.batch_size)),
                                best_val_loss,
                                new_val_loss,
                                best_val_loss - new_val_loss
                            )
                        )
                        break
                elif i > 1:
                    best_val_loss = new_val_loss
                    
                    best_user_matrix = self.user_matrix.numpy().copy()
                    best_item_matrix = self.item_matrix.numpy().copy()
                    
                if verbose:
                    print(
                        'Validation loss = {}'.format(
                            new_val_loss
                        )
                    )
        self.item_matrix = tfe.Variable(initial_value = best_item_matrix)
        self.user_matrix = tfe.Variable(initial_value = best_user_matrix)

        return train_loss, val_loss


class ImplicitSGDRecommender(ImplicitMixin, SGDRecommender):
    """An implicit version of SGD PMF.

    When passing the TF Dataset to this class, instead of providing the
    explicit, partial user-item matrix as before , pass the user-item
    confidence matrix. Note that you *must include all of the user-item pairs*
    """

    def __init__(self, *args, **kwargs):
        """
        :param dataset: TF dataset
        :param num_users: number of users
        :param num_items: number of items
        :param rank: rank of user-item matrix
        :param optimizer: TF optmizer
        :param learning_rate: learning rate
        :param lambda_users: regularisation of user params
        :param lambda_items: regularisation of item params
        :param batch_size: batch size
        :param item_nn_module: Keras model for neural network item module
        :param text_descriptions: textual description of items
        :param lambda_weights: regularisation of NN weights
        :param validation_set: TF dataset
        :param alpha: alpha parameter in inverse variance calculation
        :param eps: epsilon parameter in inverse variance calculation
        """
        rec_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["alpha", "eps"]
        }
        SGDRecommender.__init__(self, *args, **rec_kwargs)
        ImplicitMixin.__init__(self,
                               alpha=kwargs.pop("alpha"),
                               eps=kwargs.pop("eps"))
                               
        self.dataset = self._mutate_dataset(self.dataset)
        if self.validation_set is not None:
            self.validation_set = self._mutate_dataset(self.validation_set)


class CoordinateDescentRecommender(RecommenderBase):
    """Recommender class based on PMF implementing coordinate descent."""

    def __init__(self,
                 dataset,
                 num_users,
                 num_items,
                 rank=15,
                 optimizer='AdamOptimizer',
                 learning_rate=0.001,
                 inner_loop_iterations=10,
                 gamma=0.5,
                 lambda_users=0.1,
                 lambda_items=0.1,
                 batch_size=500,
                 item_nn_module=None,
                 text_descriptions=None,
                 lambda_weights=None,
                 validation_set=None):
        """
        :param dataset: TF dataset
        :param num_users: number of users
        :param num_items: number of items
        :param rank: rank of user-item matrix
        :param optimizer: TF optmizer
        :param learning_rate: learning rate
        :param gamma: step size for coordinate updates in [0, 1]
        :param lambda_users: regularisation of user params
        :param lambda_items: regularisation of item params
        :param batch_size: batch size
        :param item_nn_module: Keras model for neural network item module
        :param text_descriptions: textual description of items
        :param lambda_weights: regularisation of NN weights
        :param validation_set: TF dataset
        """
        super(CoordinateDescentRecommender, self).__init__(
            dataset,
            num_users,
            num_items,
            rank=rank,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lambda_users=lambda_users,
            lambda_items=lambda_items,
            batch_size=batch_size,
            item_nn_module=item_nn_module,
            text_descriptions=text_descriptions,
            lambda_weights=lambda_weights,
            validation_set=validation_set
        )
        self.inner_loop_iterations = inner_loop_iterations
        assert(gamma >= 0 and gamma <= 1)
        self.gamma = gamma

    def nn_loss(self):
        """ Compute neural network module loss. """
        idx = np.random.permutation(range(self.num_items))[:self.batch_size]
        pred = self.item_module.call_features(tf.gather(self.texts, idx))

        cost = tf.reduce_sum((tf.gather(self.item_matrix, idx) - pred) ** 2.) / self.batch_size
        for w in self.item_module.trainable_weights:
            cost += self.lambda_weights * (tf.reduce_sum(w ** 2.))
        return cost


    def _update_U(self, gamma, verbose=True):
        """ Update user matrix. """
        if verbose:
            print("Updating U...")
        (users, items), ratings = self.dataset._tensors
        a = 1
        b = 0
        VV = b * (
            tf.matmul(
                self.item_matrix,
                self.item_matrix,
                transpose_a=True
            )
        )
        VV += self.lambda_users * tf.eye(self.rank)
        i = tf.constant(0)
        condition = lambda i: tf.less(i, self.num_users)
        def body(i):
            item_idx = tf.boolean_mask(items, tf.equal(users, i))
            V_i = tf.gather(self.item_matrix, item_idx)
            R_i = tf.boolean_mask(ratings, tf.equal(users, i))
            A = VV + (a - b) * (tf.matmul(V_i, V_i, transpose_a=True))
            B = a * tf.matmul(V_i, R_i[:, None], transpose_a=True)
            update = tf.matrix_solve(A, B)
            update = tf.transpose(gamma * self.user_matrix[i, :]) + (1 - gamma) * update
            tf.scatter_update(self.user_matrix, i, update)
            return tf.add(i, 1)
        tf.while_loop(condition, body, [i], parallel_iterations=128)

    def _update_V(self, gamma, verbose=True):
        """ Update item matrix. """
        if verbose:
            print("Updating V...")
        (users, items), ratings = self.dataset._tensors
        a = 1
        b = 0
        UU = b * tf.matmul(
            self.user_matrix,
            self.user_matrix,
            transpose_a=True
            )
        if self.item_module is not None:
            theta = self.item_module.features_in_batches(
                self.texts,
                self.batch_size
            )

        j = tf.constant(0)
        condition = lambda j: tf.less(j, self.num_items)
        def body(j):
            user_idx = tf.boolean_mask(users, tf.equal(items, j))
            U_j = tf.gather(self.user_matrix, user_idx)
            R_j = tf.boolean_mask(ratings, tf.equal(items, j))

            tmp_A = UU + (a - b) * tf.matmul(U_j, U_j, transpose_a=True)
            A = tmp_A + self.lambda_items * tf.eye(self.rank)
            B = a * tf.matmul(U_j, R_j[:, None], transpose_a=True)
            if self.item_module is not None:
                B += self.lambda_items * theta[j][:, None]
            update = tf.matrix_solve(A, B)
            update = tf.transpose(gamma * self.item_matrix[j, :]) + (1 - gamma) * update
            tf.scatter_update(self.item_matrix, j, update)
            return tf.add(j, 1)
        tf.while_loop(condition, body, [j], parallel_iterations=128)

    def _update_W(self, optimizer, n_iter, verbose=True):
        """ Update neural network weights.

        :param optimizer: TF optmizer
        :param n_iter: number of iterations
        """
        if verbose:
            print("Updating W...")
        for _ in range(n_iter):
            optimizer.minimize(
                lambda: self.nn_loss(),
                global_step=tf.train.get_or_create_global_step()
            )

    def train(
        self, n_iter, verbose=True, tol=1e-3, monitor="val", stop_iter=3
    ):
        """Train model with coordinate descent.
        Use coordinate descent, alternating between fitting user params,
        then item params then neural network params.

        :param optimizer: TF optimizer
        :param n_iter: number of iterations
        :param verbose: verbose output if True.
        :param tol:
            absolute tolerance for decrease in loss for early stopping.
        :param stop_iter:
            number of iterations to wait before loss decreases again.
        """

        if monitor == "val" and self.validation_set is None:
            raise ValueError(
                ("You must pass a validation set to the constructor if the "
                 "validation loss is used for early stopping.")
            )
        elif monitor not in ["val", "train"]:
            raise ValueError("monitor must be set to 'val' or 'train'")

        if monitor == "val":
            monitor_dataset = self.validation_set
        else:
            monitor_dataset = self.dataset

        until_stop = stop_iter
        best_loss = self.loss(monitor_dataset)
        losses = [best_loss.numpy(), ]
        
        best_item_matrix = self.item_matrix.numpy().copy()
        best_user_matrix = self.user_matrix.numpy().copy()

        if verbose:
            print("Initial {0} loss = {1:.4f}".format(monitor, best_loss))
        for i in range(n_iter):
            if verbose:
                print("Entering iteration {}".format(i))

            with tf.device("/cpu:0"):
                self._update_U(self.gamma, verbose=verbose)
                self._update_V(self.gamma, verbose=verbose)

            if self.item_module is not None:
                if tfe.num_gpus() > 0:
                    with tf.device("/gpu:0"):
                        self._update_W(
                            self.optimizer,
                            self.inner_loop_iterations,
                            verbose=verbose
                        )
                else:
                    self._update_W(
                        self.optimizer,
                        self.inner_loop_iterations,
                        verbose=verbose
                    )

            new_loss = self.loss(monitor_dataset)
            losses.append(new_loss.numpy())
            print(monitor + " loss is {:.4f}".format(new_loss))
            delta = best_loss - new_loss
            if delta < tol:
                until_stop -= 1
                if until_stop <= 0:
                    print(
                        ("Stopping after {} iterations, "
                         "convergence criterion met.").format(i + 1)
                    )
                    break
            else:
                best_loss = new_loss
                best_item_matrix = self.item_matrix.numpy().copy()
                best_user_matrix = self.user_matrix.numpy().copy()

        print("Best loss is: {}".format(best_loss))
        print("Compare item matrix {}".format(tf.reduce_all(tf.equal(self.item_matrix, tfe.Variable(initial_value=best_item_matrix)))))
        print("Compare user matrix {}".format(tf.reduce_all(tf.equal(self.user_matrix, tfe.Variable(initial_value=best_user_matrix)))))

        self.item_matrix = tfe.Variable(initial_value = best_item_matrix)
        self.user_matrix = tfe.Variable(initial_value = best_user_matrix)
        return losses


class ImplicitCDRecommender(ImplicitMixin, CoordinateDescentRecommender):
    """An implicit version of coordinate descent PMF.
    When passing the TF Dataset to this class, instead of providing the
    explicit, partial user-item matrix as before , pass the user-item
    confidence matrix. Note that you *must include all of the user-item pairs*.
    """

    def __init__(self, *args, **kwargs):
        """
        :param dataset: TF dataset
        :param num_users: number of users
        :param num_items: number of items
        :param rank: rank of user-item matrix
        :param optimizer: TF optmizer
        :param learning_rate: learning rate
        :param lambda_users: regularisation of user params
        :param lambda_items: regularisation of item params
        :param batch_size: batch size
        :param item_nn_module: Keras model for neural network item module
        :param text_descriptions: textual description of items
        :param lambda_weights: regularisation of NN weights
        :param validation_set: TF dataset
        :param alpha: alpha parameter in inverse variance calculation
        :param eps: epsilon parameter in inverse variance calculation
        """
        rec_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["alpha", "eps"]
        }
        CoordinateDescentRecommender.__init__(self, *args, **rec_kwargs)
        ImplicitMixin.__init__(self,
                               alpha=kwargs.pop("alpha"),
                               eps=kwargs.pop("eps"))
        self.dataset = self._mutate_dataset(self.dataset)
        if self.validation_set is not None:
            self.validation_set = self._mutate_dataset(self.validation_set)

    def _update_U(self, gamma, verbose=True):
        """ Update user matrix. """
        if verbose:
            print("Updating U...")
        (users, items), (inv_variances, ps) = self.dataset._tensors
        i = tf.constant(0)
        condition = lambda i: tf.less(i, self.num_users)
        def body(i):
            item_idx = tf.boolean_mask(items, tf.equal(users, i))

            C_i = tf.diag(tf.gather(inv_variances, item_idx))
            p_i = tf.gather(ps, item_idx)
            VT_Ci_V = tf.matmul(
                self.item_matrix,
                tf.matmul(
                    C_i,
                    self.item_matrix
                ),
                transpose_a=True
            )
            A = VT_Ci_V + self.lambda_users * tf.eye(self.rank)
            B = tf.matmul(
                self.item_matrix,
                tf.matmul(
                    C_i,
                    p_i[:, None]
                ),
                transpose_a=True
            )
            update = tf.matrix_solve(A, B)
            update = gamma * tf.expand_dims(self.user_matrix[i, :], -1) + (1 - gamma) * update
            tf.scatter_update(self.user_matrix, i, update)
            return tf.add(i, 1)

        tf.while_loop(condition, body, [i], parallel_iterations=128)

    def _update_V(self, gamma, verbose=True):
        """ Update item matrix. """

        if verbose:
            print("Updating V...")
        (users, items), (inv_variances, ps) = self.dataset._tensors
        if self.item_module is not None:
            theta = self.item_module.features_in_batches(
                self.texts,
                self.batch_size
            )
        j = tf.constant(0)
        condition = lambda j: tf.less(j, self.num_items)
        def body(j):
            user_idx = tf.boolean_mask(users, tf.equal(items, j))

            C_j = tf.diag(tf.gather(inv_variances, user_idx))
            p_j = tf.gather(ps, user_idx)
            UT_Cj_U = tf.matmul(
                self.user_matrix,
                tf.matmul(
                    C_j,
                    self.user_matrix
                ),
                transpose_a=True
            )
            A = UT_Cj_U + self.lambda_items * tf.eye(self.rank)
            B = tf.matmul(
                self.user_matrix,
                tf.matmul(
                    C_j,
                    p_j[:, None]
                ),
                transpose_a=True
            )
            if self.item_module is not None:
                B += self.lambda_items * theta[j][:, None]

            update = tf.matrix_solve(A, B)
            update = gamma * tf.expand_dims(self.item_matrix[j, :], -1) + (1 - gamma) * update
            tf.scatter_update(self.item_matrix, j, update)
            return tf.add(j, 1)

        tf.while_loop(condition, body, [j], parallel_iterations=128)
