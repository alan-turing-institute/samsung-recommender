"""Set of utilities that do not fit elsewhere."""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import os

from tqdm import tqdm
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error

def _assert_same_structure(t1, t2):
    """Throw error if tuples t1 and t2 have different structures.

    :param t1: first tuple
    :param t2: second tuple
    :type t1: tuple
    :type t2: tuple

    """
    if isinstance(t1, tuple):
        if isinstance(t2, tuple):
            assert(len(t1)==len(t2))
            for tt1, tt2 in zip(t1, t2):
                _assert_same_structure(tt1, tt2)
        else:
            raise(ValueError)
    else:
        if isinstance(t2, tuple):
            raise(ValueError)


def prepare_data(data, types):
    """Takes in tuple of numpy arrays and data types and creates TF tensors
     corresponding to each array and type.

    :param data: tuple of numpy arrays
    :param types: tuple of data types
    :returns: tuple of TF tensors

    """
    _assert_same_structure(data, types)

    def _to_tensor(datum, dtype):
        return tfe.Variable(datum, dtype=dtype, trainable=False)

    if isinstance(data, tuple):
        return tuple(
            prepare_data(datum, dtype) for datum, dtype in zip(data, types)
        )
    else:
        return _to_tensor(data, types)


class MLDataManager:
    """A simple python class for manipulating the movies dataset from kaggle.
    
    The url for this dataset is:
        https://www.kaggle.com/rounakbanik/the-movies-dataset/version/2/data
    
    Download the three tables, unzip them where required and put them in some
    directory. Then set `data_path` in the constructor to the path of this
    directory.
    """

    def __init__(self,
                 data_path,
                 vocab_size=8000,
                 max_len=200,
                 implicit=False,
                 pretrained_embeddings=False):
        """"
        :param data_path: path to MovieLens dataset
        :param vocab_size: vocabulary size
        :param max_len: maximum length of movie descriptions
        :param implicit: if True turns dataset into implicit
        :param pretrained_embeddings: if True use pretrained embeddings
        """

        tables_to_filenames = {
            "movie_summaries": "movies_metadata",
            "ratings": "ratings_small",
            "source_ids": "links_small"
        }
        tables = {
            table: pd.read_csv(
                os.path.join(data_path, tables_to_filenames[table] + ".csv")
            )
            for table in tables_to_filenames.keys()
        }
        self.data_path = data_path
        self.tables = tables
        self.movie_summaries = tables["movie_summaries"]
        self.ratings = tables["ratings"]
        self.source_ids = tables["source_ids"]
        self.train = None
        self.test = None
        self.implicit = implicit
        
        # remove spurious movie ids from the summaries table
        self.movie_summaries = self.movie_summaries.loc[
            ~self.movie_summaries["id"].str.contains('-'),
            :
        ]
        self.movie_summaries.loc[:, "id"] = self.movie_summaries["id"].astype(int)

        self.movie_summaries = self.movie_summaries.merge(
            self.source_ids, left_on="id", right_on="tmdbId"
        )[["movieId", "original_title", "overview", "genres"]]

        # remove movies without summaries
        ids_no_description = self.movie_summaries.loc[
            self.movie_summaries["overview"].isnull(),
            "movieId"
        ]
        self.ratings = self.ratings.loc[
            ~self.ratings["movieId"].isin(ids_no_description),
            :
        ]
        self.movie_summaries = self.movie_summaries.loc[
            ~self.movie_summaries["movieId"].isin(ids_no_description),
            :
        ]
        self._make_genre_vectors()
        self._make_movie_pretraining_data(vocab_size, max_len, pretrained_embeddings)

        # remove movies that are in the ratings but aren't in descriptions
        to_remove = list(set(self.ratings["movieId"]).symmetric_difference(
            set(self.movie_summaries["movieId"])
        ))
        self.ratings = self.ratings.loc[
            ~self.ratings["movieId"].isin(to_remove),
            :
        ]
        self.movie_summaries = self.movie_summaries.loc[
            ~self.movie_summaries["movieId"].isin(to_remove),
            :
        ]
        
        if self.implicit:
            self.ratings = self.ratings.pivot(
                index="userId",
                columns="movieId",
                values="rating"
            ).stack(dropna=False).fillna(0.0).reset_index().rename(
                columns={0: "rating"}
            )   
    
    def train_test_split(self, training_frac=0.8, random_state=42):
        """Split the data into training and test sets.
        
        :param training_frac: fraction of data to include in training set.
        (Default value = 0.8)
        :type training_frac: float
        :param random_state: random seed. (Default value = 42)
        :type random_state: int
        
        """
        if self.implicit:
            self.test = self.ratings.sample(
                frac=training_frac, random_state=random_state
            )
            self.train = self.ratings.copy()
            # implicit case, set missing values to zero
            self.train.loc[self.test.index, "rating"] = 0.0
        else:
            self.train = self.ratings.sample(
                frac=training_frac, random_state=random_state
            )
            self.test = self.ratings.drop(self.train.index)

            # remove users and movies from test that aren't in both sets
            users = list(
                set(
                    self.test["userId"]
                ).symmetric_difference(set(self.train["userId"]))
            )
            items = list(
                set(
                    self.test["movieId"]
                ).symmetric_difference(set(self.train["movieId"]))
            )
            include_train = ~self.train["userId"].isin(users)
            include_train = include_train & ~self.train["movieId"].isin(items)
            self.train = self.train.loc[include_train, :]
            include_test = ~self.test["userId"].isin(users)
            include_test = include_test & ~self.test["movieId"].isin(items)
            self.test = self.test.loc[include_test, :]

        # Relabel movie & users
        self.user_dict = {u: i for i, u in enumerate(self.train['userId'].unique())}
        self.item_dict = {i: j for j, i in enumerate(self.train['movieId'].unique())}

        self.reverse_user_dict = {v: k for k, v in self.user_dict.items()}
        self.reverse_item_dict = {v: k for k, v in self.item_dict.items()}

        self.train['userId'] = self.train['userId'].apply(lambda x: self.user_dict[x])
        self.train['movieId'] = self.train['movieId'].apply(lambda x: self.item_dict[x])
        self.train['rating'] = self.train['rating'] / 5.0

        self.test['userId'] = self.test['userId'].apply(lambda x: self.user_dict[x])
        self.test['movieId'] = self.test['movieId'].apply(lambda x: self.item_dict[x])
        self.test['rating'] = self.test['rating'] / 5.0

        self.texts = [
            self.pretraining_data[self.reverse_item_dict[i]][0]
            for i in range(len(self.item_dict))
        ]
        self.texts = np.vstack(self.texts)

    def validation_split(self, folds=10, random_state=42):
        """Split the data into folds for cross validation.

        :param folds: number of folds. (Default value = 10)
        :type folds: optional
        :param random_state: random seed. (Default value = 42)
        :type random_state: optional

        """
        train_shuffle = self.train.sample(frac=1.0, random_state=random_state)
        return np.array_split(train_shuffle, folds)
    
    def _make_movie_pretraining_data(self, vocab_size, maxlen, pretrained_embeddings):
        """Tokenize the movie descriptions and create training data with genres.

        :param vocab_size: size of vocabulary to use by taking the top n most
                              frequent words. (Default value = 8000)
        :type vocab_size: int
        :param maxlen: the maximum sequence length to use. (Default value = 200)
        :type maxlen: int

        """
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=True)
        tokenizer.fit_on_texts(self.movie_summaries['overview'])
        text_input = tokenizer.texts_to_sequences(self.movie_summaries['overview'])
        text_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            text_input, maxlen=maxlen, padding='post'
        )
        self.pretraining_data = {
            item_id:(item_text, item_genres) 
            for item_id, item_text, item_genres
            in zip(self.movie_summaries['movieId'],
                   text_sequences,
                   self._make_genre_vectors())
        }
        
        if pretrained_embeddings:
            embeddings_index = dict()
            embedding_file = self.data_path + "/glove.6B.100d.txt"
            f = open(embedding_file)
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
            f.close()
    
            num_words = np.min([vocab_size, len(tokenizer.word_index)+1])
            embedding_matrix = np.zeros((num_words + 1, 100))

            for word, i in tokenizer.word_index.items():
                if i>vocab_size:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector    
     
            self.embedding_matrix = embedding_matrix
    
    
    def _make_genre_vectors(self):
        """Create binary vectors of genre labels for pre-training."""
        f = lambda x: sorted([int(y.split(',')[0].strip("}]")) for y in x.split("'id': ")[1:]])
        genres = self.movie_summaries["genres"].apply(f)
        genre_set = set([genre for sublist in genres for genre in sublist])
        g = lambda x: np.array([1.0 if h in x else 0.0 for h in genre_set])
        return list(map(g, genres))
    
class GridSearchCV:
    """Wrapper method around Recommender class allowing for cross-validaiton"""
    def __init__(self, recommender_class, split_data, folds, metric,
                 param_dict, optimizer, n_iter,
                 batch_size=500,
                 random_state=42,
                 **kwargs):
        """

        :param recommender_class: recommender system class
        :param split_data: dataset that is split into train/test
        :type split_data: MLDataManager
        :param folds: number of folds
        :param metric: metric to compare configurations in
        :param param_dict: dictionary containing parameters to be searched
        :param optimizer: TF optimizer
        :param n_iter: number of training iterations per fit
        :param batch_size: batch_size
        :param random_state: random seed
        :param kwargs: keyword arguments to be passed to Recommender class
        """
        self.recommender_class = recommender_class
        self.data = split_data
        self.folds = folds
        self.metric = metric
        self.param_dict = param_dict
        self.params = list(param_dict.keys())
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.best_model = None
        self.mesh_dict = None

        self.num_users = self.data.train['userId'].unique().shape[0]
        self.num_items = self.data.train['movieId'].unique().shape[0]

        self.recommender_kwargs = kwargs
        self.rank = kwargs.get('rank')
        self.lambda_users = kwargs.get('lambda_users')
        self.lambda_items = kwargs.get('lambda_items')
        self.lambda_weights = kwargs.get('lambda_weights')

    def _prepare_folds(self):
        """ Split training data into folds. """
        df_folds = self.data.validation_split(self.folds, self.random_state)
        fold_splits = []
        for i in range(len(df_folds)):
            validation = df_folds[i]
            train = pd.concat(df_folds[:i] + df_folds[i + 1:])

            validation_tuple = (
                (validation['userId'].values,
                 validation['movieId'].values),
                validation['rating'].values
            )

            validation_tuple = prepare_data(validation_tuple,
                                            ((tf.int32, tf.int32), tf.float32))

            train_tuple = (
                (train['userId'].values,
                 train['movieId'].values),
                train['rating'].values
            )
            train_tuple = prepare_data(train_tuple,
                                       ((tf.int32, tf.int32), tf.float32))

            validation_dataset = tf.data.Dataset.from_tensor_slices(validation_tuple)
            train_dataset = tf.data.Dataset.from_tensor_slices(train_tuple)

            fold_splits.append(((train_dataset, validation_dataset), validation['rating']))

        return fold_splits

    def cross_validate(self, tol=1e-3, stop_iter=3):
        """ Run cross-validation. """
        folds = self._prepare_folds()
        param_mesh = np.meshgrid(
            *(np.array(self.param_dict[k]) for k in self.params),
            indexing='ij'
        )
        self.mesh_dict = {self.params[i]: param_mesh[i].flatten() for i in range(len(param_mesh))}
        scores = []
        best = 1e6
        for i in tqdm(range(len(self.mesh_dict[self.params[0]]))):
            params = {k: v[i] for k, v in self.mesh_dict.items()}
            kwargs = {**self.recommender_kwargs, **params}
            fold_scores = []
            for (train, validation), rating in folds:
                model = self.recommender_class(
                    dataset=train,
                    num_users=self.num_users,
                    num_items=self.num_items,
                    validation_set=validation,
                    optimizer=self.optimizer,
                    **kwargs
                )
                model.train(
                    n_iter=self.n_iter,
                    verbose=False,
                    tol=tol,
                    stop_iter=stop_iter
                )
                preds = model.predict(validation)
                fold_scores.append(self.metric(rating.values, preds.numpy()))
            avg_score = np.mean(fold_scores)
            if avg_score < best:
                best = avg_score
                self.best_model = params
            scores.append(avg_score)
        self.mesh_dict["score"] = scores

class MLBaseline:
    """A simple python class for creating baseline model. 
    
    It fills in the missing values in user-item matrix by using either mean or median values. """
    
    def __init__(self, data):
        """
        :param data: dataset to create baseline model for benchmarking
        :type data: MLDataManager
        """
        
        if data.train is None:
            data.train_test_split()
            
        self.data = data
            
        self.mean = self.data.train[['movieId', 'rating']] \
                        .groupby('movieId').mean().reset_index()
        self.median = self.data.train[['movieId', 'rating']] \
                          .groupby('movieId').median().reset_index()
                
    def predict_mean(self):
        out = self.data.test.join(
            self.mean,
            on='movieId',
            rsuffix='_predicted'
        )[['userId', 'movieId', 'rating', 'rating_predicted']]
        return out

    def predict_median(self):
        out = self.data.test.join(
            self.median,
            on='movieId',
            rsuffix='_predicted'
        )[['userId', 'movieId', 'rating', 'rating_predicted']]
        return out
    
class MLBaseline:
    """A simple python class for creating baseline model. 
    
    It fills in the missing values in user-item matrix by using either mean or median values. """
    
    def __init__(self, data):
        """
        :param data: dataset to create baseline model for benchmarking
        :type data: MLDataManager
        """
        
        if data.train is None:
            data.train_test_split()
            
        self.data = data
            
        self.mean = self.data.train[['movieId', 'rating']] \
                        .groupby('movieId').mean().reset_index()
        self.median = self.data.train[['movieId', 'rating']] \
                          .groupby('movieId').median().reset_index()
                
    def predict_mean(self):
        out = self.data.test.join(
            self.mean,
            on='movieId',
            rsuffix='_predicted'
        )[['userId', 'movieId', 'rating', 'rating_predicted']]
        return out

    def predict_median(self):
        out = self.data.test.join(
            self.median,
            on='movieId',
            rsuffix='_predicted'
        )[['userId', 'movieId', 'rating', 'rating_predicted']]
        return out
    
class Evaluate:
    """A simple python class for evaluating the performance of model predictions. 
    
    Compare model predictions to base model of choice by using kendall's rank correlation or 
    mean absolute error. """
    
    def __init__(self, data, base):
        """
        
        :param data: dataset to create baseline model for benchmarking
        :type data: MLDataManager
        :param base: denotes to use mean or median rating for the baseline model
        :type base: string
        """

        benchmark = MLBaseline(data)
        
        if "mean" in base:
            predicted_ratings = benchmark.predict_mean()
        elif "median" in base:
            predicted_ratings = benchmark.predict_median()
        else:
            "Please enter a valid base function"
         
        self.combined_ratings = predicted_ratings
         
    def kendall_tau(self):
        return(kendalltau(self.combined_ratings["rating"], self.combined_ratings["rating_predicted"]))
        
    def mean_abs_error(self):
        return(mean_absolute_error(self.combined_ratings["rating"], self.combined_ratings["rating_predicted"]))        