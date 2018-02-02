# vim: set fileencoding=utf-8

# Code based loosely on https://github.com/hiwonjoon/cycle-gan-tf/, heavily on papers
# https://arxiv.org/pdf/1704.00028.pdf, https://arxiv.org/pdf/1703.06490.pdf,
# https://arxiv.org/pdf/1703.10593.pdf

from __future__ import print_function, division
from functools import reduce
import tensorflow as tf, numpy as np, pandas as pd
import ml_toolkit.tensorflow_constructions as tfc, ml_toolkit.pandas_constructions as pdc
import time, math, os, pickle

from .utils import *
from .benchmarks_printer import BenchmarkPrinter, ITER_HEADER, TIME_HEADER, STEP_HEADER

# TODO(mmd): Make this not necessary so these can be scoped within the class as appropriate!
# TODO(mmd): I have to make sure these names won't hit names I assign manually right now, which is stupid.
TRANSLATOR_ROOT_SCOPE = 'TRANSLATOR_WGAN_translator'
CRITIC_ROOT_SCOPE     = 'TRANSLATOR_WGAN_critic'

CRITIC_ADVERSARIAL     = 'critic_adversarial_loss'
CRITIC_L2_LOSS         = 'critic_L2_loss'
CRITIC_L1_LOSS         = 'critic_L1_loss'
GRADIENT_PENALTY       = 'gradient_loss'
CRITIC_OVERALL         = 'critic'
CYCLE_PENALTY          = 'cycle_loss'
TRANSLATOR_ADVERSARIAL = 'translator_adversarial_loss'
TRANSLATOR_L2_LOSS     = 'translator_L2_loss'
TRANSLATOR_L1_LOSS     = 'translator_L1_loss'
TRANSLATOR_OVERALL     = 'translator'

PREDICTIVE_EUC_LOSS_S = 'predictive_euc_loss_S'
PREDICTIVE_MAN_LOSS_S = 'predictive_man_loss_S'
PREDICTIVE_COS_LOSS_S = 'predictive_cos_loss_S'
PREDICTIVE_EUC_LOSS_T = 'predictive_euc_loss_T'
PREDICTIVE_MAN_LOSS_T = 'predictive_man_loss_T'
PREDICTIVE_COS_LOSS_T = 'predictive_cos_loss_T'

ITERATION_TIME           = 'iteration_time'
TRAINING_STAGE           = 'training_stage'
CRITIC_EPOCHS            = 'critic_epochs'
TRANSLATOR_EPOCHS        = 'translator_epochs'
GLOBAL_EPOCHS            = 'global_epochs'
TRANSLATOR_LEARNING_RATE = 'translator_learning_rate'
CRITIC_LEARNING_RATE     = 'critic_learning_rate'

DEV_SIZE = 0.15

MIN_GLOBAL_EPOCHS = 20
# How many global epochs to run at the very least (b/c early the translator may fool the critic to an extreme
# extent which yields erroneously low losses).

# Benchmark Printing:
BENCHMARK_PARAMS = {
    PREDICTIVE_EUC_LOSS_T:  ('Euc. Pred. Loss (T)',         '{:.2e}', '1.34e10'),
    PREDICTIVE_MAN_LOSS_T:  ('Man. Pred. Loss (T)',         '{:.2e}', '1.34e10'),
    PREDICTIVE_COS_LOSS_T:  ('Cos. Pred. Loss (T)',         '{:.2e}', '1.34e10'),
    PREDICTIVE_EUC_LOSS_S:  ('Euc. Pred. Loss (S)',         '{:.2e}', '1.34e10'),
    PREDICTIVE_MAN_LOSS_S:  ('Man. Pred. Loss (S)',         '{:.2e}', '1.34e10'),
    PREDICTIVE_COS_LOSS_S:  ('Cos. Pred. Loss (S)',         '{:.2e}', '1.34e10'),
    GRADIENT_PENALTY:       ('Gradient Loss',               '{:.2e}', '7e13'),
    CRITIC_ADVERSARIAL:     ('Critic Adversarial Loss',     '{:.2e}', '1.34e10'),
    CRITIC_L2_LOSS:         ('Critic L2 Loss',              '{:.2e}', '1.34e10'),
    CRITIC_L1_LOSS:         ('Critic L1 Loss',              '{:.2e}', '1.34e10'),
    CRITIC_OVERALL:         ('Critic Loss',                 '{:.2e}', '7e13'),
    CYCLE_PENALTY:          ('Cycle Loss',                  '{:.2e}', '7e13'),
    TRANSLATOR_ADVERSARIAL: ('Translator Adversarial Loss', '{:.2e}', '1.34e10'),
    TRANSLATOR_L2_LOSS:     ('Translator L2 Loss',          '{:.2e}', '7e10'),
    TRANSLATOR_L1_LOSS:     ('Translator L1 Loss',          '{:.2e}', '7e10'),
    TRANSLATOR_OVERALL:     ('Translator Loss',             '{:.2e}', '7e13'),
}

CRITIC_STEP_BENCHMARKS = set([CRITIC_L2_LOSS, CRITIC_L1_LOSS, CRITIC_LEARNING_RATE])
TRANSLATOR_STEP_BENCHMARKS = set([
    PREDICTIVE_EUC_LOSS_S,
    PREDICTIVE_MAN_LOSS_S,
    PREDICTIVE_COS_LOSS_S,
    PREDICTIVE_EUC_LOSS_T,
    PREDICTIVE_MAN_LOSS_T,
    PREDICTIVE_COS_LOSS_T,
    TRANSLATOR_L2_LOSS,
    TRANSLATOR_L1_LOSS,
    CYCLE_PENALTY,
    TRANSLATOR_LEARNING_RATE,
])
GLOBAL_STEP_BENCHMARKS = set([
    CRITIC_ADVERSARIAL,
    GRADIENT_PENALTY,
    CRITIC_OVERALL,
    TRANSLATOR_ADVERSARIAL,
    TRANSLATOR_OVERALL,
    ITERATION_TIME,
    TRAINING_STAGE,
    CRITIC_EPOCHS,
    TRANSLATOR_EPOCHS,
    GLOBAL_EPOCHS,
])

#  Adam Optimizer Constants
#  Defaults taken from https://arxiv.org/pdf/1704.00028.pdf
BETA_1 = 0.5
BETA_2 = 0.9

#  Loss & Training Constants
GRADIENT_LOSS_MULTIPLIERS = [5000.0]
CYCLE_LOSS_MULTIPLIERS = [10.0]
EUC_DIST_T_LOSS_MULTIPLIERS = [0.0]
MAN_DIST_T_LOSS_MULTIPLIERS = [0.0]
COS_DIST_T_LOSS_MULTIPLIERS = [0.0]
EUC_DIST_S_LOSS_MULTIPLIERS = [0.0]
MAN_DIST_S_LOSS_MULTIPLIERS = [0.0]
COS_DIST_S_LOSS_MULTIPLIERS = [0.0]
ADVERSARIAL_LOSS_MULTIPLIERS = [1.0]
MAX_GLOBAL_EPOCHS = [10000]
MAX_CRITIC_EPOCHS = [5]
MAX_TRANSLATOR_EPOCHS = [1]
MAX_GLOBAL_PATIENCES = [10]
MAX_CRITIC_PATIENCES = [2]
MAX_TRANSLATOR_PATIENCES = [5]

# TODO(mmd): Experiment with increasing num_hidden_layers.
NUM_HIDDEN_LAYERS = 1 # = 2 layers total (Linear + activation + Linear). More does not seem to do better.
BATCH_SIZE = 15

def TRANSLATOR_DEFAULT_LEARNING_RATE(s): return tf.train.piecewise_constant(s, [15000], [1e-4, 1e-5])
def CRITIC_DEFAULT_LEARNING_RATE(s): return tf.train.piecewise_constant(s, [15000], [1e-4, 1e-5])

def _get_longest_length(*args): return max([len(x) for x in args])
def _any_not_none(*xs): return reduce(lambda x, y: x or y, map(lambda x: x is not None, xs))
class TranslatorWGAN(object):
    """
    Epochs, loss multipliers, patiences are lists to represent different training stages (so we can train one
    critic heavy stage with an unsupervised loss to optimality, and then a translator heavy stage with a
    predictive loss more fully as a second stage of trianing)
    """
    def __init__(
        self,
        source_paired_df                       = None,
        target_paired_df                       = None,
        source_unpaired_df                     = None,
        target_unpaired_df                     = None,
        side_info_paired_df                    = None,
        side_info_source_unpaired_df           = None,
        side_info_target_unpaired_df           = None,
        config                                 = tf.ConfigProto(),
        gradient_loss_multipliers              = GRADIENT_LOSS_MULTIPLIERS,
        cycle_loss_multipliers                 = CYCLE_LOSS_MULTIPLIERS,
        euc_dist_T_loss_multipliers            = EUC_DIST_T_LOSS_MULTIPLIERS,
        man_dist_T_loss_multipliers            = MAN_DIST_T_LOSS_MULTIPLIERS,
        cos_dist_T_loss_multipliers            = COS_DIST_T_LOSS_MULTIPLIERS,
        euc_dist_S_loss_multipliers            = EUC_DIST_S_LOSS_MULTIPLIERS,
        man_dist_S_loss_multipliers            = MAN_DIST_S_LOSS_MULTIPLIERS,
        cos_dist_S_loss_multipliers            = COS_DIST_S_LOSS_MULTIPLIERS,
        adversarial_loss_multipliers           = ADVERSARIAL_LOSS_MULTIPLIERS,
        max_global_epochs                      = MAX_GLOBAL_EPOCHS,
        max_critic_epochs                      = MAX_CRITIC_EPOCHS,
        max_translator_epochs                  = MAX_TRANSLATOR_EPOCHS,
        max_global_patiences                   = MAX_GLOBAL_PATIENCES,
        max_critic_patiences                   = MAX_CRITIC_PATIENCES,
        max_translator_patiences               = MAX_TRANSLATOR_PATIENCES,
        translator_learning_rate               = TRANSLATOR_DEFAULT_LEARNING_RATE,
        critic_learning_rate                   = CRITIC_DEFAULT_LEARNING_RATE,
        num_hidden_layers                      = NUM_HIDDEN_LAYERS,
        num_translator_hidden_layers           = None,
        num_critic_hidden_layers               = None,
        translator_hidden_dim                  = -1,
        critic_hidden_dim                      = -1,
        train_critic_dropout_keep_prob         = 1.0,
        train_translator_dropout_keep_prob     = 1.0,
        critic_L2_regularization_penalties     = [0],
        translator_L2_regularization_penalties = [0],
        critic_L1_regularization_penalties     = [0],
        translator_L1_regularization_penalties = [0],
        use_batch_norm                         = False,
        skip_connections                       = False,
        network_activation                     = tfc.leaky_relu,
        dim_change_strategy                    = 'jump',
        source_output_weights_init             = None,
        source_output_bias_init                = None,
        target_output_weights_init             = None,
        target_output_bias_init                = None,
        batch_size                             = BATCH_SIZE,
        beta1                                  = BETA_1,
        beta2                                  = BETA_2,
        wasserstein_critic                     = True,
        save_dir                               = '',
        train_suffix                           = '/train',
        dev_suffix                             = '/dev',
        dev_size                               = DEV_SIZE,
        model_ckpt_name                        = 'Translator_WGAN.ckpt',
        model_params_name                      = 'model_params',
        flush_secs                             = 15,
        random_state                           = None,
        min_global_epochs                      = MIN_GLOBAL_EPOCHS,
        # Must be a subset of the keys of BENCHMARK_PARAMS
        benchmarks = [CRITIC_OVERALL, TRANSLATOR_OVERALL, PREDICTIVE_EUC_LOSS_T, PREDICTIVE_EUC_LOSS_S,
            GRADIENT_PENALTY, CRITIC_L2_LOSS, CRITIC_L1_LOSS, CRITIC_ADVERSARIAL, CYCLE_PENALTY,
            TRANSLATOR_L2_LOSS, TRANSLATOR_L1_LOSS, TRANSLATOR_ADVERSARIAL],
        print_anything               = True,
        **kwargs # Because I am lazy and want to pass in the commandline args dict without filtering it.
    ):
        assert (source_paired_df is not None and target_paired_df is not None) or \
               (source_unpaired_df is not None and target_unpaired_df is not None), \
               "Must provide workable source and target dataframes"
        if source_paired_df is None:
            source_paired_df = pd.DataFrame(columns=source_unpaired_df.columns)
            target_paired_df = pd.DataFrame(columns=target_unpaired_df.columns)
        elif source_unpaired_df is None:
            source_unpaired_df = pd.DataFrame(columns=source_paired_df.columns)
        elif target_unpaired_df is None:
            target_unpaired_df = pd.DataFrame(columns=target_paired_df.columns)

        assert tfc.num_samples(source_paired_df) == tfc.num_samples(target_paired_df), \
               "Paired data must have the same number of samples."
        assert tfc.get_dim(source_paired_df) == tfc.get_dim(source_unpaired_df), \
               "Source data must be at the same dimensionality."
        assert tfc.get_dim(target_paired_df) == tfc.get_dim(target_unpaired_df), \
               "Target data must be at the same dimensionality."

        # TODO(mmd): Clean up the bit below:
        if _any_not_none(side_info_paired_df, side_info_source_unpaired_df, side_info_target_unpaired_df):
            cols = None
            if side_info_paired_df is not None: cols = side_info_paired_df.columns
            elif side_info_source_unpaired_df is not None: cols = side_info_source_unpaired_df.columns
            elif side_info_target_unpaired_df is not None: cols = side_info_target_unpaired_df.columns

            if side_info_paired_df is None:
                assert tfc.num_samples(source_paired_df) == 0, \
                    "No paired side_info provided, but paired data was provided."
                side_info_paired_df = pd.DataFrame(index=source_paired_df.index, columns=cols)
            else:
                assert tfc.num_samples(source_paired_df) == tfc.num_samples(side_info_paired_df), \
                    "Must provide side_info for all paired samples."
            if side_info_source_unpaired_df is None:
                assert tfc.num_samples(source_unpaired_df) == 0, \
                    "No source_unpaired side info was provided, but source_unpaired data was provided."
                side_info_source_unpaired_df = pd.DataFrame(index=source_unpaired_df.index, columns=cols)
            else:
                assert tfc.num_samples(source_unpaired_df) == tfc.num_samples(side_info_source_unpaired_df), \
                    "Must provide side_info for every source unpaired sample."
            if side_info_target_unpaired_df is None:
                assert tfc.num_samples(target_unpaired_df) == 0, \
                    "No target unpaired side info was provided, but unpaired target data was provided."
                side_info_target_unpaired_df = pd.DataFrame(index=target_unpaired_df.index, columns=cols)
            else:
                assert tfc.num_samples(target_unpaired_df) == tfc.num_samples(side_info_target_unpaired_df), \
                    "Must provide side_info for every target unpaired sample."
        else:
            side_info_paired_df = pd.DataFrame(index=source_paired_df.index)
            side_info_source_unpaired_df = pd.DataFrame(index=source_unpaired_df.index)
            side_info_target_unpaired_df = pd.DataFrame(index=target_unpaired_df.index)

        self.training_stages = _get_longest_length(
            max_global_epochs, max_critic_epochs, max_translator_epochs, max_global_patiences,
            max_critic_patiences, max_translator_patiences, gradient_loss_multipliers, cycle_loss_multipliers,
            euc_dist_T_loss_multipliers, cos_dist_T_loss_multipliers, euc_dist_S_loss_multipliers,
            cos_dist_S_loss_multipliers, adversarial_loss_multipliers, critic_L2_regularization_penalties,
            translator_L2_regularization_penalties, critic_L1_regularization_penalties,
            translator_L1_regularization_penalties

        )

        fill = lambda l: l + [l[-1]]*(self.training_stages - len(l))

        self.config                                 = config
        self.gradient_loss_multipliers              = fill(gradient_loss_multipliers)
        self.cycle_loss_multipliers                 = fill(cycle_loss_multipliers)
        self.euc_dist_T_loss_multipliers            = fill(euc_dist_T_loss_multipliers)
        self.man_dist_T_loss_multipliers            = fill(man_dist_T_loss_multipliers)
        self.cos_dist_T_loss_multipliers            = fill(cos_dist_T_loss_multipliers)
        self.euc_dist_S_loss_multipliers            = fill(euc_dist_S_loss_multipliers)
        self.man_dist_S_loss_multipliers            = fill(man_dist_S_loss_multipliers)
        self.cos_dist_S_loss_multipliers            = fill(cos_dist_S_loss_multipliers)
        self.adversarial_loss_multipliers           = fill(adversarial_loss_multipliers)
        self.max_global_epochs                      = fill(max_global_epochs)
        self.max_critic_epochs                      = fill(max_critic_epochs)
        self.max_translator_epochs                  = fill(max_translator_epochs)
        self.max_global_patiences                   = fill(max_global_patiences)
        self.max_critic_patiences                   = fill(max_critic_patiences)
        self.max_translator_patiences               = fill(max_translator_patiences)
        self.translator_learning_rate_fn            = translator_learning_rate
        self.critic_learning_rate_fn                = critic_learning_rate
        self.num_translator_hidden_layers           = num_hidden_layers if num_translator_hidden_layers is None else num_translator_hidden_layers
        self.num_critic_hidden_layers               = num_hidden_layers if num_critic_hidden_layers is None else num_critic_hidden_layers
        self.translator_hidden_dim                  = translator_hidden_dim
        self.critic_hidden_dim                      = critic_hidden_dim
        self.skip_connections                       = skip_connections
        self.network_activation                     = network_activation
        self.dim_change_strategy                    = dim_change_strategy
        self.source_output_weights_init             = source_output_weights_init
        self.source_output_bias_init                = source_output_bias_init
        self.target_output_weights_init             = target_output_weights_init
        self.target_output_bias_init                = target_output_bias_init
        self.train_critic_dropout_keep_prob         = train_critic_dropout_keep_prob
        self.train_translator_dropout_keep_prob     = train_translator_dropout_keep_prob
        self.critic_L2_regularization_penalties     = fill(critic_L2_regularization_penalties)
        self.translator_L2_regularization_penalties = fill(translator_L2_regularization_penalties)
        self.critic_L1_regularization_penalties     = fill(critic_L1_regularization_penalties)
        self.translator_L1_regularization_penalties = fill(translator_L1_regularization_penalties)
        self.use_batch_norm                         = use_batch_norm
        self.batch_size                             = batch_size
        self.beta1                                  = beta1
        self.beta2                                  = beta2
        self.wasserstein_critic                     = wasserstein_critic
        self.random_state                           = random_state
        self.save_dir                               = save_dir
        self.train_writer                           = None
        self.dev_writer                             = None
        self.train_suffix                           = train_suffix
        self.dev_suffix                             = dev_suffix
        self.flush_secs                             = flush_secs
        self.min_global_epochs                      = min_global_epochs
        self.model_ckpt_name                        = model_ckpt_name
        self.model_params_name                      = model_params_name
        self.benchmarks                             = set(benchmarks)
        self.printer                                = BenchmarkPrinter(params=BENCHMARK_PARAMS, columns=benchmarks)
        self.print_anything                         = print_anything
        self.tensor_names                           = {}
        self.dev_size                               = dev_size

        assert self.benchmarks.issubset(BENCHMARK_PARAMS.keys()), "Unknown benchmarks."

        model_params_file = os.path.join(self.save_dir, self.model_params_name) + '.pkl'
        if os.path.isfile(model_params_file):
            with open(model_params_file, 'rb') as f: self.__dict__ = pickle.load(f).copy()
            self.save_dir = save_dir

            # A temporary hack.
            if 'translator_learning_rate_fn' not in self.__dict__ and\
               'translator_learning_rate' in self.__dict__:
                self.translator_learning_rate_fn = self.translator_learning_rate
                self.critic_learning_rate_fn     = critic_learning_rate

        with open(model_params_file, 'wb') as f: pickle.dump(self.__dict__, f)

        self.curr_epoch = 0

        self.source_paired_df    = source_paired_df
        self.target_paired_df    = target_paired_df
        self.side_info_paired_df = side_info_paired_df

        self.source_unpaired_df           = source_unpaired_df
        self.target_unpaired_df           = target_unpaired_df
        self.side_info_source_unpaired_df = side_info_source_unpaired_df
        self.side_info_target_unpaired_df = side_info_target_unpaired_df

        self.source_dim    = tfc.get_dim(source_paired_df)
        self.target_dim    = tfc.get_dim(target_paired_df)
        self.side_info_dim = 0 if side_info_paired_df is None else tfc.get_dim(side_info_paired_df)
        # This looks a bit weird, but that's just because we can't set test_size=None with this splitting
        # function.
        (
            self.train_source_paired, self.dev_source_paired, self.train_target_paired,
            self.dev_target_paired, self.train_side_info_paired, self.dev_side_info_paired
        ) = pdc.split([self.source_paired_df, self.target_paired_df, self.side_info_paired_df],
            random_state=self.random_state, test_size=self.dev_size, dev_size=None)
        (
            self.train_source_unpaired, self.dev_source_unpaired, self.train_side_info_source_unpaired,
            self.dev_side_info_source_unpaired
        ) = pdc.split([self.source_unpaired_df, self.side_info_source_unpaired_df],
            random_state=self.random_state, test_size=self.dev_size, dev_size=None)
        (
            self.train_target_unpaired, self.dev_target_unpaired, self.train_side_info_target_unpaired,
            self.dev_side_info_target_unpaired
        ) = pdc.split([self.target_unpaired_df, self.side_info_target_unpaired_df],
            random_state=self.random_state, test_size=self.dev_size, dev_size=None)

        self.num_paired_train_samples = tfc.num_samples(self.train_source_paired)
        self.paired_train_indices     = list(range(self.num_paired_train_samples))
        self.num_paired_dev_samples   = tfc.num_samples(self.dev_source_paired)
        self.paired_dev_indices       = list(range(self.num_paired_dev_samples))

        self.num_unpaired_train_source_samples = tfc.num_samples(self.train_source_unpaired, static=False)
        self.unpaired_train_source_indices     = list(range(self.num_unpaired_train_source_samples))
        self.num_unpaired_dev_source_samples   = tfc.num_samples(self.dev_source_unpaired, static=False)
        self.unpaired_dev_source_indices       = list(range(self.num_unpaired_dev_source_samples))

        self.num_unpaired_train_target_samples = tfc.num_samples(self.train_target_unpaired, static=False)
        self.unpaired_train_target_indices     = list(range(self.num_unpaired_train_target_samples))
        self.num_unpaired_dev_target_samples   = tfc.num_samples(self.dev_target_unpaired, static=False)
        self.unpaired_dev_target_indices       = list(range(self.num_unpaired_dev_target_samples))

        self.maximal_train_samples = max(
            self.num_paired_train_samples,
            self.num_unpaired_train_source_samples,
            self.num_unpaired_train_target_samples,
        )
        self.maximal_dev_samples = max(
            self.num_paired_dev_samples,
            self.num_unpaired_dev_source_samples,
            self.num_unpaired_dev_target_samples,
        )

        self.graph_is_built = False
        self.build_graph()
        assert self.graph_is_built, "Graph should be built!"

        self.session_is_started = False
        self.start_session()
        assert self.session_is_started, "Session should be started!"

    def predict_target(self, source, side_info, target):
        if side_info is None: side_info = pd.DataFrame(index=source.index)
        predicted_source, true_target = self.sess.run(
            [self.F_source_paired, self.target_paired],
            feed_dict={
                self.source_paired: source,
                self.target_paired: target,
                self.side_info_paired: side_info,
                self.translator_dropout_keep_prob: 1.0,
                self.critic_dropout_keep_prob: 1.0,
                self.training: False,
            }
        )
        return (
            pd.DataFrame(predicted_source, index=source.index, columns=target.columns),
            pd.DataFrame(true_target, index=target.index, columns=target.columns)
        )

    def add_tensor(self, attr_name, tensor):
        """
        Sets self.attr_name = tensor.
        Also adds {attr_name: tensor.name} to the mapping of field names to tensor names to be restored in a loaded model.
        :param str attr_name: name to use for the attribute of self
        :param tensor: a tensorflow Tensor; tensor's name cannot already be used in this model
        :returns: None
        """
        #assert tensor.name not in self.tensor_names.values(),\
        #    "Trying to overwrite existing tensor %s!" % tensor.name
        self.tensor_names.update({attr_name: tensor.name})
        self.__setattr__(attr_name, tensor)

    def build_graph(self):
        self.graph = tf.Graph()
        self.build_model()
        self.set_losses()
        self.set_train_ops()
        with self.graph.as_default():
            self.init_op = tf.global_variables_initializer()

        self.graph_is_built = True

    def translate(self, source, target, side_info, out_domain='target'):
        assert out_domain in ['source', 'target'], 'Can only translate from source <-> target!'

        out_weights_init = self.target_output_weights_init
        out_bias_init = self.target_output_bias_init
        if out_domain == 'source':
            out_weights_init = self.source_output_weights_init
            out_bias_init = self.source_output_bias_init

        if self.num_translator_hidden_layers == 0:
            return tfc.linear(
                source if side_info is None else tf.concat([source, side_info], axis=1),
                tfc.get_dim(target),
                'translator_layer',
                weights_initializer = out_weights_init,
                bias_initializer = out_bias_init,
            )
        return tfc.feedforward(
            source if side_info is None else tf.concat([source, side_info], axis=1),
            tfc.get_dim(target),
            hidden_layers             = self.num_translator_hidden_layers,
            hidden_dim                = self.translator_hidden_dim,
            activation                = self.network_activation,
            skip_connections          = self.skip_connections,
            dim_change                = self.dim_change_strategy,
            dropout_keep_prob         = self.translator_dropout_keep_prob,
            batch_normalization       = self.use_batch_norm,
            training                  = self.training,
            output_layer_weights_init = out_weights_init,
            output_layer_bias_init    = out_bias_init,
        )

    def critic(self, sample, side_info):
        return tfc.feedforward(
            sample if side_info is None else tf.concat([sample, side_info], axis=1),
            1,
            hidden_layers       = self.num_critic_hidden_layers,
            hidden_dim          = self.critic_hidden_dim,
            activation          = self.network_activation,
            skip_connections    = self.skip_connections,
            dim_change          = self.dim_change_strategy,
            dropout_keep_prob   = self.critic_dropout_keep_prob,
            batch_normalization = False, # With a wasserstein critic, you cannot use batch normalization.
            training            = self.training,
        )

    def build_model(self):
        with self.graph.as_default():
            if self.random_state is not None: tf.set_random_seed(self.random_state)
            # State placeholders:
            self.add_tensor('critic_dropout_keep_prob',
                    tf.placeholder(tf.float32, name='critic_dropout_keep_prob'))
            self.add_tensor('translator_dropout_keep_prob',
                    tf.placeholder(tf.float32, name='translator_dropout_keep_prob'))
            self.add_tensor('training', tf.placeholder(tf.bool, name='training'))

            # Data placeholders:
            self.add_tensor('source_paired',
                tf.placeholder(tf.float32, shape=(None, self.source_dim), name='source_paired'))
            self.add_tensor('source_unpaired',
                tf.placeholder(tf.float32, shape=(None, self.source_dim), name='source_unpaired'))
            self.add_tensor('target_paired',
                tf.placeholder(tf.float32, shape=(None, self.target_dim), name='target_paired'))
            self.add_tensor('target_unpaired',
                tf.placeholder(tf.float32, shape=(None, self.target_dim), name='target_unpaired'))

            # Side info placeholders:
            self.add_tensor('side_info_paired',
                tf.placeholder(tf.float32, shape=(None,self.side_info_dim), name='side_info_paired'))
            self.add_tensor('side_info_source_unpaired',
                tf.placeholder(tf.float32, shape=(None,self.side_info_dim), name='side_info_source_unpaired'))
            self.add_tensor('side_info_target_unpaired',
                tf.placeholder(tf.float32, shape=(None,self.side_info_dim), name='side_info_target_unpaired'))

            # Concatenated data sources:
            self.add_tensor('source',
                tf.concat([self.source_paired, self.source_unpaired], axis=0))
            self.add_tensor('target',
                tf.concat([self.target_paired, self.target_unpaired], axis=0))
            self.add_tensor('side_info_source',
                tf.concat([self.side_info_paired, self.side_info_source_unpaired], axis=0))
            self.add_tensor('side_info_target',
                tf.concat([self.side_info_paired, self.side_info_target_unpaired], axis=0))

            # Epoch Counter
            self.add_tensor('critic_step', tfc.step_variable(name='critic_step'))
            self.add_tensor('translator_step', tfc.step_variable(name='translator_step'))
            self.add_tensor('global_step', self.critic_step + self.translator_step)

            # Learning Rates
            self.add_tensor('translator_learning_rate',
                    self.translator_learning_rate_fn(self.translator_step))
            self.add_tensor('critic_learning_rate', self.critic_learning_rate_fn(self.critic_step))

            # Translators:
            with tf.variable_scope(TRANSLATOR_ROOT_SCOPE):
                # Let $F: S -> T$, $G: T -> S$.
                with tf.variable_scope('S_to_T') as F_scope:
                    # Read as F(source) \in T
                    self.add_tensor('F_source',
                        self.translate(self.source, self.target, self.side_info_source))

                    F_scope.reuse_variables()
                    self.add_tensor('F_source_paired',
                        self.translate(self.source_paired, self.target_paired, self.side_info_paired)
                    )
                with tf.variable_scope('T_to_S') as G_scope:
                    # Read as G(target) \in S
                    self.add_tensor('G_target',
                        self.translate(self.target, self.source, self.side_info_target))

                    G_scope.reuse_variables()
                    self.add_tensor('G_target_paired',
                        self.translate(self.target_paired, self.source_paired, self.side_info_paired)
                    )

                # For cycle loss
                # Here we will impose that $F \compose G \approx I$.
                with tf.variable_scope(G_scope, reuse=True):
                    self.add_tensor('G_F_source',
                        self.translate(self.F_source, self.source, self.side_info_source))
                with tf.variable_scope(F_scope, reuse=True):
                    self.add_tensor('F_G_target',
                        self.translate(self.G_target, self.target, self.side_info_target))

            # Critics:
            with tf.variable_scope(CRITIC_ROOT_SCOPE):
                with tf.variable_scope('S_critic') as S_critic_scope:
                    # So this is a bit weird. In some cases, we don't have unpaired targets, but we
                    # do have unpaired sources, so the number of samples differs between source and target, so
                    # we pad to equalize this issue.
                    if self.num_unpaired_train_source_samples == self.num_unpaired_train_target_samples:
                        S, G_T, SI_S, SI_T = self.source, self.G_target, self.side_info_source, self.side_info_target
                    else:
                        S, G_T, SI_S, SI_T = tfc.make_compatible(self.source, self.G_target,
                            self.side_info_source, self.side_info_target)

                    epsilon = tf.random_uniform(shape=[tfc.num_samples(S), 1], minval=0.0, maxval=1.0)
                    #test_stat = tf.random_uniform(shape=[tfc.num_samples(S), 1], minval=0.0, maxval=1.0)
                    self.add_tensor('source_mixed', epsilon * S + (1-epsilon) * G_T)
                    #self.add_tensor('side_info_mixed_S',
                    #    tf.where(tf.reshape(test_stat < epsilon, [-1]), SI_S, SI_T))
                    self.add_tensor('side_info_mixed_S', epsilon * SI_S + (1-epsilon) * SI_T)

                    self.add_tensor('C_source_real', self.critic(self.source, self.side_info_source))
                    S_critic_scope.reuse_variables()
                    self.add_tensor('C_source_gen', self.critic(self.G_target, self.side_info_target))
                    self.add_tensor('C_source_mixed', self.critic(self.source_mixed, self.side_info_mixed_S))

                with tf.variable_scope('T_critic') as T_critic_scope:
                    if self.num_unpaired_train_source_samples == self.num_unpaired_train_target_samples:
                        T, F_S, SI_S, SI_T = self.target, self.F_source, self.side_info_source, self.side_info_target
                    else:
                        T, F_S, SI_S, SI_T = tfc.make_compatible(self.target, self.F_source,
                            self.side_info_source, self.side_info_target)
                    epsilon = tf.random_uniform(shape=[tfc.num_samples(T), 1], minval=0.0, maxval=1.0)
                    #test_stat = tf.random_uniform(shape=[tfc.num_samples(T), 1], minval=0.0, maxval=1.0)
                    self.add_tensor('target_mixed', epsilon * T + (1-epsilon) * F_S)
                    #self.add_tensor('side_info_mixed_T',
                    #    tf.where(tf.reshape(test_stat < epsilon, [-1]), SI_T, SI_S))
                    self.add_tensor('side_info_mixed_T', SI_S)

                    self.add_tensor('C_target_real', self.critic(self.target, self.side_info_target))
                    T_critic_scope.reuse_variables()
                    self.add_tensor('C_target_gen', self.critic(self.F_source, self.side_info_source))
                    self.add_tensor('C_target_mixed', self.critic(self.target_mixed, self.side_info_mixed_T))

            self.critic_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=CRITIC_ROOT_SCOPE)
            self.translator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                scope=TRANSLATOR_ROOT_SCOPE)

            for v in self.translator_variables + self.critic_variables:
                tf.summary.histogram(v.name, v)

            self.add_tensor('weight_histograms', tf.summary.merge_all())

            # Network Losses:
            #   General Information:
            #     In the Wasserstein formulation, our goal is to find a translator $T$ and critic $C$ to
            #     realize the following minimax problem:
            #     \[\min_{T}\max_{C \in \mathcal C} \EV{\rv x}{C(x)} - \EV{T(\rv z)}{C(T(z))},\]
            #     where $\rv x$ is a random variable representing the space of real samples, and $\rv z$ is
            #     the input to the translator $T$ within the GAN, with the constraint that $\mathcal C$
            #     contains only 1-Lipschitz functions. This 1-Lipschitz constraint is realized via a penalty
            #     in our final loss function on the gradient of the critic function itself.
            #
            #     In the standard GAN framework, we realize a different equation, where the critic is a true
            #     *discriminator*, yielding the probability that a given sample is real or fake:
            #     \[\min_{T}\max_{D} \EV{\rv x}{\log D(x)} - \EV{T(\rv z)}{\log (1 - D(T(z)))},\]
            #     where we note that we no longer have the 1-Lipschitz constraint.
            #
            #     In the BEGAN framework, we realize our discriminator as the reconstruction loss of an
            #     autoencoder. Here, we have the game:
            #     \[\min_{T}\max_{R \in \mathcal R(T)} \EV{\rv x}{R(x)} - \EV{T(\rv z)}{R(T(z))},\]
            #     where $R$ is the reconstruction error of an autoencoder, and $\mathcal R(T)$ is the set of all
            #     autoencoders that obey the following $\gamma$-parametrized _boundary equilibrium condition_
            #     given by
            #     \[\mathcal R(T) = \{R \text{ autoencoder reconstruction losses} |
            #                           \EV{\rv z}{R(T(z))} = \gamma \EV{\rv x}{R(x)}\}.\]
            #
            #     In the case of a cyclic translation process, we also include a cycle loss component that
            #     penalizes the L1 distance of any given point in either space and its image under the
            #     appropriate composed mapping.
            #
            #     We also include some predictive, distance-based losses, in order to examine and train on a
            #     true predictive loss, when paired samples are available.

            # Wasserstein Critic Loss Components:
            #   Gradient Loss:
            self.add_tensor(
                'gradient_loss',
                tf.add(tf.reduce_mean(
                    (tf.norm(tf.gradients(self.C_source_mixed, self.source_mixed)[0], axis=1) - 1.0)**2
                ), tf.reduce_mean(
                    (tf.norm(tf.gradients(self.C_target_mixed, self.target_mixed)[0], axis=1) - 1.0)**2
                )),
            )
            #   Wasserstein Adversarial Loss Component:
            self.add_tensor(
                'critic_adversarial_loss',
                tf.add_n((
                    -tf.reduce_mean(self.C_source_real), tf.reduce_mean(self.C_source_gen),
                    -tf.reduce_mean(self.C_target_real), tf.reduce_mean(self.C_target_gen),
                )),
            )

            # L2 Regularization
            self.add_tensor(
                'critic_L2_loss',
                tf.reduce_mean([tf.nn.l2_loss(var) for var in self.critic_variables])
            )
            self.add_tensor(
                'translator_L2_loss',
                tf.reduce_mean([tf.nn.l2_loss(var) for var in self.translator_variables])
            )
            self.add_tensor(
                'critic_L1_loss',
                tf.reduce_mean([tf.reduce_sum(tf.abs(var)) for var in self.critic_variables])
            )
            self.add_tensor(
                'translator_L1_loss',
                tf.reduce_mean([tf.reduce_sum(tf.abs(var)) for var in self.translator_variables])
            )

            # Translator Loss Components:
            #   Cycle Loss:
            self.add_tensor(
                'cycle_loss',
                tfc.dist(self.source, self.G_F_source) + tfc.dist(self.target, self.F_G_target)
            )
            #   Wasserstein Adversarial Loss Component:
            self.add_tensor('translator_adversarial_loss', -self.critic_adversarial_loss)
            #   Distances:
            #   TODO(mmd): This is a bit garbage, because we actually use these as attributes later (e.g.
            #   self.euc_dist_S_loss, etc.)
            for dist in tfc.DISTANCES:
                self.add_tensor('%s_dist_S_loss' % dist,
                    tfc.dist(self.source_paired, self.G_target_paired, ord=dist))
                self.add_tensor('%s_dist_T_loss' % dist,
                    tfc.dist(self.target_paired, self.F_source_paired, ord=dist))

    def set_losses(self):
        self.loss_ops = []
        self.loss_op_names = []
        for training_stage in range(self.training_stages):
            self.add_losses(training_stage)
            self.loss_op_names += [{key: tensor.name for key,tensor in self.loss_ops[training_stage].items()}]

    def add_losses(self, stg):
        with self.graph.as_default(), tf.variable_scope("stage_%d" % stg):
            translator_losses = [
                (self.adversarial_loss_multipliers[stg],           self.translator_adversarial_loss),
                (self.cycle_loss_multipliers[stg],                 self.cycle_loss),
                (self.euc_dist_T_loss_multipliers[stg],            self.euc_dist_T_loss),
                (self.euc_dist_S_loss_multipliers[stg],            self.euc_dist_S_loss),
                (self.man_dist_T_loss_multipliers[stg],            self.man_dist_T_loss),
                (self.man_dist_S_loss_multipliers[stg],            self.man_dist_S_loss),
                (self.cos_dist_T_loss_multipliers[stg],            self.cos_dist_T_loss),
                (self.cos_dist_S_loss_multipliers[stg],            self.cos_dist_S_loss),
                (self.translator_L2_regularization_penalties[stg], self.translator_L2_loss),
                (self.translator_L1_regularization_penalties[stg], self.translator_L1_loss),
            ]
            self.add_tensor('translator_loss', tf.add_n([m * l for m, l in translator_losses if m > 0]))
            critic_losses = [
                (1,                                            self.critic_adversarial_loss),
                (self.gradient_loss_multipliers[stg],          self.gradient_loss),
                (self.critic_L2_regularization_penalties[stg], self.critic_L2_loss),
                (self.critic_L1_regularization_penalties[stg], self.critic_L1_loss),
            ]
            self.add_tensor('critic_loss', tf.add_n([m * l for m, l in critic_losses if m > 0]))

        self.loss_ops.append({
            GRADIENT_PENALTY: self.gradient_loss,
            CRITIC_L2_LOSS: self.critic_L2_loss,
            CRITIC_L1_LOSS: self.critic_L1_loss,
            CRITIC_ADVERSARIAL: self.critic_adversarial_loss,
            CRITIC_OVERALL: self.critic_loss,
            CYCLE_PENALTY: self.cycle_loss,
            TRANSLATOR_L1_LOSS: self.translator_L1_loss,
            TRANSLATOR_L2_LOSS: self.translator_L2_loss,
            TRANSLATOR_OVERALL: self.translator_loss,
            TRANSLATOR_ADVERSARIAL: self.translator_adversarial_loss,
            PREDICTIVE_EUC_LOSS_S: self.euc_dist_S_loss,
            PREDICTIVE_MAN_LOSS_S: self.man_dist_S_loss,
            PREDICTIVE_COS_LOSS_S: self.cos_dist_S_loss,
            PREDICTIVE_EUC_LOSS_T: self.euc_dist_T_loss,
            PREDICTIVE_MAN_LOSS_T: self.man_dist_T_loss,
            PREDICTIVE_COS_LOSS_T: self.cos_dist_T_loss,
        })

    def set_train_ops(self):
        self.train_ops = []
        self.train_op_names = []
        for training_stage in range(self.training_stages):
            self.add_train_ops(training_stage)
            self.train_op_names.append(
                {key: tensor.name for key, tensor in self.train_ops[training_stage].items()})

    def add_train_ops(self, training_stage):
        # TODO(mmd): X-val over parameters here.
        with self.graph.as_default(),\
             tf.variable_scope(str(training_stage)),\
             tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): # Might not work with >1 stg

            train_translator_op = tf.train.AdamOptimizer(
                learning_rate=self.translator_learning_rate, beta1=self.beta1, beta2=self.beta2
            ).minimize(
                self.loss_ops[training_stage][TRANSLATOR_OVERALL],
                var_list=self.translator_variables,
                global_step=self.translator_step,
                name='train_trans_op'
            )

            train_critic_op = tf.train.AdamOptimizer(
                learning_rate=self.critic_learning_rate, beta1=self.beta1, beta2=self.beta2
            ).minimize(
                self.loss_ops[training_stage][CRITIC_OVERALL],
                var_list=self.critic_variables,
                global_step=self.critic_step,
                name='train_crit_op'
            )

            self.train_ops.append({
                CRITIC_OVERALL: train_critic_op,
                TRANSLATOR_OVERALL: train_translator_op,
            })

    def feed(self, batch_start=None, dev=False, train_phase=True):
        # TODO(mmd): there must be a better way to do this. This is pretty screwy.
        source_paired = self.dev_source_paired.values if dev else self.train_source_paired.values
        target_paired = self.dev_target_paired.values if dev else self.train_target_paired.values
        source_unpaired = self.dev_source_unpaired.values if dev else self.train_source_unpaired.values
        target_unpaired = self.dev_target_unpaired.values if dev else self.train_target_unpaired.values

        side_info_paired = self.dev_side_info_paired.values if dev else \
            self.train_side_info_paired.values
        side_info_source_unpaired = self.dev_side_info_source_unpaired.values if dev else \
            self.train_side_info_source_unpaired.values
        side_info_target_unpaired = self.dev_side_info_target_unpaired.values if dev else \
            self.train_side_info_target_unpaired.values

        if batch_start is not None:
            # TODO(mmd): We use one set of indices to ensure that all non-empty entries always contribute the
            # same number of (potentially resampled) entries, if possible. Batches will rotate around smaller
            # sets of indices as we progress in the epoch via the modular_slice helper in utils.
            batch_end = batch_start + self.batch_size
            paired_indices = self.paired_dev_indices if dev else self.paired_train_indices
            unpaired_source_indices = (
                self.unpaired_dev_source_indices if dev else self.unpaired_train_source_indices
            )
            unpaired_target_indices = (
                self.unpaired_dev_target_indices if dev else self.unpaired_train_target_indices
            )
            source_paired   = source_paired[modular_slice(paired_indices, batch_start, batch_end)]
            target_paired   = target_paired[modular_slice(paired_indices, batch_start, batch_end)]
            source_unpaired = source_unpaired[modular_slice(unpaired_source_indices, batch_start, batch_end)]
            target_unpaired = target_unpaired[modular_slice(unpaired_target_indices, batch_start, batch_end)]
            side_info_paired = side_info_paired[modular_slice(paired_indices, batch_start, batch_end)]
            side_info_source_unpaired = \
                side_info_source_unpaired[modular_slice(unpaired_source_indices, batch_start, batch_end)]
            side_info_target_unpaired = \
                side_info_target_unpaired[modular_slice(unpaired_target_indices, batch_start, batch_end)]

        return {
            self.source_paired: source_paired, self.target_paired: target_paired,
            self.source_unpaired: source_unpaired, self.target_unpaired: target_unpaired,
            self.side_info_paired: side_info_paired,
            self.side_info_source_unpaired: side_info_source_unpaired,
            self.side_info_target_unpaired: side_info_target_unpaired,
            self.translator_dropout_keep_prob: self.train_translator_dropout_keep_prob if train_phase else 1.0,
            self.critic_dropout_keep_prob: self.train_critic_dropout_keep_prob if train_phase else 1,
            self.training: train_phase
        }

    def start_session(self):
        """
        Starts a session with self.graph.
        If self.save_dir contains a previously trained model, then the graph from that run is loaded for
        further training/inference. If self.save_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """
        meta_graph_file_base = os.path.join(self.save_dir, self.model_ckpt_name)
        latest_checkpoint_file = tf.train.latest_checkpoint(self.save_dir)
        f = meta_graph_file_base
        if latest_checkpoint_file is not None:
            f = latest_checkpoint_file
        f += '.meta'
        if os.path.isfile(f): # load saved model
            self.sess = tf.Session(config=self.config)
            if self.print_anything: print("Loading graph from:", f)

            self.saver = tf.train.import_meta_graph(f)
            #try: self.saver = tf.train.import_meta_graph(f)
            #except Exception as e:
            #    print('\n\n\n\n\n\n')
            #    print('Existing Variables: ')
            #    print('\n'.join(map(lambda t: t.name, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))))
            #    print('Existing Operations: ')
            #    with open('out_blah.txt', mode='w') as f:
            #        f.write('\n'.join(map(lambda t: t.name, self.graph.get_operations())))
            #    print('tfld_0_drn_6_int_dopamine_siil_gender-age_anch_False_nthl_'
            #          '1_nchl_3_tl2_2_tl1_2_tdo_0.5_cl2_2_cl1_2_chd_-1_cdo_0.5_vfld_0_joint'
            #          '/TRANSLATOR_WGAN_translator/S_to_T/layer_0/weights' in map(lambda t: t.name, self.graph.get_operations()))

            #    print('\n\n\n\n\n\n')
            #    raise e

            if latest_checkpoint_file is not None:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
            else:
                self.saver.restore(self.sess, os.path.join(self.save_dir, self.model_ckpt_name))

            self.graph = tf.get_default_graph()

            # update self's fields
            for attr_name, tensor_name in self.tensor_names.items():
                try: self.__setattr__(attr_name, self.graph.get_tensor_by_name(tensor_name))
                except KeyError: continue

            # update indirect references
            for training_stage in range(self.training_stages):
                self.loss_ops[training_stage] = {}
                for key, name in self.loss_op_names[training_stage].items():
                    try: self.loss_ops[training_stage][key] = self.graph.get_tensor_by_name(name)
                    except KeyError: continue
                self.train_ops[training_stage] = {}
                for key, name in self.train_op_names[training_stage].items():
                    try: self.train_ops[training_stage][key] = self.graph.get_operation_by_name(name)
                    except KeyError: continue

            self.session_is_started = True
            with self.graph.as_default():
                self.train_writer = tf.summary.FileWriter(self.save_dir + self.train_suffix, self.sess.graph,
                                                          flush_secs=self.flush_secs)
                self.dev_writer = tf.summary.FileWriter(self.save_dir + self.dev_suffix, self.sess.graph,
                                                        flush_secs=self.flush_secs)
            return

        self.sess = tf.Session(config=self.config, graph=self.graph)
        self.sess.run(self.init_op)

        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=5)
            self.train_writer = tf.summary.FileWriter(self.save_dir + self.train_suffix, self.sess.graph,
                                                      flush_secs=self.flush_secs)
            self.dev_writer = tf.summary.FileWriter(self.save_dir + self.dev_suffix, self.sess.graph,
                                                    flush_secs=self.flush_secs)
        self.summarize_losses(0, time_delta=0, step_label='random_initialization')
        self.session_is_started = True

    def save(self):
        if self.save_dir == '': return
        assert self.session_is_started, "Session must be started to save."

        # TODO(mmd): Save multiple checkpoints, and one optimal model path.
        self.saver.save(self.sess, os.path.join(self.save_dir, self.model_ckpt_name),
            self.sess.run(self.global_step))

    def summarize_losses(self, training_stage, return_loss=None, time_delta=None, step_label=''):
        # TODO(mmd): DRY this up.
        train_losses_container = {key: 0 for key in self.benchmarks}
        dev_losses_container = {key: 0 for key in self.benchmarks}

        num_train_batches, num_dev_batches = 0, 0
        for batch_start in range(0, self.maximal_train_samples, self.batch_size):
            num_train_batches += 1
            sum_dict(
                train_losses_container,
                self.sess.run(
                    intersect(self.loss_ops[training_stage], self.benchmarks),
                    feed_dict=self.feed(batch_start=batch_start),
                ),
            )
        for batch_start in range(0, self.maximal_dev_samples, self.batch_size):
            num_dev_batches += 1
            sum_dict(
                dev_losses_container,
                self.sess.run(
                    intersect(self.loss_ops[training_stage], self.benchmarks),
                    feed_dict=self.feed(dev=True, batch_start=batch_start),
                ),
            )

        train_losses = div_dict(train_losses_container, num_train_batches)
        if self.dev_size > 0: dev_losses = div_dict(dev_losses_container, num_dev_batches)
        else: dev_losses = {key: np.NaN for key in self.benchmarks}

        if self.print_anything: self.print_benchmarks(dev_losses, train_losses, time_delta, step_label)
        if self.train_writer is None and self.dev_writer is None:
            if return_loss is None: return
            else: return dev_losses[return_loss] if self.dev_size > 0 else train_losses[return_loss]

        (
            critic_epochs, translator_epochs, global_epochs, global_step,
            translator_learning_rate, critic_learning_rate, weight_histograms,
        )= self.sess.run([
            self.critic_step, self.translator_step, self.global_step, self.global_step,
            self.translator_learning_rate, self.critic_learning_rate, self.weight_histograms,
        ])

        extra_benchmarks = {
            ITERATION_TIME: time_delta,
            TRAINING_STAGE: training_stage,
            CRITIC_EPOCHS: critic_epochs,
            TRANSLATOR_EPOCHS: translator_epochs,
            GLOBAL_EPOCHS: global_epochs,
            TRANSLATOR_LEARNING_RATE: translator_learning_rate,
            CRITIC_LEARNING_RATE: critic_learning_rate,
        }
        train_losses.update(extra_benchmarks)
        if self.dev_size > 0: dev_losses.update(extra_benchmarks)

        def smooth(val):
            if type(val) == np.float32: return float(val)
            elif type(val) == np.int32: return int(val)
            return val
        def summary(tag, val): return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=smooth(val))])
        for group, step in [
            (CRITIC_STEP_BENCHMARKS, critic_epochs),
            (TRANSLATOR_STEP_BENCHMARKS, translator_epochs),
            (GLOBAL_STEP_BENCHMARKS, global_step),
        ]:
            for bench in intersect(train_losses, group).keys():
                name = bench.replace('_', ' ').title()
                self.train_writer.add_summary(summary(name, train_losses[bench]), step)
                if self.dev_size > 0: self.dev_writer.add_summary(summary(name, dev_losses[bench]), step)
        self.train_writer.add_summary(weight_histograms, global_step)

        if return_loss is not None:
            return dev_losses[return_loss] if self.dev_size > 0 else train_losses[return_loss]

    def shuffle_indices(self):
        np.random.shuffle(self.paired_train_indices)
        np.random.shuffle(self.paired_dev_indices)
        np.random.shuffle(self.unpaired_train_source_indices)
        np.random.shuffle(self.unpaired_dev_source_indices)
        np.random.shuffle(self.unpaired_train_target_indices)
        np.random.shuffle(self.unpaired_dev_target_indices)

    def step(self, ops):
        # TODO(mmd): Maybe could do all in tf graph via producer queues and such?
        start_time = time.time()
        self.shuffle_indices()
        for batch_start in range(0, self.maximal_train_samples, self.batch_size):
            self.sess.run(ops, feed_dict=self.feed(batch_start=batch_start))
        return time.time() - start_time

    def train_till_stop(
        self, train_fn, max_epochs, max_patience,
        save            = False,
        best_loss_init  = float('inf'),
        min_epochs      = -1,
        start_epoch_cnt = 0,
    ):
        start_time = time.time()
        local_best_loss, patience = best_loss_init, max_patience
        for epoch in range(start_epoch_cnt, max_epochs):
            loss = train_fn()

            if math.isnan(loss): raise ValueError("Obtained NaN In Training!") # TODO(mmd): Do better.
            if epoch > min_epochs:
                if loss < local_best_loss:
                    local_best_loss, patience = loss, max_patience
                    if save: self.save()
                elif patience <= 1: return local_best_loss, time.time() - start_time
                else: patience -= 1
        return local_best_loss, time.time() - start_time

    def _step_critic(self, training_stage):
        step_time = self.step(self.train_ops[training_stage][CRITIC_OVERALL])
        #self.critic_epochs += 1
        return self.summarize_losses(training_stage, return_loss=CRITIC_OVERALL, time_delta=step_time,
            step_label=CRITIC_OVERALL)

    def _step_translator(self, training_stage):
        step_time = self.step(self.train_ops[training_stage][TRANSLATOR_OVERALL])
        #self.translator_epochs += 1
        return self.summarize_losses(training_stage, return_loss=TRANSLATOR_OVERALL,
            time_delta=step_time, step_label=TRANSLATOR_OVERALL)

    def _global_step(self, training_stage):
        # Train the critic:
        self.train_till_stop(lambda: self._step_critic(training_stage),
            self.max_critic_epochs[training_stage], self.max_critic_patiences[training_stage])
        # Train the translator and return:
        best_loss, _ = self.train_till_stop(lambda: self._step_translator(training_stage),
            self.max_translator_epochs[training_stage], self.max_translator_patiences[training_stage])
        #self.total_global_epochs += 1
        return best_loss

    def train_stage(self, training_stage):
        if self.print_anything: self.printer.print_benchnames()
        return self.train_till_stop(lambda: self._global_step(training_stage),
            self.max_global_epochs[training_stage], self.max_global_patiences[training_stage],
            save            = True,
            best_loss_init  = self.best_loss,
            min_epochs      = self.min_global_epochs,
            #start_epoch_cnt = self.sess.run(self.total_global_epochs),
        )

    def train(self):
        for stage in range(self.training_stages):
            if self.print_anything: print("\n\n\nTraining Stage %d\n\n\n" % stage)
            self.best_loss = float('inf')
            self.best_loss, delta_t = self.train_stage(stage)
            if self.print_anything: print(
                "Training Stage %d took %d seconds and attained loss %.2e" % (stage, delta_t, self.best_loss)
            )
        return self.best_loss, delta_t

    def print_benchmarks(self, dev_losses, train_losses, time_delta, step_label):
        global_step = self.sess.run(self.global_step)

        train_benchmarks = {
            ITER_HEADER: global_step, TIME_HEADER: time_delta, STEP_HEADER: 'train, %s' % step_label
        }
        train_benchmarks.update(train_losses)
        dev_benchmarks = {
            ITER_HEADER: global_step, TIME_HEADER: time_delta, STEP_HEADER: 'dev, %s' % step_label
        }
        dev_benchmarks.update(dev_losses)

        self.printer.print_data_groups([train_benchmarks, dev_benchmarks])
