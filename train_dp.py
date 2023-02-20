# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pyformat: disable

import functools
import os
import shutil
from typing import Callable
import json
import math
import jax
import time
import jax.numpy as jn
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf  # For data augmentation.
import tensorflow_datasets as tfds
from absl import app, flags
from objax.module import Module, ModuleList
from flax.training import lr_schedule
import objax
from objax.jaxboard import SummaryWriter, Summary
from objax.util import EasyDict
from objax.zoo import convnet, wide_resnet
from objax.zoo import resnet_v2
from objax.variable import TrainRef, StateVar, TrainVar, VarCollection
import jax.numpy as jnp
from dataset import DataSet
from flax.metrics import tensorboard
import sys
from copy import deepcopy
FLAGS = flags.FLAGS
CUDA_VISIBLE_DEVICES = '0'

summary_writer = tensorboard.SummaryWriter
def augment(x, shift: int, mirror=True):
    """
    Augmentation function used in training the model.
    """
    y = x['image']
    if mirror:
        y = tf.image.random_flip_left_right(y)
    y = tf.pad(y, [[shift] * 2, [shift] * 2, [0] * 2], mode='REFLECT')
    y = tf.image.random_crop(y, tf.shape(x['image']))
    return dict(image=y, label=x['label'])


class TrainLoop(objax.Module):
    """
    Training loop for general machine learning models.
    Based on the training loop from the objax CIFAR10 example code.
    """
    predict: Callable
    train_op: Callable
    get_sam_gradient: Callable

    def __init__(self, nclass: int, **kwargs):
        self.nclass = nclass
        self.params = EasyDict(kwargs)

    def train_step(self, summary: Summary, data: dict, progress: np.ndarray, lr_cos):

        # if FLAGS.fair == True:
        #     if progress >= .9500:
        #         inv_gray_images = self.convert2invGray(data)
        #         data['image'] = tf.convert_to_tensor(inv_gray_images)
        #     else:
        #         gray_images = self.convert2gray(data)
        #         data['image'] = tf.convert_to_tensor(gray_images)

        kvg = self.train_op(progress, data['image'].numpy(), data['label'].numpy(), lr_cos)
        for k, v in kvg.items():
            if jn.isnan(v):
                raise ValueError('NaN, try reducing learning rate', k)
            if summary is not None:
                summary.scalar(k, float(v))


    def get_cosine_schedule(self, num_epochs: int, learning_rate: float,
                            num_training_obs: int,
                            batch_size: int) -> Callable[[int], float]:

        steps_per_epoch = int(math.floor(num_training_obs / batch_size))
        learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(
            learning_rate, steps_per_epoch // jax.host_count(), num_epochs,
            warmup_length=0)
        return learning_rate_fn



    def train(self, num_train_epochs: int, train_size: int, train: DataSet, test: DataSet, logdir: str, save_steps=100,
              patience=None):
        """
        Completely standard training. Nothing interesting to see here.
        """
        checkpoint = objax.io.Checkpoint(logdir, keep_ckpts=20, makedir=True)
        start_epoch, last_ckpt = checkpoint.restore(self.vars())
        train_iter = iter(train)
        progress = np.zeros(jax.local_device_count(), 'f')  # for multi-GPU
        self.total_num_replicas = jax.device_count()
        best_acc = 0
        best_acc_epoch = -1
        batch_size = self.params.batch
        num_train_obs = 25000

        learning_rate_fn = self.get_cosine_schedule(num_train_epochs, FLAGS.lr, num_train_obs, batch_size)
        with SummaryWriter(os.path.join(logdir, 'tb')) as tensorboard:
            start = time.time()
            steps = 0
            for epoch in range(start_epoch, FLAGS.epochs):
                # Train
                summary = Summary()
                loop = range(0, train_size, self.params.batch)
                lr_cos = learning_rate_fn(epoch*batch_size)
                for step in loop:
                    # print(steps)
                    progress[:] = (step + (epoch * train_size)) / (num_train_epochs * train_size)
                    lr_cos = learning_rate_fn(steps)
                    self.train_step(summary, next(train_iter), progress, lr_cos)
                    steps += 1

                if FLAGS.dpsgd or FLAGS.dpsam:
                    dp_epsilon = objax.privacy.dpsgd.analyze_dp(
                        q=FLAGS.batch * FLAGS.grad_acc_steps / num_train_obs,
                        noise_multiplier=FLAGS.dp_sigma * np.sqrt(self.total_num_replicas * FLAGS.grad_acc_steps),
                        steps=int(steps / FLAGS.grad_acc_steps),
                        delta=FLAGS.dp_delta)
                    summary.scalar('privacy/epsilon', dp_epsilon)
                # Eval
                accuracy, total = 0, 0
                if epoch % FLAGS.eval_steps == 0 and test is not None:
                    for data in test:
                        total += data['image'].shape[0]

                        preds = np.argmax(self.predict(data['image'].numpy()), axis=1)
                        accuracy += (preds == data['label'].numpy()).sum()
                    accuracy /= total
                    summary.scalar('eval/accuracy', 100 * accuracy)
                    tensorboard.write(summary, step=(epoch + 1) * train_size)
                    print('Epoch %04d  Loss %.4f  Accuracy %.4f' % (epoch + 1, summary['losses/xe'](),
                                                                    summary['eval/accuracy']()))

                    if summary['eval/accuracy']() > best_acc:
                        best_acc = summary['eval/accuracy']()
                        best_acc_epoch = epoch
                    elif patience is not None and epoch > best_acc_epoch + patience:
                        print("early stopping!")
                        checkpoint.save(self.vars(), epoch + 1)
                        return


                else:
                    print('Epoch %04d  Loss %.4f  Accuracy --' % (epoch + 1, summary['losses/xe']()))


                if epoch % save_steps == save_steps - 1:
                    checkpoint.save(self.vars(), epoch + 1)
            end = time.time()
            print("time is: ", end - start)


# We inherit from the training loop and define predict and train_op.
class MemModule(TrainLoop):
    def __init__(self, model: Callable, nclass: int, mnist=False, dp=False, **kwargs):
        """
        Completely standard training. Nothing interesting to see here.
        """
        super().__init__(nclass, **kwargs)
        self.model = model(1 if mnist else 3, nclass)

        self.opt = objax.optimizer.SGD(self.model.vars())
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999, debias=True)

        model_c = model(1 if mnist else 3, nclass)
        self.opt_sam = objax.optimizer.SGD(model_c.vars())
        self.model_ema_sam = objax.optimizer.ExponentialMovingAverageModule(model_c, momentum=0.999, debias=True)

        @objax.Function.with_vars(self.model.vars())
        def loss(x, label):
            logit = self.model(x, training=True)
            loss_wd = 0.5 * sum((v.value ** 2).sum() for k, v in self.model.vars().items() if k.endswith('.w'))
            loss_xe = objax.functional.loss.cross_entropy_logits(logit, label).mean()
            return loss_xe + loss_wd * self.params.weight_decay, {'losses/xe': loss_xe, 'losses/wd': loss_wd}

        if dp:
            print("dpsgd will be used.")
            gv = objax.privacy.dpsgd.PrivateGradValues(
                loss,
                self.model.vars(),
                FLAGS.dp_sigma,
                FLAGS.dp_clip_norm,
                microbatch=1,
                batch_axis=(0, 0),
                use_norm_accumulation=True)
        else:
            gv = objax.GradValues(loss, self.model.vars())
        self.gv = gv

        @objax.Function.with_vars(self.vars())
        def train_op(progress, x, y, lr_loss):
            g, v = gv(x, y)
            lr = self.params.lr * jn.cos(progress * (7 * jn.pi) / (2 * 8))
            lr = lr * jn.clip(progress * 100, 0, 1)
            self.opt(lr, g)
            self.model_ema.update_ema()

            return {'monitors/lr': lr, **v[1]}

        self.predict = objax.Jit(objax.nn.Sequential([objax.ForceArgs(self.model_ema, training=False)]))

        self.train_op = objax.Jit(train_op)


class MemModule_sam(TrainLoop):
    def __init__(self, model: Callable, nclass: int, mnist=False, rho=0.05, eps=1e-12, **kwargs):
        """
        Completely standard training. Nothing interesting to see here.
        """
        super().__init__(nclass, **kwargs)
        self.model = model(1 if mnist else 3, nclass)

        self.opt = objax.optimizer.Momentum(self.model.vars())
        self.model_ema = objax.optimizer.ExponentialMovingAverageModule(self.model, momentum=0.999, debias=True)

        model_c = model(1 if mnist else 3, nclass)
        self.opt_sam = objax.optimizer.Momentum(model_c.vars())
        self.model_ema_sam = objax.optimizer.ExponentialMovingAverageModule(model_c, momentum=0.999, debias=True)

        # @objax.Function.with_vars(self.model.vars())
        def loss(x, label):
            logit = self.model(x, training=True)
            loss_wd = 0.5 * sum((v.value ** 2).sum() for k, v in self.model.vars().items() if k.endswith('.w'))
            loss_xe = objax.functional.loss.cross_entropy_logits(logit, label).mean()
            return loss_xe + loss_wd * self.params.weight_decay , {'losses/xe': loss_xe, 'losses/wd': loss_wd}

        def global_norm(updates) -> jnp.ndarray:
            """Returns the l2 norm of the input.

            Args:
              updates: A pytree of ndarrays representing the gradient.
            """
            return jnp.sqrt(
                sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))

        def clip_by_global_norm(updates):
            """Clips the gradient by global norm.

            Will have no effect if FLAGS.gradient_clipping is set to zero (no clipping).

            Args:
              updates: A pytree of numpy ndarray representing the gradient.

            Returns:
              The gradient clipped by global norm.
            """
            gradient_clipping = 5
            if gradient_clipping > 0:
                g_norm = global_norm(updates)
                trigger = g_norm < gradient_clipping
                updates = jax.tree_util.tree_map(
                    lambda t: jnp.where(trigger, t, (t / g_norm) * gradient_clipping),
                    updates)
            return updates

        if FLAGS.dpsgd:
            print("dpsgd will be used")
            gv = objax.privacy.dpsgd.PrivateGradValues(
                loss,
                self.model.vars(),
                FLAGS.dp_sigma,
                FLAGS.dp_clip_norm,
                microbatch=FLAGS.microbatch,
                batch_axis=(0, 0),
                use_norm_accumulation=True)

        elif FLAGS.dpsam:
            print("dpsam will be used")
            gv = objax.privacy.dpsam.PrivateGradValues(
                loss,
                self.model.vars(),
                FLAGS.dp_sigma,
                FLAGS.dp_clip_norm,
                microbatch=FLAGS.microbatch,
                batch_axis=(0, 0),
                use_norm_accumulation=False,
                rho=FLAGS.rho)
        else:
            gv = objax.GradValues(loss, self.model.vars())
        self.gv = gv

        # @objax.Function.with_vars(self.model.vars())
        def get_sam_gradient1(model, g, x, y):
            # gv = objax.GradValues(loss, model.vars())
            vc = self.model.vars()
            train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
            # gradient_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(g)]))
            # normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, g)
            # g = normalized_gradient
            norm_g = jnp.linalg.norm(jnp.array([jnp.linalg.norm(g_) for g_ in g]))
            # self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in self.train_vars)
            assert len(g) == len(train_vars), 'Expecting as many gradients as trainable variables'
            scale = rho / (norm_g + 1e-12)
            e_ws = []
            # for l, gr in zip(self.train_vars, normalized_gradient):
            #
            # train_vars = jax.tree_util.tree_map(lambda a, b: a + rho * b, train_vars, g)
                # l.value -= m.value
            for i in range(len(train_vars)):
                ew = g[i] * scale
                train_vars[i].assign(train_vars[i] + ew)
                e_ws.append(ew)

            # noised_gv = objax.GradValues(loss, vc)
            g1, v1 = gv(x, y)

            # train_vars_tmp = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))

            for i in range(len(train_vars)):
                train_vars[i].assign(train_vars[i] - e_ws[i])
            # for l, gr in zip(self.train_vars, normalized_gradient):
            #     l.value += rho * gr
            # self.model.vars = train_vars_tmp
            return g1, v1

        def get_asam_gradient2(model, g, progress):
            # not implemented yet correctly
            # gv = objax.GradValues(loss, model.vars())
            vc = model.vars()
            train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
            gradient_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(g)]))
            normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, g)
            # self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in train_vars)
            assert len(g) == len(train_vars), 'Expecting as many gradients as trainable variables'
            scale = rho / (gradient_norm + 1e-12)
            lr = self.params.lr * jn.cos(progress * (7 * jn.pi) / (2 * 8))
            lr = lr * jn.clip(progress * 100, 0, 1)
            self.opt_sam(lr + scale, normalized_gradient)
            self.model_ema_sam.update_ema()
            noised_gv = objax.GradValues(loss, vc)

            return noised_gv

        def get_sam_gradient2(model, g, progress):
            # gv = objax.GradValues(loss, model.vars())
            vc = model.vars()
            train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
            gradient_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(g)]))
            normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, g)
            # self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in train_vars)
            assert len(g) == len(train_vars), 'Expecting as many gradients as trainable variables'
            scale = rho / (gradient_norm + 1e-12)
            lr = self.params.lr * jn.cos(progress * (7 * jn.pi) / (2 * 8))
            lr = lr * jn.clip(progress * 100, 0, 1)
            self.opt_sam(lr + scale, normalized_gradient)
            self.model_ema_sam.update_ema()
            noised_gv = objax.GradValues(loss, vc)

            return noised_gv

        @objax.Function.with_vars(self.vars())
        def train_op(progress, x, y, lr_cos):
            g, v = self.gv(x, y)
            # g1 = objax.TrainVar(g)
            print("rho", rho)
            # print("rho", rho)
            if rho > 0 and not FLAGS.dpsam:
                print("SAM1 optimizer will be used")
                # noised_gv = get_sam_gradient2(self.model, g, progress)
                g, v = get_sam_gradient1(self.model, g, x, y)
                # print(noised_gv(x, y)[0])
                # g, v = noised_gv(x, y)
                # print(v)
            elif FLAGS.dpsam:
                print("dpsam will be used.")

            # if rho == 0:
            # lr = self.params.lr * jn.cos(progress * (7 * jn.pi) / (2 * 8))
            # jax.debug.print('lr_cos{lr_cos}', lr_cos=lr_cos)
            # lr = lr * jn.clip(progress * 100, 0, 1)
            # else:
            # lr = lr_cos
            g = clip_by_global_norm(g)
            self.opt(lr_cos, g)
            self.model_ema.update_ema()
            gradient_norm = jnp.sqrt(sum(
                [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(g)]))
            return {'monitors/lr': lr_cos, **v[1], 'gradient_norm/gradient_norm':gradient_norm}

        self.predict = objax.Jit(objax.nn.Sequential([objax.ForceArgs(self.model_ema, training=False)]))

        self.train_op = objax.Jit(train_op)


def network(arch: str):
    if arch == 'cnn32-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=32, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn32-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=16, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'cnn64-3-max':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.max_pool_2d)
    elif arch == 'cnn64-3-mean':
        return functools.partial(convnet.ConvNet, scales=3, filters=64, filters_max=1024,
                                 pooling=objax.functional.average_pool_2d)
    elif arch == 'wrn28-1':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=1)
    elif arch == 'wrn28-2':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=2)
    elif arch == 'wrn28-10':
        return functools.partial(wide_resnet.WideResNet, depth=28, width=10)
    elif arch == 'resnet18':
        return functools.partial(resnet_v2.ResNet18)
    elif arch == 'resnet50':
        return functools.partial(resnet_v2.ResNet50, normalization_fn=objax.nn.GroupNorm2D)
    raise ValueError('Architecture not recognized', arch)


def get_data(seed):
    """
    This is the function to generate subsets of the data for training models.

    First, we get the training dataset either from the numpy cache
    or otherwise we load it from tensorflow datasets.

    Then, we compute the subset. This works in one of two ways.

    1. If we have a seed, then we just randomly choose examples based on
       a prng with that seed, keeping FLAGS.pkeep fraction of the data.

    2. Otherwise, if we have an experiment ID, then we do something fancier.
       If we run each experiment independently then even after a lot of trials
       there will still probably be some examples that were always included
       or always excluded. So instead, with experiment IDs, we guarantee that
       after FLAGS.num_experiments are done, each example is seen exactly half
       of the time in train, and half of the time not in train.

    """
    DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')


    dataset_name = "x_train.npy"

    if os.path.exists(os.path.join(FLAGS.logdir, dataset_name)):
        inputs = np.load(os.path.join(FLAGS.logdir, dataset_name))
        labels = np.load(os.path.join(FLAGS.logdir, "y_train.npy"))
    else:
        print("First time, creating dataset")
        data = tfds.as_numpy(tfds.load(name=FLAGS.dataset, batch_size=-1, data_dir=DATA_DIR))
        inputs = data['train']['image']
        labels = data['train']['label']

        inputs = (inputs / 127.5) - 1
        np.save(os.path.join(FLAGS.logdir, "x_train.npy"), inputs)
        np.save(os.path.join(FLAGS.logdir, "y_train.npy"), labels)

    nclass = np.max(labels) + 1

    np.random.seed(seed)
    if FLAGS.num_experiments is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(FLAGS.num_experiments, FLAGS.dataset_size))
        order = keep.argsort(0)
        keep = order < int(FLAGS.pkeep * FLAGS.num_experiments)
        keep = np.array(keep[FLAGS.expid], dtype=bool)
    else:
        keep = np.random.uniform(0, 1, size=FLAGS.dataset_size) <= FLAGS.pkeep

    if FLAGS.only_subset is not None:
        keep[FLAGS.only_subset:] = 0

    xs = inputs[keep]
    ys = labels[keep]

    if FLAGS.augment == 'weak':
        aug = lambda x: augment(x, 4)
    elif FLAGS.augment == 'mirror':
        aug = lambda x: augment(x, 0)
    elif FLAGS.augment == 'none':
        aug = lambda x: augment(x, 0, mirror=False)
    else:
        raise

    train = DataSet.from_arrays(xs, ys,
                                augment_fn=aug)
    test = DataSet.from_tfds(tfds.load(name=FLAGS.dataset, split='test', data_dir=DATA_DIR), xs.shape[1:])
    train = train.cache().shuffle(8192).repeat().parse().augment().batch(FLAGS.batch)
    train = train.nchw().one_hot(nclass).prefetch(16)
    test = test.cache().parse().batch(FLAGS.batch).nchw().prefetch(16)

    if FLAGS.fair:
        inputs = np.load(os.path.join("x_gray_inv95_train.npy"))
        labels = np.load(os.path.join("y_gray_train.npy"))
        xs = inputs[keep]
        ys = labels[keep]
        train = DataSet.from_arrays(xs, ys,augment_fn=aug)
        train = train.cache().shuffle(8192).repeat().parse().augment().batch(FLAGS.batch)
        train = train.nchw().one_hot(nclass).prefetch(16)

        inputs_test = np.load(os.path.join("x_gray_inv95_test.npy"))
        labels_test = np.load(os.path.join("y_gray_test.npy"))
        test = DataSet.from_arrays(inputs_test, labels_test)
        # test = DataSet.from_tfds(tfds.load(name=FLAGS.dataset, split='test', data_dir=DATA_DIR), xs.shape[1:])
        test = test.cache().parse().batch(FLAGS.batch).nchw().prefetch(16)

    return train, test, xs, ys, keep, nclass


def main():
    # del argv
    tf.config.experimental.set_visible_devices([], "GPU")

    seed = FLAGS.seed
    if seed is None:
        import time
        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())

    args = EasyDict(arch=FLAGS.arch,
                    lr=FLAGS.lr,
                    batch=FLAGS.batch,
                    weight_decay=FLAGS.weight_decay,
                    augment=FLAGS.augment,
                    seed=seed)

    logdir = FLAGS.dataset+"/"+"fair_gray_inv95_"+str(FLAGS.fair)+"/"+FLAGS.arch+"/augment_"+FLAGS.augment+"/dpsgd_" + str(FLAGS.dpsgd)+ "-dpsam_" + str(FLAGS.dpsam) + "/microbatch_"+ str(FLAGS.microbatch)+ "/p_sigma_"+ str(FLAGS.dp_sigma)+"/rho_" + str(FLAGS.rho)

    if FLAGS.tunename:
        logdir = '_'.join(sorted('%s=%s' % k for k in args.items()))
    elif FLAGS.expid is not None:
        logdir = logdir + "/exp-%d_%d" % (FLAGS.expid, FLAGS.num_experiments)
    else:
        logdir = logdir + "/exp-" + str(seed)
    logdir = os.path.join(FLAGS.logdir, logdir)

    if os.path.exists(os.path.join(logdir, "ckpt", "%010d.npz" % FLAGS.epochs)):
        print(f"run {FLAGS.expid} already completed.")
        return
    else:
        if os.path.exists(logdir):
            print(f"deleting run {FLAGS.expid} that did not complete.")
            shutil.rmtree(logdir)

    print(f"starting run {FLAGS.expid}.")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    train, test, xs, ys, keep, nclass = get_data(seed)

    # Define the network and train_it
    tm = MemModule_sam(network(FLAGS.arch), nclass=nclass,
                       mnist=FLAGS.dataset == 'mnist',
                       epochs=FLAGS.epochs,
                       expid=FLAGS.expid,
                       num_experiments=FLAGS.num_experiments,
                       pkeep=FLAGS.pkeep,
                       save_steps=FLAGS.save_steps,
                       only_subset=FLAGS.only_subset,
                       dp=FLAGS.dpsam,
                       rho=FLAGS.rho,
                       **args
                       )

    r = {}
    r.update(tm.params)

    open(os.path.join(logdir, 'hparams.json'), "w").write(json.dumps(tm.params))
    np.save(os.path.join(logdir, 'keep.npy'), keep)

    tm.train(FLAGS.epochs, len(xs), train, test, logdir,
             save_steps=FLAGS.save_steps, patience=FLAGS.patience)



if __name__ == '__main__':
    flags.DEFINE_string('arch', 'resnet18', 'Model architecture.')

    flags.DEFINE_float('lr', 0.1, 'Learning rate.')
    flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
    flags.DEFINE_float('weight_decay', 0.001, 'Weight decay ratio.')
    flags.DEFINE_integer('batch', 256, 'Batch size')
    flags.DEFINE_integer('epochs', 200, 'Training duration in number of epochs.')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_integer('seed', None, 'Training seed.')
    flags.DEFINE_float('pkeep', .5, 'Probability to keep examples.')
    flags.DEFINE_integer('expid', None, 'Experiment ID')
    flags.DEFINE_integer('num_experiments', 16, 'Number of experiments')
    flags.DEFINE_string('augment', 'none', 'Strong or weak augmentation')
    flags.DEFINE_integer('only_subset', None, 'Only train on a subset of images.')
    flags.DEFINE_integer('dataset_size', 50000, 'number of examples to keep.')
    flags.DEFINE_integer('eval_steps', 1, 'how often to get eval accuracy.')
    flags.DEFINE_integer('abort_after_epoch', None, 'stop trainin early at an epoch')
    flags.DEFINE_integer('save_steps', 100, 'how often to get save model.')
    flags.DEFINE_integer('patience', None, 'Early stopping after this many epochs without progress')
    flags.DEFINE_integer('microbatch', 1, 'microbatch for dp')
    flags.DEFINE_bool('tunename', False, 'Use tune name?')
    flags.DEFINE_bool('dpsgd', False, 'Use dp sgd ?')
    flags.DEFINE_bool('dpsam', False, 'Use dp sam ?')
    flags.DEFINE_bool('fair', True, 'Use dp sam ?')
    flags.DEFINE_float('dp_sigma', 0.01, 'DP noise multiplier.')
    flags.DEFINE_float('rho', 0.05, 'rho for SAM.')
    flags.DEFINE_float('dp_clip_norm', 1, 'DP gradient clipping norm.')
    flags.DEFINE_float('dp_delta', 1e-5, 'DP-SGD delta for eps computation.')
    flags.DEFINE_integer('grad_acc_steps', 1,
                         'Number of steps for gradients accumulation, used to simulate large batches.')

    for i in range(16):
        FLAGS(sys.argv)
        FLAGS.expid = i
        FLAGS.rho = 0.0
        FLAGS.dpsam = False
        FLAGS.dpsgd = False
        FLAGS.microbatch = 16
        main()

