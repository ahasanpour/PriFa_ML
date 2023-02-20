# Copyright 2020 Google LLC
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

import random

import numpy as np
import tensorflow as tf

import objax
from objax.zoo.wide_resnet import WideResNet
from objax.module import Module, ModuleList
from flax.training import lr_schedule
import jax
import objax
import math
from objax.jaxboard import SummaryWriter, Summary
from objax.util import EasyDict
from objax.zoo import convnet, wide_resnet
from objax.zoo import resnet_v2
from objax.variable import TrainRef, StateVar, TrainVar, VarCollection
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple
# Data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.transpose(0, 3, 1, 2) / 255.0
X_test = X_test.transpose(0, 3, 1, 2) / 255.0

# Model
model = WideResNet(nin=3, nclass=10, depth=28, width=2)
opt = objax.optimizer.Adam(model.vars())


# Losses
@objax.Function.with_vars(model.vars())
def loss(x, label, train=True):
    logit = model(x, training=train)
    return objax.functional.loss.cross_entropy_logits_sparse(logit, label).mean()

def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(sum(
      [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient

gv = objax.GradValues(loss, model.vars())


rho = 0.05
import copy
# @objax.Function.with_vars(model.vars())
vc = model.vars()
vc1 = copy.deepcopy(vc)
def get_sam_gradient1( g, v, x1, y):
    # gv = objax.GradValues(loss, model.vars())

    train_vars = ModuleList(x for x in vc.subset(TrainVar))
    # g = jax.lax.pmean(g, 'batch')
    # jax.debug.print('vc _ train_vars{train_vars}', train_vars=train_vars[0][0])
    g = dual_vector(g)
    # train_vars = jax.tree_util.tree_map(lambda a, b: a + rho * b,
    #                                  train_vars, g)
    # self.m = ModuleList(StateVar(jn.zeros_like(x.value)) for x in self.train_vars)
    assert len(g) == len(train_vars), 'Expecting as many gradients as trainable variables'
    scale = rho  # / (normalized_gradient + 1e-12)
    e_ws = []
    # for l, gr in zip(self.train_vars, normalized_gradient):
    #
    # l.value -= m.value
    for i in range(len(train_vars)):
        ew = g[i] * scale
        train_vars[i].assign(train_vars[i] + ew)
        e_ws.append(ew)
    # jax.debug.print('train_vars after noise{train_vars}', train_vars=train_vars[0][0])
    # gvn = objax.GradValues(loss, train_vars)
    g1, v1 = gv(x1, y, train=True)
    # jax.debug.print('e_ws {e_ws}', e_ws=e_ws[0][0])

    # jax.debug_infs(v1)
    # train_vars_tmp = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))
    # for i in range(len(train_vars_tmp)):
    #     train_vars_tmp[i] -= e_ws[i]

    # jax.debug.print('vc next _ train_vars{train_vars}', train_vars=train_vars[0][0])
    # jax.debug.print('vc _ train_vars_tmp{train_vars_tmp}', train_vars_tmp=train_vars_tmp[0][0])
    # for l, gr in zip(self.train_vars, normalized_gradient):
    #     l.value += rho * gr
    # model.vars = train_vars_tmp
    # g2, v2 = gv(x1, y)
    # jax.debug.print("{v}, {v1}, {v2}", v=v, v1=v1, v2=v2)
    # jax.debug.breakpoint()
    return g1, v1


# @objax.Function.with_vars(model.vars() + gv.vars() + opt.vars())
@objax.Function.with_vars(model.vars() + gv.vars() + opt.vars())
def train_op(x, y):
    g, v = gv(x, y,train=True)

    if rho > 0:
        print("SAM will be used.")
        g, v = get_sam_gradient1(g, v, x, y)

    lr = learning_rate_fn(opt.step.value)
    # jax.debug.print("lr{lr}", lr=lr)

    opt(lr=lr, grads=g)
    return v



train_op = objax.Jit(train_op)
predict = objax.Jit(objax.nn.Sequential([
    objax.ForceArgs(model, training=False), objax.functional.softmax
]))


def augment(x):
    if random.random() < .5:
        x = x[:, :, :, ::-1]  # Flip the batch images about the horizontal axis
    # Pixel-shift all images in the batch by up to 4 pixels in any direction.
    x_pad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'reflect')
    rx, ry = np.random.randint(0, 8), np.random.randint(0, 8)
    x = x_pad[:, :, rx:rx + 32, ry:ry + 32]
    return x

def get_cosine_schedule(num_epochs: int, learning_rate: float,
                        num_training_obs: int,
                        batch_size: int) -> Callable[[int], float]:
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(
      learning_rate, steps_per_epoch // jax.host_count(), num_epochs,
      warmup_length=0)
  return learning_rate_fn

learning_rate_fn = get_cosine_schedule(60, 0.2, num_training_obs=50000, batch_size=256)

# Training
print(model.vars())
for epoch in range(60):
    # Train
    loss = []

    sel = np.arange(len(X_train))
    np.random.shuffle(sel)


    for it in range(0, X_train.shape[0], 256):
        loss.append(train_op(augment(X_train[sel[it:it + 256]]), Y_train[sel[it:it + 256]].flatten()))

    # Eval
    test_predictions = [predict(x_batch).argmax(1) for x_batch in X_test.reshape((50, -1) + X_test.shape[1:])]
    accuracy = np.array(test_predictions).flatten() == Y_test.flatten()
    print(f'Epoch {epoch + 1:4d}  Loss {np.mean(loss):.2f}  Accuracy {100 * np.mean(accuracy):.2f}')