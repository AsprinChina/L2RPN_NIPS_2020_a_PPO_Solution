"""
In this file, a PPO model is reproduced following OPENAI baselines' PPO model.
@https://github.com/openai/baselines/tree/master/baselines/ppo2
"""
import numpy as np
import tensorflow as tf


class PPO(tf.Module):
    def __init__(self, coef_entropy=0.01, coef_value_func=0.5, max_grad_norm=0.5):
        super(PPO, self).__init__()
        self.model = Policy_Value_Network()
        self.optimizer = tf.keras.optimizers.Adam()
        self.coef_entropy = coef_entropy
        self.coef_value_func = coef_value_func
        self.max_grad_norm = max_grad_norm
        self.step = self.model.step
        self.value = self.model.value
        self.initial_state = None
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approx_kl', 'clip_ratio']

    def train(self, obs, returns, actions, values, neg_log_p_old, advs, lr=3e-4, clip_range=0.2):
        grads, policy_loss, value_loss, entropy, approx_kl, clip_ratio = self._get_grad(obs, returns, actions, values, neg_log_p_old, advs, clip_range)
        self.optimizer.learning_rate = lr
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return policy_loss, value_loss, entropy, approx_kl, clip_ratio

    @tf.function
    def _get_grad(self, obs, returns, actions, values, neg_log_p_old, advs, clip_range):
        # advs = returns - values
        # advs = (advs - tf.reduce_mean(advs)) / (tf.keras.backend.std(advs) + 1e-8)

        # obtain current gradient
        with tf.GradientTape() as tape:
            l, p, _ = self.model.model(obs)
            actions = tf.cast(actions, tf.int32)
            actions_one_hot = tf.one_hot(actions, l.get_shape().as_list()[-1])
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=l, labels=actions_one_hot)

            # calculate entropy bonus
            entropy = tf.reduce_mean(self._get_entropy(l))

            # calculate value loss
            vpred = self.model.value(obs)
            vpred_clip = values + tf.clip_by_value(vpred - values, -clip_range, clip_range)
            value_loss1 = tf.square(vpred - returns)
            value_loss2 = tf.square(vpred_clip - returns)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss1, value_loss2))

            # calculate policy loss
            ratio = tf.exp(neg_log_p_old - neg_log_p)
            policy_loss1 = -advs * ratio
            policy_loss2 = -advs * tf.clip_by_value(ratio, (1 - clip_range), (1 + clip_range))
            policy_loss = tf.reduce_mean(tf.maximum(policy_loss1, policy_loss2))

            approx_kl = 0.5 * tf.reduce_mean(tf.square(neg_log_p_old - neg_log_p))
            clip_ratio = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1), clip_range), tf.float32))

            # Sigma loss
            loss = policy_loss * 10 + value_loss * self.coef_value_func - entropy * self.coef_entropy

        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        return grads, policy_loss * 10, value_loss * self.coef_value_func, entropy, approx_kl, clip_ratio

    @staticmethod
    def _get_entropy(l):
        a0 = l - tf.reduce_max(l, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)


class Policy_Value_Network(tf.Module):
    def __init__(self, junior_path='./JuniorModel'):
        # warm start from Junior Student
        self.model = PVNet()
        self.model.build((None, 1221,))
        self._params_copy(junior_path)

    def _params_copy(self, path):
        model = tf.keras.models.load_model(path)
        self.model.layers[0].set_weights(model.layers[0].get_weights())
        self.model.layers[1].set_weights(model.layers[1].get_weights())
        self.model.layers[2].set_weights(model.layers[2].get_weights())
        self.model.layers[3].set_weights(model.layers[4].get_weights())
        self.model.layers[4].set_weights((*map(lambda x: x / 5, model.layers[6].get_weights()),))

    @tf.function
    def step(self, obs):
        # l for logits, p for possibility, and v for value
        l, p, v = self.model(obs)
        # sampling by Gumbel-max trick
        u = tf.random.uniform(tf.shape(l), dtype=np.float32)
        a = tf.argmax(l - tf.math.log(-tf.math.log(u)), axis=-1)
        a_one_hot = tf.one_hot(a, l.get_shape().as_list()[-1])  # important!
        # calculate -log(pi)
        neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=l, labels=a_one_hot)
        v = tf.squeeze(v, axis=1)
        return a, v, neg_log_p, l

    @tf.function
    def value(self, obs):
        _, _, v = self.model(obs)
        v = tf.squeeze(v, axis=1)
        return v


class PVNet(tf.keras.Model):
    def __init__(self):
        super(PVNet, self).__init__()
        n_cell = 1000
        initializer = tf.keras.initializers.Orthogonal()
        self.layer1 = tf.keras.layers.Dense(n_cell, activation='relu', kernel_initializer=initializer)
        self.layer2 = tf.keras.layers.Dense(n_cell, activation='relu', kernel_initializer=initializer)
        self.layer3 = tf.keras.layers.Dense(n_cell, activation='relu', kernel_initializer=initializer)
        self.layer4 = tf.keras.layers.Dense(n_cell, activation='relu', kernel_initializer=initializer)
        self.act_layer = tf.keras.layers.Dense(208, activation=None, kernel_initializer=initializer)
        self.val_hidden_layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)
        self.val_layer = tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer)

    def call(self, s):
        s = self.layer1(s)
        s = self.layer2(s)
        s = self.layer3(s)
        s = self.layer4(s)
        l = self.act_layer(s)  # logits
        p = tf.nn.softmax(l)  # probability distribution of actions
        vh = self.val_hidden_layer(s)
        v = self.val_layer(vh)  # state value
        return l, p, v


if __name__ == '__main__':
    # for test only
    m = Policy_Value_Network()
    l, p, v = m.model(np.ones((1, 1221)))
