import tensorflow as tf
import numpy as np

from DRRB.util import *
from DRRB.distance import wasserstein
class DRRB_net(object):

    def __init__(self, x, t, y_ , p_t, z_norm, FLAGS, r_lambda, do_in, do_out, dims):
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, z_norm, FLAGS, r_lambda, do_in, do_out, dims)

    def _add_variable(self, var, name):
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i)
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, z_norm, FLAGS, r_lambda, do_in, do_out, dims):
        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out
        self.z_norm = z_norm

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]
        dim_d = dims[3]



        ''' Construct input/representation layers '''
        with tf.variable_scope('encoder') as scope:
            weights_in = []; biases_in = []

            if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
                dim_in = dim_input
            if FLAGS.n_out == 0:
                if FLAGS.split_output == False:
                    dim_out = dim_in+1
                else:
                    dim_out = dim_in

            if FLAGS.batch_norm:
                bn_biases = []
                bn_scales = []

            h_in = [x]

            for i in range(0, FLAGS.n_in):
                if i==0:
                    ''' If using variable selection, first layer is just rescaling'''
                    if FLAGS.varsel:
                        weights_in.append(tf.Variable(1.0/dim_input*tf.ones([dim_input])))
                    else:
                        weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_in], \
                            stddev=FLAGS.weight_init/np.sqrt(dim_input))))
                else:
                    weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], \
                        stddev=FLAGS.weight_init/np.sqrt(dim_in))))

                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel and i==0:
                    biases_in.append([])
                    h_in.append(tf.mul(h_in[i],weights_in[i]))
                else:
                    biases_in.append(tf.Variable(tf.zeros([1,dim_in])))
                    z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                    if FLAGS.batch_norm:
                        batch_mean, batch_var = tf.nn.moments(z, [0])

                        if FLAGS.normalization == 'bn_fixed':
                            z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                        else:
                            bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                            bn_scales.append(tf.Variable(tf.ones([dim_in])))
                            z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                    h_in.append(self.nonlin(z))
                    h_in[i+1] = tf.nn.dropout(h_in[i+1], do_in)

            h_rep = h_in[len(h_in)-1]

            if FLAGS.normalization == 'divide':
                h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
            else:
                h_rep_norm = 1.0*h_rep
        '''representation: min IPM'''
        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5
        imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                        sq=False, backpropT=FLAGS.wass_bpt)
        rep_loss = imb_dist
        '''Adversarial'''
        tpre, weights_dis, weights_discore = self._build_discriminator(h_rep_norm, dim_in, dim_d, do_out, FLAGS)
        if FLAGS.t_pre_smooth==1:
            tpre = (tpre + 0.01) / 1.02
            # tpre = 0.995 / (1.0 + tf.exp(-tpre)) + 0.0025
        if FLAGS.reweight_sample_t==1:
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * (1 - p_t))
            sample_weight_t = w_t + w_c
            if FLAGS.safelog_t==1:
                discriminator_loss = -tf.reduce_mean(sample_weight_t * (t * safe_log(tpre) + (1.0 - t) * safe_log((1.0 - tpre))))
            else:
                discriminator_loss = -tf.reduce_mean(sample_weight_t * (t * tf.log(tpre) + (1.0 - t) * tf.log((1.0 - tpre))))
            # rep_loss = tf.reduce_mean(sample_weight_t * (t * tf.log(tpre) + (1.0 - t) * tf.log(1.0 - tpre)))
        else:
            if FLAGS.safelog_t == 1:
                discriminator_loss = -tf.reduce_mean(t * safe_log(tpre) + (1.0 - t) * safe_log(1.0 - tpre))
            else:
                discriminator_loss = -tf.reduce_mean(t * tf.log(tpre) + (1.0 - t) * tf.log(1.0 - tpre))
            # rep_loss = tf.reduce_mean(t * tf.log(tpre) + (1.0 - t) * tf.log(1.0 - tpre))


        ''' Construct ouput layers '''
        y0_f, y1_f, y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)
        y0_cf, y1_cf, ycf, weights_out, weights_pred = self._build_output_graph(h_rep_norm, 1-t, dim_in, dim_out, do_out, FLAGS)
        y1 = t*y + (1-t)*ycf
        y0 = t * ycf + (1 - t) * y
        imb_dist_y0, _ = wasserstein(y0, t, 0.5, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                               sq=False, backpropT=FLAGS.wass_bpt)
        imb_dist_y1, _ = wasserstein(y1, t, 0.5, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                               sq=False, backpropT=FLAGS.wass_bpt)
        ''' Compute sample reweighting '''
        if FLAGS.reweight_sample==1:
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * (1 - p_t))
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if FLAGS.loss == 'l1':
            risk = tf.reduce_mean(sample_weight*tf.abs(y_-y))
            pred_error = -tf.reduce_mean(tf.abs(y_-y))
        elif FLAGS.loss == 'log':
            if FLAGS.y_pre_smooth==1:
                y = (y + 0.01) / 1.02
                # y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            if safelog_y == 1:
                res = y_ * safe_log(y) + (1.0 - y_) * safe_log(1.0 - y)
            else:
                res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)

            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        elif FLAGS.loss == 'mse':
            risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))
        elif FLAGS.loss == 'rmse':
            risk = tf.sqrt(tf.reduce_mean(sample_weight * tf.square(y_ - y)))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])
        'DR estimator'

        DR_y1 = y1+t/tpre*(y_-y1)
        DR_y0 = y0 + (1-t) / (1-tpre) * (y_ - y0)
        DR_ITE = DR_y1 - DR_y0
        DR_risk = tf.reduce_mean(tf.square(t*(DR_y1) + (1-t)*(DR_y0)-y_))
        ''' Total error '''
        # check1 = (1-t)*(1-tpre)
        # check2 = t*(1-tpre)*tf.square(y_-y1)
        # outcome_reg = check1 + check2
        # outcome_reg = tf.reduce_mean((1-t)*tpre*tf.square(y_-y0) + t*(1-tpre)*tf.square(y_-y1))
        tot_error = risk + FLAGS.p_ydis*(imb_dist_y0 + imb_dist_y1)



        if FLAGS.p_lambda>0:
            tot_error = tot_error + r_lambda*self.wd_loss

        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        # '''min ITE risk'''
        # with tf.variable_scope('tau') as scope:
        #     tau = tf.Variable(tf.random_normal([]), name='tau')

        self.output = y
        self.tot_loss = tot_error
        self.discriminator_loss = discriminator_loss
        self.rep_loss = rep_loss
        self.pred_loss = pred_error
        self.DR_loss = DR_risk
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_dis = weights_dis
        self.weights_discore = weights_discore
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm
        self.t_pre = tpre
        # self.epsilon_t = epsilon_t
        # self.epsilon_y = epsilon_y
        self.DR_ITE = DR_ITE
    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*FLAGS.n_out)
        with tf.variable_scope('pred') as scope:
            weights_out = []; biases_out = []

            for i in range(0, FLAGS.n_out):
                wo = self._create_variable_with_weight_decay(
                        tf.random_normal([dims[i], dims[i+1]],
                            stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                        'out_w_%d' % i, 1.0)
                weights_out.append(wo)

                biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
                z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

                h_out.append(self.nonlin(z))
                h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

            weights_pred = self._create_variable(tf.random_normal([dim_out,1],
                stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
            bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

            if FLAGS.varsel or FLAGS.n_out == 0:
                self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
            else:
                self.wd_loss += tf.nn.l2_loss(weights_pred)

            ''' Construct linear classifier '''
            h_pred = h_out[-1]
            y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''
        if FLAGS.split_output:

            i0 = tf.to_int32(tf.where(t < 1)[:,0])
            i1 = tf.to_int32(tf.where(t > 0)[:,0])

            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)

            y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat([rep, t], 1)
            y, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)

        return y0, y1, y, weights_out, weights_pred

    def _build_discriminator(self, hrep, dim_in, dim_d, do_out, FLAGS, reuse=False):
        ''' Construct adversarial discriminator layers '''
        h_dis = [hrep]
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            weights_dis = []
            biases_dis = []
            for i in range(0, FLAGS.n_dc):

                if i==0:
                    weights_dis.append(tf.Variable(tf.random_normal([dim_in,dim_d], \
                    stddev=FLAGS.weight_init/np.sqrt(dim_in))))
                else:
                    weights_dis.append(tf.Variable(tf.random_normal([dim_d,dim_d], \
                    stddev=FLAGS.weight_init/np.sqrt(dim_d))))
                biases_dis.append(tf.Variable(tf.zeros([1,dim_d])))
                z = tf.matmul(h_dis[i], weights_dis[i])+biases_dis[i]
                h_dis.append(self.nonlin(z))
                # if i != FLAGS.n_dc - 1:
                #     h_dis.append(self.nonlin(z))
                # else:
                #     h_dis.append(tf.tanh(z))
                h_dis[i + 1] = tf.nn.dropout(h_dis[i + 1], do_out)

            weights_discore = self._create_variable(tf.random_normal([dim_d,1],
                stddev=FLAGS.weight_init/np.sqrt(dim_d)), 'dc_p')
            bias_dc = self._create_variable(tf.zeros([1]), 'dc_b_p')

            h_score = h_dis[-1]
            dis_score = tf.nn.sigmoid(tf.matmul(h_score, weights_discore) + bias_dc)
            # dis_score = 0.995 / (1.0 + tf.exp(-dis_score)) + 0.0025

        return dis_score, weights_dis, weights_discore


