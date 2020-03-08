import tensorflow as tf

from mocks.decoders.base import BaseDecoder

# pylint: disable=invalid-name
class ComplEx(BaseDecoder):
    def parse_settings(self):
        super(ComplEx, self).parse_settings()
        self.dimension = int(self.settings["CodeDimension"])

    def extract_real_and_imaginary(self, composite_vector):
        embedding_dim = int(self.dimension / 2)
        r = tf.slice(composite_vector, [0, 0], [-1, embedding_dim])
        i = tf.slice(composite_vector, [0, embedding_dim], [-1, embedding_dim])
        return r, i

    def calc_score(self, e1s, rs, e2s):
        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s)
        rs_r, rs_i = self.extract_real_and_imaginary(rs)
        energies = (
            tf.reduce_sum(e1s_r * rs_r * e2s_r, 1)
            + tf.reduce_sum(e1s_i * rs_r * e2s_i, 1)
            + tf.reduce_sum(e1s_r * rs_i * e2s_i, 1)
            - tf.reduce_sum(e1s_i * rs_i * e2s_r, 1)
        )
        return energies

    def calc_all_subjects_score(self, all_subjects, rs, e2s):
        e1s_r, e1s_i = self.extract_real_and_imaginary(all_subjects)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s)
        rs_r, rs_i = self.extract_real_and_imaginary(rs)
        all_energies = (
            tf.matmul(e1s_r, tf.transpose(rs_r * e2s_r))
            + tf.matmul(e1s_i, tf.transpose(rs_r * e2s_i))
            + tf.matmul(e1s_r, tf.transpose(rs_i * e2s_i))
            - tf.matmul(e1s_i, tf.transpose(rs_i * e2s_r))
        )
        all_energies = tf.transpose(all_energies)
        return all_energies

    def calc_all_objects_score(self, e1s, rs, all_objects):
        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s)
        e2s_r, e2s_i = self.extract_real_and_imaginary(all_objects)
        rs_r, rs_i = self.extract_real_and_imaginary(rs)
        all_energies = (
            tf.matmul(e1s_r * rs_r, tf.transpose(e2s_r))
            + tf.matmul(e1s_i * rs_r, tf.transpose(e2s_i))
            + tf.matmul(e1s_r * rs_i, tf.transpose(e2s_i))
            - tf.matmul(e1s_i * rs_i, tf.transpose(e2s_r))
        )
        return all_energies
