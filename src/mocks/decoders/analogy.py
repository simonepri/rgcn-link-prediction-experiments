import tensorflow as tf

from mocks.decoders.base import BaseDecoder

# pylint: disable=invalid-name
class Analogy(BaseDecoder):
    def parse_settings(self):
        super(Analogy, self).parse_settings()
        self.dimension = int(self.settings["CodeDimension"])

    def extract_real_and_imaginary(self, composite_vector):
        embedding_dim = int(self.dimension / 4)
        r = tf.slice(composite_vector, [0, 0], [-1, embedding_dim])
        i = tf.slice(composite_vector, [0, embedding_dim], [-1, embedding_dim])
        return r, i

    def calc_score(self, e1s, rs, e2s):
        half = self.dimension // 2

        e1s_c = e1s[..., :half]
        e2s_c = e2s[..., :half]
        rs_c = rs[..., :half]
        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s_c)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s_c)
        rs_r, rs_i = self.extract_real_and_imaginary(rs_c)
        energies_comp = (
            tf.reduce_sum(e1s_r * rs_r * e2s_r, 1)
            + tf.reduce_sum(e1s_i * rs_r * e2s_i, 1)
            + tf.reduce_sum(e1s_r * rs_i * e2s_i, 1)
            - tf.reduce_sum(e1s_i * rs_i * e2s_r, 1)
        )

        e1s_d = e1s[..., half:]
        e2s_d = e2s[..., half:]
        rs_d = rs[..., half:]
        energies_dist = tf.reduce_sum(e1s_d * rs_d * e2s_d, 1)
        return energies_comp + energies_dist


    def local_get_regularization(self):
        half = self.dimension // 2
        e1s, rs, e2s = self.compute_codes(mode="train")

        e1s_c = e1s[..., :half]
        e2s_c = e2s[..., :half]
        rs_c = rs[..., :half]
        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s_c)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s_c)
        rs_r, rs_i = self.extract_real_and_imaginary(rs_c)
        regularization += tf.reduce_mean(e1s_r ** 2)
        regularization += tf.reduce_mean(e1s_i ** 2)
        regularization += tf.reduce_mean(e2s_r ** 2)
        regularization += tf.reduce_mean(e2s_i ** 2)
        regularization += tf.reduce_mean(rs_r ** 2)
        regularization += tf.reduce_mean(rs_i ** 2)

        e1s_d = e1s[..., half:]
        e2s_d = e2s[..., half:]
        rs_d = rs[..., half:]
        regularization += tf.reduce_mean(e1s_d ** 2)
        regularization += tf.reduce_mean(e2s_d ** 2)
        regularization += tf.reduce_mean(rs_d ** 2)
        return self.regularization_parameter * regularization

    def calc_all_subjects_score(self, all_subjects, rs, e2s):
        half = self.dimension // 2

        e1s_c = all_subjects[..., :half]
        e2s_c = e2s[..., :half]
        rs_c = rs[..., :half]
        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s_c)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s_c)
        rs_r, rs_i = self.extract_real_and_imaginary(rs_c)
        all_energies_comp = (
            tf.matmul(e1s_r, tf.transpose(rs_r * e2s_r))
            + tf.matmul(e1s_i, tf.transpose(rs_r * e2s_i))
            + tf.matmul(e1s_r, tf.transpose(rs_i * e2s_i))
            - tf.matmul(e1s_i, tf.transpose(rs_i * e2s_r))
        )
        all_energies_comp = tf.transpose(all_energies_comp)

        e1s_d = all_subjects[..., half:]
        e2s_d = e2s[..., half:]
        rs_d = rs[..., half:]
        all_energies_dist = tf.matmul(e1s_d, tf.transpose(rs_d * e2s_d))
        all_energies_dist = tf.transpose(all_energies_dist)
        return all_energies_comp + all_energies_dist

    def calc_all_objects_score(self, e1s, rs, all_objects):
        half = self.dimension // 2

        e1s_c = e1s[..., :half]
        e2s_c = all_objects[..., :half]
        rs_c = rs[..., :half]
        e1s_r, e1s_i = self.extract_real_and_imaginary(e1s_c)
        e2s_r, e2s_i = self.extract_real_and_imaginary(e2s_c)
        rs_r, rs_i = self.extract_real_and_imaginary(rs_c)
        all_energies_comp = (
            tf.matmul(e1s_r * rs_r, tf.transpose(e2s_r))
            + tf.matmul(e1s_i * rs_r, tf.transpose(e2s_i))
            + tf.matmul(e1s_r * rs_i, tf.transpose(e2s_i))
            - tf.matmul(e1s_i * rs_i, tf.transpose(e2s_r))
        )

        e1s_d = e1s[..., half:]
        e2s_d = all_objects[..., half:]
        rs_d = rs[..., half:]
        all_energies_dist = tf.matmul(e1s_d * rs_d, tf.transpose(e2s_d))
        return all_energies_comp + all_energies_dist
