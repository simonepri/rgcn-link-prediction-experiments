import tensorflow as tf

from mocks.decoders.base import BaseDecoder

# pylint: disable=invalid-name
class DistMult(BaseDecoder):
    def calc_score(self, e1s, rs, e2s):
        energies = tf.reduce_sum(e1s * rs * e2s, 1)
        return energies

    def calc_all_subjects_score(self, all_subjects, rs, e2s):
        all_energies = tf.matmul(all_subjects, tf.transpose(rs * e2s))
        all_energies = tf.transpose(all_energies)
        return all_energies

    def calc_all_objects_score(self, e1s, rs, all_objects):
        all_energies = tf.matmul(e1s * rs, tf.transpose(all_objects))
        return all_energies
