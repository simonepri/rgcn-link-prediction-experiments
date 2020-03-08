import tensorflow as tf

from submodules.rgcn.code.model import Model

# pylint: disable=invalid-name
class BaseDecoder(Model):
    encoder_cache = {"train": None, "test": None}

    def __init__(self, next_component, settings):
        super(BaseDecoder, self).__init__(next_component, settings)
        self.X = None
        self.Y = None

    def parse_settings(self):
        self.regularization_parameter = float(
            self.settings["RegularizationParameter"]
        )
        self.negative_sample_rate = int(self.settings["NegativeSampleRate"])

    def compute_codes(self, mode="train"):
        if self.encoder_cache[mode] is not None:
            return self.encoder_cache[mode]

        (
            subject_codes,
            relation_codes,
            object_codes,
        ) = self.next_component.get_all_codes(mode=mode)
        e1s = tf.nn.embedding_lookup(subject_codes, self.X[:, 0])
        rs = tf.nn.embedding_lookup(relation_codes, self.X[:, 1])
        e2s = tf.nn.embedding_lookup(object_codes, self.X[:, 2])

        self.encoder_cache[mode] = (e1s, rs, e2s)
        return self.encoder_cache[mode]

    def local_initialize_train(self):
        self.Y = tf.placeholder(tf.float32, shape=[None])
        self.X = tf.placeholder(tf.int32, shape=[None, 3])

    def local_get_train_input_variables(self):
        return [self.X, self.Y]

    def local_get_test_input_variables(self):
        return [self.X]

    def calc_score(self, e1s, rs, e2s):
        raise NotImplementedError()

    def calc_all_subjects_score(self, all_subjects, rs, e2s):
        raise NotImplementedError()

    def calc_all_objects_score(self, e1s, rs, all_objects):
        raise NotImplementedError()

    def get_loss(self, mode="train"):
        e1s, rs, e2s = self.compute_codes(mode=mode)
        energies = self.calc_score(e1s, rs, e2s)
        weight = self.negative_sample_rate
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(self.Y, energies, weight)
        )

    def local_get_regularization(self):
        e1s, rs, e2s = self.compute_codes(mode="train")
        regularization = tf.reduce_mean(tf.square(e1s))
        regularization += tf.reduce_mean(tf.square(rs))
        regularization += tf.reduce_mean(tf.square(e2s))

        return self.regularization_parameter * regularization

    def predict(self):
        e1s, rs, e2s = self.compute_codes(mode="test")
        energies = self.calc_score(e1s, rs, e2s)
        return tf.nn.sigmoid(energies)

    def predict_all_subject_scores(self):
        _, rs, e2s = self.compute_codes(mode="test")
        all_subjects = self.next_component.get_all_subject_codes(mode="test")
        all_energies = self.calc_all_subjects_score(all_subjects, rs, e2s)
        return tf.nn.sigmoid(all_energies)

    def predict_all_object_scores(self):
        e1s, rs, _ = self.compute_codes(mode="test")
        all_objects = self.next_component.get_all_object_codes(mode="test")
        all_energies = self.calc_all_objects_score(e1s, rs, all_objects)
        return tf.nn.sigmoid(all_energies)
