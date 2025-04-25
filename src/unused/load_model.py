class PensieveActorTF:
    def __init__(self, ckpt_prefix):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(ckpt_prefix + ".meta")
            self.state_ph = self.graph.get_tensor_by_name("actor/InputData/X:0")
            self.logits   = self.graph.get_tensor_by_name("actor/FullyConnected_4/MatMul:0")

        self.sess = tf.Session(graph=self.graph)
        self.saver.restore(self.sess, ckpt_prefix)

    def __call__(self, state_np):              # state_np : (B,6,8)
        return self.sess.run(self.logits, {self.state_ph: state_np})

    def close(self):
        self.sess.close()
