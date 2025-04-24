import tensorflow as tf

MODEL_CKPT = "../pensieve_rl_model/pretrain_linear_reward.ckpt"

graph = tf.Graph()
with graph.as_default():
    tf.compat.v1.train.import_meta_graph(MODEL_CKPT + ".meta")

print("\n=== tensors with shape (None, 6) under 'actor/' ===")
found = False
for op in graph.get_operations():
    if not op.name.startswith("actor/"):
        continue
    for out in op.outputs:
        if out.shape.as_list() == [None, 6]:
            print(f"{out.name}")
            found = True

if not found:
    print("⚠️  none found — paste this output back to me")
