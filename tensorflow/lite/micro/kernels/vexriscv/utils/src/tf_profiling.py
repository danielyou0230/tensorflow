import argparse
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from google.protobuf import text_format
import tensorflow.keras.backend as K
"""
python tf_profiling.py ./micro_speech_ckpt/tiny_conv.pbtxt tf \
--ckpt=./micro_speech_ckpt/checkpoint \
--meta=./micro_speech_ckpt/tiny_conv.ckpt-11000.meta \
--count

Keras
python tf_profiling.py magic_wand.h5 k
"""


def pbtxt_to_pb(log_dir, filename, output=None):
    with open(filename) as f:
        content = f.read()
        gdef = text_format.Parse(content, tf.compat.v1.GraphDef())

    if output is None:
        name = os.path.splitext(os.path.basename(filename))[0]
        output = "{}.pb".format(name)
    tf.io.write_graph(gdef, log_dir, output, as_text=False)


def load_pb(pb):
    with tf.io.gfile.GFile(pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


def get_tf_model_flops(filename):
    name, ext = os.path.splitext(filename)
    if ext.lower() == ".pbtxt":
        pbtxt_to_pb(os.path.dirname(filename), filename)
        filename = "{}.pb".format(name)

    graph = load_pb(filename)
    with graph.as_default():
        flops = tf.compat.v1.profiler.profile(
            graph,
            options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(
            ))

    return flops.total_float_ops


def get_keras_model_flops(filename):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(filename)

            counts = int(
                np.sum(
                    [K.count_params(p) for p in set(model.trainable_weights)]))

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta,
                                                  cmd='op',
                                                  options=opts)
        print("Trainable: {}".format(counts))
        print("FLOPS: {}".format(flops.total_float_ops))
        return flops.total_float_ops


def count_tf_trainable_params(ckpt, file):
    tf.compat.v1.disable_eager_execution()
    saver = tf.compat.v1.train.import_meta_graph(file)

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(file)))

    # total_parameters = 0
    # for variable in tf.compat.v1.trainable_variables():
    #     print(variable)
    #     total_parameters += 1

    counts = np.sum([
        np.prod(v.get_shape().as_list())
        for v in tf.compat.v1.trainable_variables()
    ])
    print("Trainable: {}".format(counts))
    # print(total_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        type=str,
                        help="Path to the model checkpoint file.")
    parser.add_argument(
        "lib",
        type=str,
        help=
        "Specify the lib (Tensorflow 'tf' or Keras 'k' that is used to save the model.)"
    )

    parser.add_argument("--meta", type=str, help="*.meta file")
    parser.add_argument("--ckpt", type=str, help="*.index file")
    parser.add_argument("--count",
                        action="store_true",
                        help="Count number of parameters")

    args = parser.parse_args()

    if args.lib == "tf":
        get_tf_model_flops(args.file)
        if args.count:
            print("*** {}".format(os.path.dirname(args.meta)))
            count_tf_trainable_params(args.ckpt, args.meta)

    elif args.lib == "k":
        get_keras_model_flops(args.file)
    else:
        print(
            "Lib argument error: Please choose a valid lib to parse, 'tf' for Tensorflow or 'k' for Keras"
        )
