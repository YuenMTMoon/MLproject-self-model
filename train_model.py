import argparse
import functools
import io
from datetime import datetime
from model_utils.model import DeepSpeech2Model
from data_utils.data_preprocessing import get_data_len, DataGenerator  # Update the import
from utils.utility import add_arguments, print_arguments

import paddle.fluid as fluid

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size', int, 16, "Batch size for training")
add_arg('num_epoch', int, 200, "Number of training epochs")
add_arg('num_conv_layers', int, 2, "Number of convolution layers")
add_arg('num_rnn_layers', int, 3, "Number of recurrent layers")
add_arg('rnn_layer_size', int, 2048, "Size of the recurrent layers")
add_arg('learning_rate', float, 5e-5, "Initial learning rate")
add_arg('min_duration', float, 1.0, "Minimum duration of audio for training")
add_arg('max_duration', float, 15.0, "Maximum duration of audio for training")
add_arg('test_off', bool, False, "Turn off testing during training")
add_arg('use_gru', bool, True, "Whether to use GRUs in the model (if False, use RNNs)")
add_arg('use_gpu', bool, True, "Whether to use GPU for training")
add_arg('share_rnn_weights', bool, False, "Whether to share weights in the recurrent layers")
add_arg('init_from_pretrained_model', str, None, "Path to a pretrained model (set to None if not using)")
add_arg('train_manifest', str, './dataset/manifest.train', "Path to the training manifest file")
add_arg('dev_manifest', str, './dataset/manifest.test', "Path to the testing manifest file")
add_arg('mean_std_path', str, './dataset/mean_std.npz', "Path to the mean and std dev file for normalization")
add_arg('vocab_path', str, './dataset/en_vocab.txt', "Path to the vocabulary file")
add_arg('output_model_dir', str, "./models", "Directory to save the trained models")
add_arg('augment_conf_path', str, './conf/augmentation.config', "Path to the augmentation configuration file (json format)")
add_arg('shuffle_method', str, 'batch_shuffle_clipped', "Method for shuffling data", choices=['instance_shuffle', 'batch_shuffle', 'batch_shuffle_clipped'])
args = parser.parse_args()

def train():
    # Determine whether to use GPU
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    # Get the training data generator
    train_generator = DataGenerator(vocab_filepath=args.vocab_path,
                                    mean_std_filepath=args.mean_std_path,
                                    augmentation_config=io.open(args.augment_conf_path, mode='r', encoding='utf8').read(),
                                    max_duration=args.max_duration,
                                    min_duration=args.min_duration,
                                    place=place)

    # Get the testing data generator
    test_generator = DataGenerator(vocab_filepath=args.vocab_path,
                                   mean_std_filepath=args.mean_std_path,
                                   keep_transcription_text=True,
                                   place=place,
                                   is_training=False)

    # Get the training data
    train_batch_reader = train_generator.batch_reader_creator(manifest_path=args.train_manifest,
                                                              batch_size=args.batch_size,
                                                              shuffle_method=args.shuffle_method)

    # Get the testing data
    test_batch_reader = test_generator.batch_reader_creator(manifest_path=args.dev_manifest,
                                                            batch_size=args.batch_size,
                                                            shuffle_method=None)

    # Initialize the DeepSpeech2 model
    ds2_model = DeepSpeech2Model(vocab_size=train_generator.vocab_size,
                                 num_conv_layers=args.num_conv_layers,
                                 num_rnn_layers=args.num_rnn_layers,
                                 rnn_layer_size=args.rnn_layer_size,
                                 use_gru=args.use_gru,
                                 share_rnn_weights=args.share_rnn_weights,
                                 place=place,
                                 init_from_pretrained_model=args.init_from_pretrained_model,
                                 output_model_dir=args.output_model_dir,
                                 vocab_list=test_generator.vocab_list)

    # Get the number of training samples
    num_samples = get_data_len(args.train_manifest, args.max_duration, args.min_duration)
    print("[%s] Number of training samples: %d\n" % (datetime.now(), num_samples))

    # Start training
    ds2_model.train(train_batch_reader=train_batch_reader,
                    dev_batch_reader=test_batch_reader,
                    learning_rate=args.learning_rate,
                    gradient_clipping=400,
                    batch_size=args.batch_size,
                    num_samples=num_samples,
                    num_epoch=args.num_epoch,
                    test_off=args.test_off)

def main():
    print_arguments(args)
    train()

if __name__ == '__main__':
    main()
