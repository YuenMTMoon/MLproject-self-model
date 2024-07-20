import os
import sys
import json
import wave
import argparse
import functools
from tqdm import tqdm
from collections import Counter

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.utility import add_arguments, print_arguments, change_rate
from data_utils.utility import read_manifest

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('annotation_path', str, 'dataset/annotation/', 'Path to the annotation files')
add_arg('manifest_prefix', str, 'dataset/', 'Prefix for the training data manifest, including audio paths and annotations')
add_arg('vocab_path', str, './dataset/en_vocab.txt', 'Path to the generated data vocabulary file')
add_arg('count_threshold', int, 0, 'Character count threshold, 0 means no limit')
add_arg('is_change_frame_rate', bool, True, 'Whether to uniformly change audio to 16000Hz, this will consume a lot of time')
args = parser.parse_args()


# Filter out non-text characters
def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str + in_str[i]
        else:
            out_str = out_str + ' '
    return ''.join(out_str.split())


# Check if it's a text character
def is_uchar(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    if u'\u0030' <= uchar <= u'\u0039':
        return False
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return False
    if uchar in ('-', ',', '.', '>', '?'):
        return False
    return False


# Create data manifest
def create_manifest(annotation_path, manifest_path_prefix):
    json_lines = []
    durations = []
    # Get all annotation files
    for annotation_text in os.listdir(annotation_path):
        print('Creating data manifest for %s, please wait...' % annotation_text)
        annotation_text = os.path.join(annotation_path, annotation_text)
        # Read the annotation file
        with open(annotation_text, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path = line.split('\t')[0]
            try:
                # Filter out illegal characters
                text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', ''))
                # Adjust audio format and save
                if args.is_change_frame_rate:
                    change_rate(audio_path)
                # Get the length of the audio
                f_wave = wave.open(audio_path, "rb")
                duration = f_wave.getnframes() / f_wave.getframerate()
                durations.append(duration)
                json_lines.append(
                    json.dumps(
                        {
                            'audio_filepath': audio_path,
                            'duration': duration,
                            'text': text
                        },
                        ensure_ascii=False))
            except Exception as e:
                print(e)
                continue

    # Write the audio path, length, and label to the data manifest
    f_train = open(os.path.join(manifest_path_prefix, 'manifest.train'), 'w', encoding='utf-8')
    f_test = open(os.path.join(manifest_path_prefix, 'manifest.test'), 'w', encoding='utf-8')
    for i, line in enumerate(json_lines):
        if i % 500 == 0:
            f_test.write(line + '\n')
        else:
            f_train.write(line + '\n')
    f_train.close()
    f_test.close()
    print("Data manifest creation completed, total data is [%d] hours!" % int(sum(durations) / 3600))


# Create noise data manifest
def create_noise(path='dataset/audio/noise'):
    if not os.path.exists(path):
        print('Noise audio files are empty, skipping!')
        return
    json_lines = []
    print('Creating noise data manifest, path: %s, please wait...' % path)
    for file in tqdm(os.listdir(path)):
        audio_path = os.path.join(path, file)
        try:
            # Noise label can be marked as empty
            text = ""
            # Adjust audio format and save
            if args.is_change_frame_rate:
                change_rate(audio_path)
            f_wave = wave.open(audio_path, "rb")
            duration = f_wave.getnframes() / f_wave.getframerate()
            json_lines.append(
                json.dumps(
                    {
                        'audio_filepath': audio_path,
                        'duration': duration,
                        'text': text
                    },
                    ensure_ascii=False))
        except:
            continue
    with open(os.path.join(args.manifest_prefix, 'manifest.noise'), 'w', encoding='utf-8') as f_noise:
        for json_line in json_lines:
            f_noise.write(json_line + '\n')


# Count characters
def count_manifest(counter, manifest_path):
    manifest_jsons = read_manifest(manifest_path)
    for line_json in manifest_jsons:
        for char in line_json['text']:
            counter.update(char)


# Create data vocabulary
def build_vocab():
    counter = Counter()
    # Get all data manifests
    manifest_paths = [path for path in args.manifest_prefix.split(',')]
    # Get characters from all data manifests
    for manifest_path in manifest_paths:
        count_manifest(counter, manifest_path)
    # Generate an ID for each character
    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with open(args.vocab_path, 'w', encoding='utf-8') as fout:
        for char, count in count_sorted:
            # Skip characters beyond the specified threshold
            if count < args.count_threshold: break
            fout.write(char + '\n')
    print('Data vocabulary has been generated and saved at: %s' % args.vocab_path)


def main():
    print_arguments(args)
    create_noise()
    create_manifest(annotation_path=args.annotation_path, manifest_path_prefix=args.manifest_prefix)
    build_vocab()


if __name__ == '__main__':
    main()
