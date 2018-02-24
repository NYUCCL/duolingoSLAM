"""
Script to reparse Duolingo data files using google Syntaxnet to remove
errors surrounding punctuation.

To run, first pull the ljm625/syntaxnet-rest-api:dragnn Docker image and run
the server as described in
https://github.com/ljm625/syntaxnet-rest-api/tree/dragnn

Don't forget to set the server to use the language of interest, e.g.:
http://localhost:9000/api/vi/use/English

Then run the script from a terminal, passing the url for the server. For
example:

python reprocess_syntax.py --original data/data_en_es/en_es.slam.20171218.dev
                           --url http://localhost:9000/api/v1/query
"""

import requests
import argparse


def main():

    parser = argparse.ArgumentParser(
        description='Reprocess Duolingo data using Syntaxnet')
    parser.add_argument('--url', help='URL for syntaxnet API',
                        required=True)
    parser.add_argument('--original', help='Original data file name',
                        required=True)
    parser.add_argument('--new', help='New data file name')
    args = parser.parse_args()

    if not args.new:
        args.new = args.original + '.new'

    new_lines = []
    ex_lines = []
    exercise = 0
    cache_dict = {}
    with open(args.original, 'r') as f:
        for line in f:
            if line[0] == '#':
                new_lines.append(line)
            elif len(line.strip()) == 0:
                ex_lines = reprocess_lines(ex_lines, args.url, cache_dict)
                new_lines.extend(ex_lines)
                new_lines.append(line)
                exercise += 1
                if exercise % 1000 == 0:
                    print('Exercise: ', exercise)
                    print('Total lines:', len(new_lines))
                    print('Unique Exercises: ', len(cache_dict))
                ex_lines = []
            else:
                ex_lines.append(line)
    with open(args.new, 'w') as f:
        f.writelines(new_lines)


def reprocess_lines(ex_lines, url, cache_dict):
    old_lines = [l.strip().split() for l in ex_lines]
    syntaxnet_input = [l[1] for l in old_lines]
    cache_key = " ".join(syntaxnet_input)
    if cache_key in cache_dict:
        cached_lines = cache_dict[cache_key]
    else:
        num_punct = 0
        punct_locs = []
        for i, l in enumerate(old_lines):
            if l[2] == 'PUNCT':
                punct = l[3][-1]
                if punct != "+":
                    syntaxnet_input.insert(i + num_punct, punct)
                    punct_locs.append(i + num_punct)
                    num_punct += 1
        syntaxnet_input_string = ' '.join(syntaxnet_input)
        r = requests.post(url, json={"strings": [syntaxnet_input_string],
                                     "tree": False})
        output_dicts_withpunct = r.json()[0]['output']
        output_dicts = []
        for i, o in enumerate(output_dicts_withpunct):
            if i not in punct_locs:
                head_offset = sum([p < o['head'] for p in punct_locs])
                # remove punctuation offset from head
                # then add 1 because Duolingo dependency head is 1-indexed
                o['head'] = o['head'] - head_offset + 1
                output_dicts.append(o)
        cached_lines = []
        for i in range(len(old_lines)):
            old_line = old_lines[i]
            new_dict = output_dicts[i]
            line = []
            line.append(old_line[1])
            line.append(new_dict['fpos'].split('+')[0])
            morph_features_list = []
            skip_keys = ['break_level', 'category', 'dep',
                         'head', 'label', 'pos_tag', 'word']
            for key, value in new_dict.items():
                if key not in skip_keys:
                    morph_features_list.append(key + '=' + value)
            line.append('|'.join(morph_features_list))
            line.append(new_dict['dep'])
            line.append(new_dict['head'])
            for j in range(len(line)-1):
                line[j] = str(line[j]) + "  "
            line = "{:<14}{:<8}{:<72}{:<13}{}".format(*line)
            cached_lines.append(line)
        cache_dict[cache_key] = cached_lines
    new_lines = []
    for i in range(len(old_lines)):
        old_line = old_lines[i]
        line = cached_lines[i]
        if len(old_line) == 7:
            new_lines.append("{:<14}{}  {}\n".format(old_line[0],
                                                     line, old_line[6]))
        else:
            new_lines.append("{:<14}{}\n".format(old_line[0], line))
    return new_lines


if __name__ == '__main__':
    main()
