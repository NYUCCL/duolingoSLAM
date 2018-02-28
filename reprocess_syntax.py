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
import spacy


def main():

    parser = argparse.ArgumentParser(
        description='Reprocess Duolingo data using Syntaxnet')

    parser.add_argument('--lang', help='2-letter code of language being learned',
                        required=True)
    parser.add_argument('--url', help='URL for syntaxnet API',
                        required=True)
    parser.add_argument('--original', help='Original data file name',
                        required=True)
    parser.add_argument('--new', help='New data file name')
    args = parser.parse_args()

    if not args.new:
        args.new = args.original + '.new'

    if args.lang == 'en':
        nlp = spacy.load('en')
        lookup = None
    elif args.lang == 'es':
        from spacy.lang.es.lemmatizer import LOOKUP
        lookup = LOOKUP
        nlp = None
    elif args.lang == 'fr':
        from spacy.lang.fr.lemmatizer import LOOKUP
        lookup = LOOKUP
        nlp = None

    new_lines = []
    ex_lines = []
    exercise = 0
    cache_dict = {}
    with open(args.original, 'r') as f:
        for line in f:
            if line[0] == '#':
                new_lines.append(line)
            elif len(line.strip()) == 0:
                ex_lines = reprocess_lines(ex_lines, args.url, args.lang,
                                           cache_dict, nlp=nlp, lookup=lookup)
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


def reprocess_lines(ex_lines, url, lang, cache_dict, nlp=None, lookup=None):
    old_lines = [l.strip().split() for l in ex_lines]
    syntaxnet_input = [l[1] for l in old_lines]
    cache_key = " ".join(syntaxnet_input)
    if cache_key in cache_dict:
        cached_lines = cache_dict[cache_key]
    else:
        if lang == 'en':
            spacy_string = syntaxnet_input[0]
            for i in range(1, len(syntaxnet_input)):
                added_string = ' '
                if len(syntaxnet_input[i-1]) > 1 and syntaxnet_input[i-1][-1] == "'":
                    added_string = ''
                if syntaxnet_input[i-1][-1] == "-":
                    added_string = ''
                if syntaxnet_input[i][0] in ["'", "-"]:
                    added_string = ''
                spacy_string = spacy_string + added_string + syntaxnet_input[i]

            spacy_out = nlp(spacy_string)
            spacy_out_list = []
            offset = 0
            try:
                for i in range(len(syntaxnet_input)):
                    j = i + offset
                    sp = spacy_out[j].lemma_ if spacy_out[j].lemma_ != '-PRON-' else str(spacy_out[j])
                    spacy_out_list.append(sp)
                    if syntaxnet_input[i] != 'vs.' and syntaxnet_input[i][-1] in ["?", '.']:
                        offset += 1
                    if syntaxnet_input[i] == 'cannot':
                        offset += 1
                assert len(syntaxnet_input) == len(spacy_out_list)
            except:
                print(syntaxnet_input)
                print(spacy_out)
                if len(syntaxnet_input) > len(spacy_out_list):
                    for i in range(len(spacy_out_list), len(syntaxnet_input)):
                        spacy_out_list.append(syntaxnet_input[i])
        # else:
        #     spacy_out_list = []

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
            word = old_line[1]
            line.append(word)
            pos = new_dict['fpos'].split('+')[0]
            if lang == 'en':
                line.append(spacy_out_list[i])
            else:
                try:
                    root = lookup[word.lower()]
                    if pos not in ['VERB', 'AUX'] and word[-1] != 'r' and root[-1] == 'r':
                        if word[-1] == 's':
                            root = word[:-1]
                        else:
                            root = word
                except:
                    root = word
                line.append(root)
            line.append(pos)
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
            line = "{:<14}{:<14}{:<8}{:<72}{:<13}{}".format(*line)
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
