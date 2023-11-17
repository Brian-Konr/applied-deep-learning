import json
import argparse
from tw_rouge import get_rouge


def main(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            preds[line['id']] = line['title'].strip() + '\n'

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    results = get_rouge(preds, refs)
    print(json.dumps(results, indent=2))
    # save results
    with open(args.output, 'w') as file:
        json.dump(results, file, indent=2)
    # print(json.dumps(get_rouge(preds, refs), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    main(args)
