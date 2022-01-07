import argparse
from .modules.nlp import NLP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text generation with transformers')
    parser.add_argument('action',
                        type=str,
                        help='"train" or "generate"')
    parser.add_argument('arg',
                        type=str,
                        default='',
                        help='Argument, can be seed if action is "generate". If "train" refers to the path to the text corpus.')
    args = parser.parse_args()

    nlp = NLP()
    if args.action == 'train':
        nlp.train(args.arg)
    elif args.action == 'generate':
        nlp.generate(seed=args.arg)
