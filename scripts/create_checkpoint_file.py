
import argparse
import glob
import os

def main():
    """Write a checkpoint file for a model. It allows us to use the model on a new computer by customizing the paths to that computer.

    :returns: 
    :rtype: 

    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    assert(os.path.isdir(args.model))
    args.model = os.path.realpath(args.model)
    model_paths = glob.glob(args.model + '/esp-model*meta')
    model_names = sorted([_.rstrip('.meta') for _ in model_paths], key=lambda x: int(x.split('-')[-1]))
    with open(args.model + '/checkpoint', 'w') as f:
        f.write('model_checkpoint_path: "{}"\n'.format(model_names[-1]))
        for model_name in model_names:
            f.write('all_model_checkpoint_paths: "{}"\n'.format(model_name))

if __name__ == '__main__':
    main()
