import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', help='model name', type=str)
    args = parser.parse_args()

    config_name = 'Image_Reconstruct' if 'image_reconstruct' in args.task else 'Finetune_single_mlm'
    basic_config = yaml.load(open(f'./configs/{config_name}.yaml'), yaml.Loader)
    extra_config = yaml.load(open(f'./output/{args.task}/config.yaml'), yaml.Loader)
    for key, val in basic_config.items():
        if key in extra_config:
            if val != extra_config[key]:
                print('******', end=' ')
            print(key + ':', extra_config[key])
        else:
            print('++++++', key + ':', val)


if __name__ == '__main__':
    main()
