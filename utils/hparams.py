# https://docs.python.org/zh-cn/3/library/argparse.html
import argparse
import os
import yaml

global_print_hparams = True
# MUMIDI: haparams for  (1) date process, (2) mumidi representation
hparams = {}

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v

def set_hparams(config='', exp_name='', hparams_str='',print_hparams=True, global_hparams=True):
    if config == '':
        print("hprams empty")
        # basic hprams
        parser = argparse.ArgumentParser(description='XAI-Lyricist')  # Create an argparser object
        parser.add_argument('-config', type=str, default='', help='location of yaml file') # 添加参数
        parser.add_argument('--exp_name', type=str, default='', help='exp_name')
        parser.add_argument('--hparams', type=str, default='', help='location of the data corpus')
        # parser.add_argument('--parse', type=str, default='', help='the parsing strategy of lyric words')
        # model training hprams
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--validate', action='store_true', help='validate')
        parser.add_argument('--prompt', action='store_true', help='validate')
        parser.add_argument('--gen_num', type=int, default=1, help='generate number from sketch')

        # 解析部分已知命令行参数（args），而将其余的未知/没定义的参数继续传递给另一个脚本或程序（unknown）
        args, unknown = parser.parse_known_args()
        print(f"unknown: {unknown}")
    # 函数传参数
    else:
        print("hparams not empty")
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str,
                    infer=False, validate=False, reset=False, debug=False, prompt=False,gen_num=1)

    #hparams for data prepare
    global hparams
    # print(f"args:\n{args}")
    assert args.config != ''
    with open(args.config) as f:
        hparams_ = yaml.safe_load(f)  ## load yaml files

    hparams_['infer'] = args.infer
    # hparams_['debug'] = args.debug
    hparams_['validate'] = args.validate
    # prompt
    hparams_['prompt'] = args.prompt
    hparams_['gen_num'] = args.gen_num
    # hparams_['parse'] = args.parse

    hparams.update(hparams_)
    print(hparams)

    return hparams_


