import json
import os
def load_file_list_recursion(fpath, result):
    allfilelist = os.listdir(fpath)
    for file in allfilelist:
        filepath = os.path.join(fpath, file)
        if os.path.isdir(filepath):
            load_file_list_recursion(filepath, result)
        else:
            result.append(filepath)
            print(len(result))



def scan(input_path, out_put):
    result_list = []
    load_file_list_recursion(input_path, result_list)
    result_list.sort()

    for i in range(len(result_list)):
        print('{}_{}'.format(i, result_list[i]))

    with open(out_put, 'w') as j:
        json.dump(result_list, j)


scan('/Users/xiaoguangli/lxg/CV/publication/ICCV_2023/final_version/code/Inpainting_pretraining/data/masks', './mask.txt')
scan('/Users/xiaoguangli/lxg/CV/publication/ICCV_2023/final_version/code/Inpainting_pretraining/data/images', './image.txt')