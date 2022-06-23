import json


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path, 'r', encoding='utf-8')
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w', encoding='utf-8')
    json.dump(obj, file)
    file.close()


def print_json(obj, file=None):
    print(json.dumps(obj, indent=4, separators=(', ', ': '), ensure_ascii=False), file=file)
