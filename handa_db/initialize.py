import os
import re
import opencc
from tqdm import tqdm
from utils import load_json, save_json
from peewee import IntegerField, TextField, AutoField, Model, SqliteDatabase, CharField

db = SqliteDatabase('handa_oracle.db')


class BaseModel(Model):
    class Meta:
        database = db


class BoneRow(BaseModel):
    """
    甲片条目表，每条数据代表甲片中的一个条目，(著录号-条号) 唯一对应一条数据
    """
    # 自增 id 域，主键
    id = AutoField()
    # 著录号（甲片的唯一标识），最大长度 511 字符
    book_name = CharField(null=False, max_length=511, column_name='book_name')
    # 条号（正整数，甲片下某个甲骨文句子的序号）
    row_order = IntegerField(null=False, column_name='row_order')
    # 释文，甲骨文句子（繁体汉字），带 font 标签
    modern_text = TextField(null=False, column_name='modern_text')
    # 汉达文库类别，最大长度 7
    category = CharField(null=False, max_length=7, column_name='category')
    # 源数据在汉达文库中的 url 后缀，最大长度 511 字符
    url = CharField(null=False, max_length=511, column_name='url')
    # 包含单字的列表，以 '\t' 分隔的字符串，每个元素都是字形表 CharShape.id
    characters = TextField(null=False, column_name='characters')
    # 甲片图的路径，最大长度 511 字符
    l_bone_img = CharField(null=False, max_length=511, column_name='l_bone_img')
    # 汉字排布图的路径，最大长度 511 字符
    r_bone_img = CharField(null=False, max_length=511, column_name='r_bone_img')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 著录号-条号 联合唯一索引
        indexes = (
            (('book_name', 'row_order'), True),
        )


class Character(BaseModel):
    """
    单字表，一个单字包含多个字形，(编码-字体) 不一定唯一对应一个单字！同一个汉字可能对应多个 Character（编码不同）!
    """
    # 自增 id 域，主键
    id = AutoField()
    # 摹本中的编号，索引字段，为 -1 表示只在汉达而不在摹本中的字
    char_index = IntegerField(null=False, index=True, column_name='char_index')
    # 对应的汉字，utf-8 编码
    char_byte = CharField(null=False, max_length=7, column_name='char_byte')
    # 字体标签 name
    font = CharField(null=False, max_length=7, column_name='font')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 编码-字体 联合唯一索引
        indexes = (
            (('char_byte', 'font'), False),
        )


class CharShape(BaseModel):
    """
    字形表，对应汉达文库/李老师摹本中的每一个单字字形
    """
    # 自增 id 域，主键
    id = AutoField()
    # 根据 (著录号 - 无字体汉字) 进行匹配
    # 0-只在汉达中，不在摹本中；1-只在摹本中，不在汉达中；2-同时存在，数据可以对上，一一对应；3-同时存在，数据可对上，多于1条
    match_case = IntegerField(null=False, index=True, column_name='match_case')
    # 属于哪一个单字 Character.id，索引字段
    char_belong = IntegerField(null=False, index=True, column_name='char_belong')
    # 在汉达文库甲片图中的坐标信息，可能为空（match_case == 1）
    coords = CharField(null=False, max_length=63, column_name='coords')
    # 汉达文库中带背景的噪声图片路径，最大长度 511 字符，可能为空（match_case == 1）
    noise_image = CharField(null=False, max_length=511, column_name='noise_image')
    # 摹写字形图片路径，最大长度 511 字符，可能为空（match_case == 0）
    shape_image = CharField(null=False, max_length=511, column_name='shape_image')
    # 所属的著录号，最大长度 511 字符，match_case == 0/2-取汉达著录号表示，1-取摹本著录号表示
    book_name = CharField(null=False, max_length=511, column_name='book_name')
    # 李老师摹本类别码，最大长度 7，可能为空（match_case == 0）
    category = CharField(null=False, max_length=7, column_name='category')
    # 页码号，可能为 -1（match_case == 0）
    page_code = IntegerField(null=False, column_name='page_code')
    # 第几行，可能为 -1（match_case == 0）
    row_number = IntegerField(null=False, column_name='row_number')
    # 第几列，可能为 -1（match_case == 0）
    col_number = IntegerField(null=False, column_name='col_number')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')


def init_db():
    db.connection()
    db.create_tables([BoneRow, Character, CharShape])


def gen_book_char_stat():
    handa_base = '/var/lib/shared_volume/data/private/songchenyang/hanzi_filter/handa'
    handa_parts = ['B', 'D', 'H', 'L', 'S', 'T', 'W', 'Y', 'HD']
    book_char_to_count = {}
    converter = opencc.OpenCC('t2s.json')

    for part in handa_parts:
        meta = load_json(f'{handa_base}/{part}/oracle_meta_{part}.json')
        for book in tqdm(meta, desc=part):
            book_name = book['book_name']
            for char in book['l_chars']:
                ch = re.sub(r'</?[^>]+>|[ ]', '', char['char']).strip()
                ch = converter.convert(ch)
                if (book_name, ch) not in book_char_to_count:
                    book_char_to_count[(book_name, ch)] = 0
                book_char_to_count[(book_name, ch)] += 1
    book_char_to_count = [(book_name, ch, count) for (book_name, ch), count in book_char_to_count.items()]
    save_json(book_char_to_count, 'output/book_char_to_count.json')


def gen_char_to_indexes():
    # ['上甲', '六牡', '四十', '小臧', '我王']
    # char_to_indexes = {key: val for key, val in load_json('handa/char_to_indexes.json').items()
    #                    if len(key) > 0 and '【' not in key}
    # li = sorted(list(char_to_indexes.keys()))
    # print(li)
    # exit()

    with open('handa/char_no_3_copy_new.txt', 'r', encoding='utf-8') as fin:
        lines = [line.strip() for line in fin.readlines()]
    char_to_indexes = {}
    for line in lines:
        if len(line) > 0 and line[0].isdigit():
            line = '-' + line
        if len(line) > 0 and line[0] == 'X':
            line = '【X】' + line[1:]
        assert len(line) == 0 or line[0] in ['-'] or ord(line[0]) > 255, line
        tokens = line.split('-')
        if len(tokens) not in [2, 3] or line[0] == '【' and line[-1] == '】':
            continue
        try:
            ch, idx = tokens[0].strip(), int(tokens[1])
        except ValueError as err:
            print(line)
            raise err
        if ch not in char_to_indexes:
            char_to_indexes[ch] = []
        char_to_indexes[ch].append(idx)
    save_json(char_to_indexes, 'handa/char_to_indexes.json')
    # 1386 865
    print(len(char_to_indexes), len([ch for ch in char_to_indexes.keys() if len(ch) == 1]))

    char_to_indexes_fil = {key: val for key, val in char_to_indexes.items() if len(key) > 0 and '【' not in key}
    idx_to_char = {}
    for key, val in char_to_indexes_fil.items():
        for idx in val:
            if idx not in idx_to_char:
                idx_to_char[idx] = []
            idx_to_char[idx].append(key)
    for key, val in idx_to_char.items():
        if len(val) > 1:
            print(key, val)
    idx_to_char = {key: val for key, val in sorted(idx_to_char.items(), key=lambda x: x[0])}
    save_json(idx_to_char, 'handa/index_to_chars.json')


def dump_data():
    """
    将现有数据导入数据库
    """
    global db
    err_log = open('output/err_db_323.txt', 'w', encoding='utf-8')
    index_to_chars = {int(key): val for key, val in load_json('handa/index_to_chars.json').items()}

    # char_base = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_37/ocr_char/'
    # char_base = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_392/ocr_char'
    # char_base = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_3223/ocr_char'  # 5007
    char_base = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_323/ocr_char'  # 6737
    folder_list = sorted(os.listdir(char_base), key=lambda x: int(x))
    bone_to_shapes = {}
    converter = opencc.OpenCC('t2s.json')
    for folder in tqdm(folder_list, desc='ocr'):
        if int(folder) not in index_to_chars:
            err_log.write(f'{folder}\t[folder index not in table]\n')
            continue
        rel_path = os.path.join(char_base, folder)
        png_list = os.listdir(rel_path)
        cur_folder_info = []
        for file in png_list:
            assert file.endswith('.png')
            file = file[:-4]
            if file in ['甲骨文字編-李宗焜_0416_correct_8_3_0848_陶_(A7)包_',
                        '甲骨文字編-李宗焜_0608_correct_7_3_1449_岳_(A7)E_',
                        '甲骨文字編-李宗焜_1209_correct_8_3_3415_酒__屯005.']:
                err_log.write(f'{folder}\t[bad format]\t{file}\n')
                continue
            if not file.endswith(')'):
                file += ')'
            try:
                _, page, _, row, col, ch_idx, _, book_age = file.split('_')
            except Exception as err:
                print(folder, file)
                raise err
            pos, pos_r = book_age.find('('), book_age.find(')')
            if int(folder) != int(ch_idx):
                err_log.write(f'{folder}\t[ch_code not match]\t{file}\n')
                continue
            if book_age.find('(', pos + 1) != -1 or book_age.find(')', pos_r + 1) != -1:
                err_log.write(f'{folder}\t[double brackets]\t{file}\n')
                continue
            book_name, age = book_age[:pos], book_age[(pos + 1):-1]
            cur_folder_info.append((int(page), int(row), int(col), int(ch_idx), book_name, age, file))
        if len(cur_folder_info) == 0:
            err_log.write(f'{folder}\t[empty folder]\n')
            continue
        folder_chars = index_to_chars[int(folder)]
        for page, row, col, ch_idx, book_name, age, file in cur_folder_info:
            # 记录甲片和字形的对应关系，去除著录号前导 0
            book_name = book_name.strip().lstrip('0 ')
            for char in folder_chars:
                char = converter.convert(char)
                if (book_name, char) not in bone_to_shapes:
                    # 最后一个 bool 表示是否与汉达匹配成功
                    bone_to_shapes[(book_name, char)] = []
                bone_to_shapes[(book_name, char)].append(
                    [page, row, col, ch_idx, age, os.path.join(rel_path, file), False])
    save_json([key + tuple(val) for key, val in bone_to_shapes.items()], 'output/bone_to_shapes.json')
    err_log.close()

    handa_base = '/var/lib/shared_volume/data/private/songchenyang/hanzi_filter/handa'
    handa_parts = ['H', 'B', 'D', 'L', 'S', 'T', 'W', 'Y', 'HD']
    match_count = {'match_handa': 0, 'match_shape': 0, 'all_match_1': 0, 'all_match_more': 0}
    book_char_to_count = {(book_name, ch): count for
                          book_name, ch, count in load_json('output/book_char_to_count.json')}
    char_font_index_to_id = {}
    err_log_match = open('output/err_db_match.txt', 'w', encoding='utf-8')

    # cur_new_char_id = 0

    for part in handa_parts:
        meta = load_json(f'{handa_base}/{part}/oracle_meta_{part}.json')
        for book in tqdm(meta, desc=part):
            book_name, row_order, modern_text, category, url, l_bone_img, r_bone_img = \
                book['book_name'], book['row_order'], book['modern_text'], book['category'], \
                book['url'], book['l_bone_img'], book['r_bone_img']
            l_bone_img = f'{handa_base}/{part}/bones/{l_bone_img}'
            r_bone_img = f'{handa_base}/{part}/bones/{r_bone_img}'
            cur_book_shape_ids = []
            for char in book['l_chars']:
                ch, coords, img = char['char'], char['coords'], f'{handa_base}/{part}/characters/{char["img"]}'
                font = re.findall(r"<font face='([^']+)'>", ch)
                font = list(set(font))
                assert len(font) <= 1
                font = font[0] if len(font) > 0 else ''
                ch = re.sub(r'</?[^>]+>|[ ]', '', ch).strip()
                ch = converter.convert(ch)
                # 创建字形，新增对数量的检查，去除前导 0，数量不对就一个都不匹配
                key = (book_name[1:].lstrip('0 '), ch) if part == 'H' else ''
                if key == '' or key not in bone_to_shapes or \
                        len(bone_to_shapes[key]) != book_char_to_count[(book_name, ch)]:
                    if key in bone_to_shapes:
                        err_log_match.write(f'not match: [{book_name}] [{ch}] '
                                            f'[{len(bone_to_shapes[key])}] [{book_char_to_count[(book_name, ch)]}]\n')
                    else:
                        err_log_match.write(f'not match: [{book_name}] [{ch}]\n')
                    # 首先创建单字
                    if (ch, font, -1) not in char_font_index_to_id:
                        new_ch = Character.create(char_index=-1, char_byte=ch, font=font)
                        char_font_index_to_id[(ch, font, -1)] = new_ch.id
                        # char_font_index_to_id[(ch, font, -1)] = cur_new_char_id
                        # cur_new_char_id += 1
                    # 考虑不匹配的情况，match_case == 0
                    match_count['match_handa'] += 1
                    new_shape = CharShape.create(
                        match_case=0,
                        char_belong=char_font_index_to_id[(ch, font, -1)],
                        coords=coords,
                        noise_image=img,
                        shape_image="",
                        book_name=book_name,
                        category="",
                        page_code=-1,
                        row_number=-1,
                        col_number=-1,
                    )
                else:
                    # 考虑匹配的情况，match_case == 2，按条目顺序匹配
                    candidate_list = bone_to_shapes[key]
                    if len(candidate_list) == 1:
                        match_count['all_match_1'] += 1
                        match_case = 2
                    else:
                        match_count['all_match_more'] += 1
                        match_case = 3
                    candidate = None
                    for candidate in candidate_list:
                        if not candidate[-1]:
                            break
                    assert candidate is not None and not candidate[-1]
                    candidate[-1] = True
                    page, row, col, ch_idx, age, shape_img, _ = candidate
                    # 首先创建单字
                    if (ch, font, ch_idx) not in char_font_index_to_id:
                        new_ch = Character.create(char_index=ch_idx, char_byte=ch, font=font)
                        char_font_index_to_id[(ch, font, ch_idx)] = new_ch.id
                        # char_font_index_to_id[(ch, font, ch_idx)] = cur_new_char_id
                        # cur_new_char_id += 1
                    new_shape = CharShape.create(
                        match_case=match_case,
                        char_belong=char_font_index_to_id[(ch, font, ch_idx)],
                        coords=coords,
                        noise_image=img,
                        shape_image=shape_img,
                        book_name=book_name,
                        category=age,
                        page_code=page,
                        row_number=row,
                        col_number=col,
                    )
                cur_book_shape_ids.append(str(new_shape.id))
            # 创建新甲片
            BoneRow.create(
                book_name=book_name,
                row_order=row_order,
                modern_text=modern_text,
                category=category,
                url=url,
                characters='\t'.join(cur_book_shape_ids),
                l_bone_img=l_bone_img,
                r_bone_img=r_bone_img,
            )
    print(match_count)
    # exit()

    # 处理剩下的字形, match_case == 1
    for book_name, char in tqdm(bone_to_shapes.keys(), desc='last'):
        check_sum = sum(int(candidate[-1]) for candidate in bone_to_shapes[(book_name, char)])
        assert check_sum == 0 or check_sum == len(bone_to_shapes[(book_name, char)])
        if check_sum == len(bone_to_shapes[(book_name, char)]):
            continue
        for page, row, col, ch_idx, age, shape_img, flag in bone_to_shapes[(book_name, char)]:
            assert not flag
            match_count['match_shape'] += 1
            # 创建新字
            if (char, '', ch_idx) not in char_font_index_to_id:
                new_ch = Character.create(char_index=ch_idx, char_byte=char, font='')
                char_font_index_to_id[(char, '', ch_idx)] = new_ch.id
            CharShape.create(
                match_case=1,
                char_belong=char_font_index_to_id[(char, '', ch_idx)],
                coords="",
                noise_image="",
                shape_image=shape_img,
                book_name=book_name,
                category=age,
                page_code=page,
                row_number=row,
                col_number=col,
            )
    # {'match_handa': 433321, 'match_shape': 21619, 'all_match_1': 6737, 'all_match_more': 869}
    print(match_count)
    err_log_match.close()
    print(BoneRow.select().count())  # 125697
    print(Character.select().count())  # 4718
    print(CharShape.select().count())  # 462546
    print('count2:', CharShape.select().where(CharShape.match_case == 2).count())  # 6737
    print(CharShape.select().where(CharShape.match_case == 3).count())  # 869
    save_json([key + (val,) for key, val in char_font_index_to_id.items()], 'output/char_font_index_to_id.json')


def check_data():
    fin = open('output/finetune_single_mlm_np/log_case_test_52.txt', 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    data = load_json('../hanzi_filter/handa/data_filter_sim_test.json')
    fout = open('output/finetune_single_mlm_np/log_case_test_52_book.txt', 'w', encoding='utf-8')
    cur_idx = 0
    for line in lines:
        if not line[0].isdigit():
            fout.write(line)
        else:
            tokens = line.strip().split('\t')
            assert int(tokens[0]) == cur_idx
            book = data[cur_idx]
            book_order = book['book_name'] + '-' + str(book['row_order'])
            fout.write(f'{cur_idx}\t{book_order}    {tokens[1]}\n')
            cur_idx += 1
    fout.close()
    exit()

    handa_base = '/var/lib/shared_volume/data/private/songchenyang/hanzi_filter/handa'
    handa_parts = ['B', 'D', 'H', 'L', 'S', 'T', 'W', 'Y', 'HD']
    for part in handa_parts:
        meta = load_json(f'{handa_base}/{part}/oracle_meta_{part}.json')
        for book in meta:
            l_chars, r_chars = book['l_chars'], book['r_chars']
            assert len(l_chars) == len(r_chars)
            for l_char, r_char in zip(l_chars, r_chars):
                l_ch, r_ch = l_char['char'], r_char['char']
                l_ch = re.sub(r'</?[^>]+>|[ ]', '', l_ch).strip()
                assert l_ch == r_ch
                assert l_char['coords'] == r_char['coords']
                assert l_char['img'] == r_char['img']


def get_char_to_index():
    with open('output/char_no_fast_2.txt', encoding='utf-8') as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    for _ in lines:
        pass


if __name__ == '__main__':
    check_data()
    exit()
    # gen_book_char_stat()
    # gen_char_to_indexes()
    # exit()
    init_db()
    dump_data()
