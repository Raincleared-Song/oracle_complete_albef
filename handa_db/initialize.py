import os
import re
import opencc
import random
import shutil
from tqdm import tqdm
from utils import load_json, save_json
from peewee import IntegerField, TextField, AutoField, Model, SqliteDatabase, CharField

db = SqliteDatabase('chant_oracle_new.db')


class BaseModel(Model):
    class Meta:
        database = db


class ChantOracleBoneItems(BaseModel):
    """
    甲片条目表，每条数据代表甲片中的一个条目，(著录号-条号) 唯一对应一条数据
    """
    # 自增 id 域，主键
    id = AutoField()
    # 著录号（甲片的唯一标识），最大长度 511 字符, 原 book_name
    chant_published_collection_number = \
        CharField(null=False, max_length=511, column_name='chant_published_collection_number')
    # 汉达条号（正整数，甲片下某个甲骨文句子的序号），原 row_order
    chant_notation_number = IntegerField(null=False, column_name='chant_notation_number')
    # 汉达释文，甲骨文句子（繁体汉字），带 font 标签，原 modern_text
    chant_transcription_text = TextField(null=False, column_name='chant_transcription_text')
    # 汉达文库字体分类，最大长度 7，原 category
    chant_calligraphy = CharField(null=False, max_length=7, column_name='chant_calligraphy')
    # 源数据在汉达文库中的 url 后缀，最大长度 511 字符，原 url
    chant_url = CharField(null=False, max_length=511, column_name='chant_url')
    # 包含字形的列表，以 '\t' 分隔的字符串，每个元素都是字形表 CharShape.id
    characters = TextField(null=False, column_name='characters')
    # 甲片图的路径，最大长度 511 字符，原 l_bone_img
    chant_processed_rubbing = CharField(null=False, max_length=511, column_name='chant_processed_rubbing')
    # 汉字排布图的路径，最大长度 511 字符，原 r_bone_img
    chant_mapped_character_image = CharField(null=False, max_length=511, column_name='chant_mapped_character_image')
    # 数据来源，新增列，如有多个来源，请用 \t 分割，默认值为 chant.org
    data_sources = TextField(null=False, default='chant.org', column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 著录号-条号 联合唯一索引
        indexes = (
            (('chant_published_collection_number', 'chant_notation_number'), True),
        )


class Character(BaseModel):
    """
    标准字表，一个标准字（单字+合文）包含多个字形，(编码-字体) 不一定唯一对应一个标准字！同一个汉字可能对应多个 Character（编码不同）!
    """
    # 自增 id 域，主键
    id = AutoField()
    # 摹本中的编号，索引字段，为 -1 表示只在汉达而不在摹本中的字，原 char_index
    wzb_character_number = IntegerField(null=False, index=True, column_name='wzb_character_number')
    # 汉达字体标签 name，原 font
    chant_font_label = CharField(null=False, max_length=7, column_name='chant_font_label')
    # 《字表目录》第1列“字形”图片路径，新增列，默认为空
    standard_inscription = CharField(null=False, default="", max_length=511, column_name='standard_inscription')
    # 《字表目录》第2列“隶定”现代汉字，utf-8 编码，原 char_byte
    standard_liding_character = CharField(null=False, max_length=7,
                                          column_name='standard_liding_character')
    # 一级字头，新增列，默认为空
    first_order_standard_character = CharField(null=False, default="", max_length=511,
                                               column_name='first_order_standard_character')
    # 二级字头，新增列，默认为空
    second_order_standard_character = CharField(null=False, max_length=511,
                                                column_name='second_order_standard_character')
    # 《字表目录》第4列“页码”，原 page_number，默认为 -1
    wzb_page_number = IntegerField(null=False, default=-1, column_name='wzb_page_number')
    # 部首编号，未指定时为 -1
    wzb_radical_number = IntegerField(null=False, default=-1, column_name='wzb_radical_number')
    # 部首，新增列，默认为空
    wzb_radical = CharField(null=False, default="", max_length=7, column_name='wzb_radical')
    # 拼音，新增列，默认为空
    wzb_spelling = CharField(null=False, default="", max_length=511, column_name='wzb_spelling')
    # 笔画数，新增列，默认为 -1
    wzb_stroke_count = IntegerField(null=False, default=-1, column_name='wzb_stroke_count')
    # 摹本中该字在目录中位于目录的哪一页，-1 表示未处理或非摹本中的字
    wzb_table_page = IntegerField(null=False, default=-1, column_name='wzb_table_page')
    # 摹本中该字在目录中位于目录的哪一行，-1 表示未处理或非摹本中的字
    wzb_table_row = IntegerField(null=False, default=-1, column_name='wzb_table_column')
    # 摹本中该字在目录中位于目录的哪一栏，0 表示左边栏，1 表示右边栏，-1 表示未处理或非摹本中的字
    # TODO: 目前针对 1-4 栏的问题，如果需要将数据进行精准识别和清晰后再放入数据库
    wzb_table_col = IntegerField(null=False, default=-1, column_name='wzb_table_col')
    # 数据来源，新增列，如有多个来源，请用 \t 分割
    data_sources = TextField(null=False, column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 编码-字体 联合唯一索引
        indexes = (
            (('standard_liding_character', 'chant_font_label'), False),
        )


class CharFace(BaseModel):
    """
    字形表，对应汉达文库/李老师摹本中的每一个字形
    """
    # 自增 id 域，主键
    id = AutoField()
    # 根据 (著录号 - 无字体汉字) 进行匹配
    # 0-只在汉达中，不在摹本中；1-只在摹本中，不在汉达中；2-同时存在，数据可以对上，一一对应；3-同时存在，数据可对上，多于1条
    match_case = IntegerField(null=False, index=True, column_name='match_case')
    # 属于哪一个标准字 Character.id，索引字段，原 char_belong，可以由此索引到一级和二级字头
    standard_character_id = IntegerField(null=False, index=True, column_name='standard_character_id')
    # 属于哪一个标准字，包含了汉达/摹本的现代汉字标准字，默认为空
    standard_character = CharField(null=False, default="", max_length=7, column_name='standard_character')
    # 在汉达文库甲片图中的坐标信息，可能为空（match_case == 1），原 coords
    chant_coordinates = CharField(null=False, max_length=63, column_name='chant_coordinates')
    # 原形，即汉达文库中带背景的噪声图片路径，最大长度 511 字符，可能为空（match_case == 1），原 noise_image
    chant_authentic_face = CharField(null=False, max_length=511, column_name='chant_authentic_face')
    # 汉达条号（正整数，甲片下某个甲骨文句子的序号），原 row_order
    chant_notation_number = IntegerField(null=False, column_name='chant_notation_number')
    # 汉达中文字图片的编号，（著录号+条号+文字图片）
    chant_face_index = IntegerField(null=False, default=-1, column_name='chant_face_index')
    # 摹写字形图片路径，最大长度 511 字符，可能为空（match_case == 0），原 shape_image
    wzb_handcopy_face = CharField(null=False, max_length=511, column_name='wzb_handcopy_face')
    # 所属的著录号，最大长度 511 字符，match_case == 0/2-取汉达著录号表示，1-取摹本著录号表示，原 book_name
    published_collection_number = CharField(null=False, max_length=511, column_name='published_collection_number')
    # 李老师摹本字体分类，最大长度 7，可能为空（match_case == 0），missing 表示找不到有效 ocr 编码，原 category
    wzb_calligraphy = CharField(null=False, max_length=7, column_name='wzb_calligraphy')
    # 页码号，可能为 -1（match_case == 0），原 page_number
    wzb_page_number = IntegerField(null=False, column_name='wzb_page_number')
    # 第几行，可能为 -1（match_case == 0）
    wzb_row_number = IntegerField(null=False, column_name='wzb_row_number')
    # 第几列，可能为 -1（match_case == 0）
    wzb_col_number = IntegerField(null=False, column_name='wzb_col_number')
    # 数据来源，新增列，如有多个来源，请用 \t 分割
    data_sources = TextField(null=False, column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 汉达字形图片的联合唯一索引，著录号-条号-编号
        indexes = (
            (('published_collection_number', 'chant_notation_number', 'chant_face_index'), False),
        )


def init_db():
    db.connection()
    db.create_tables([ChantOracleBoneItems, Character, CharFace])


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
    err_log = open('output/err_db_323+324.txt', 'w', encoding='utf-8')
    index_to_chars = {int(key): val for key, val in load_json('handa/index_to_chars.json').items()}
    age_list = ['A1', 'A2', 'AS', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'AB', 'A9', 'A10', 'A11', 'A12', 'A13',
                'B1', 'B2', 'B3', 'BL', 'B4', 'B5', 'B6', 'B7', 'C1', 'C2', 'C3', 'C4', 'C5', '一']
    book_name_map = {
        '合': 'H', '合補': 'B', '東大': 'D', '屯': 'T', '英': 'Y', '日天': 'L', '花': 'HD', '懷': 'W',
    }

    def zero_padding(src: str, pad_len=5):
        src = src.lstrip('0 ')
        zid = 0
        if src.isdigit():
            zid = len(src)
        else:
            for zid in range(len(src)):
                if not src[zid].isdigit():
                    break
        if zid >= pad_len:
            return src
        else:
            return '0' * (pad_len - zid) + src

    # folder_idx = 323  # 7662
    folder_idx = 324  # 7706
    char_base = f'/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_{folder_idx}/ocr_char'
    folder_list = sorted([(folder, folder_idx) for folder in os.listdir(char_base)])
    bone_to_shapes = {}
    converter = opencc.OpenCC('t2s.json')
    for folder, ocr_idx in tqdm(folder_list, desc='ocr'):
        if int(folder) not in index_to_chars:
            err_log.write(f'{folder}\t[folder index not in table]\n')
            continue
        png_list = os.listdir(os.path.join(char_base, folder))
        cur_folder_info = []
        for o_file in png_list:
            assert o_file.endswith('.png')
            file = o_file[:-4]
            if file in ['甲骨文字編-李宗焜_0416_correct_8_3_0848_陶_(A7)包_',
                        '甲骨文字編-李宗焜_0608_correct_7_3_1449_岳_(A7)E_',
                        '甲骨文字編-李宗焜_0608_7_3_1449_岳_(A7)E_',
                        '甲骨文字編-李宗焜_1209_correct_8_3_3415_酒__屯005.']:
                err_log.write(f'{folder}\t{ocr_idx}\t[bad format]\t{file}\n')
                continue
            if not file.endswith(')'):
                file += ')'
            try:
                if ocr_idx == 323:
                    _, page, _, row, col, ch_idx, _, book_age = file.split('_')
                else:
                    _, page, row, col, ch_idx, _, book_age = file.split('_')
            except Exception as err:
                print(folder, ocr_idx, file)
                raise err
            if int(folder) != int(ch_idx):
                err_log.write(f'{folder}\t{ocr_idx}\t[ch_code not match]\t{file}\n')
                continue
            # 修改年代号的匹配规则
            age_pos, book_name, age = -1, '', ''
            for age_sample in age_list:
                age_pos = book_age.find(f'({age_sample})')
                if age_pos != -1:
                    book_name, age = book_age[:age_pos], age_sample
                    break
            if age_pos != -1 and len(book_name) == 0:
                err_log.write(f'{folder}\t{ocr_idx}\t[null book_name]\t{file}\n')
                continue
            if age_pos == -1:
                pos = book_age.find('(')
                book_name, age = book_age[:pos], 'missing'
                if len(book_name) == 0:
                    err_log.write(f'{folder}\t{ocr_idx}\t[null book_name and age code not found]\t{file}\n')
                    continue
                else:
                    err_log.write(f'{folder}\t{ocr_idx}\t[age code not found]\t{file}\n')
            assert len(book_name) > 0
            cur_folder_info.append((int(page), int(row), int(col), int(ch_idx), book_name, age, o_file))
        if len(cur_folder_info) == 0:
            err_log.write(f'{folder}\t{ocr_idx}\t[empty folder]\n')
            continue
        folder_chars = index_to_chars[int(folder)]
        for page, row, col, ch_idx, book_name, age, file in cur_folder_info:
            # 记录甲片和字形的对应关系，去除著录号前导 0
            if book_name[0].isdigit():
                book_name = 'H' + zero_padding(book_name)
            elif book_name[:1] in book_name_map:
                book_name = book_name_map[book_name[:1]] + zero_padding(book_name[1:])
                char = index_to_chars[ch_idx] if ch_idx in index_to_chars else 'null'
                err_log.write(f'[special] {book_name} {ch_idx} {char}\n')
            elif book_name[:2] in book_name_map:
                book_name = book_name_map[book_name[:2]] + zero_padding(book_name[2:])
                char = index_to_chars[ch_idx] if ch_idx in index_to_chars else 'null'
                err_log.write(f'[special] {book_name} {ch_idx} {char}\n')
            for char in folder_chars:
                char = converter.convert(char)
                if (book_name, char) not in bone_to_shapes:
                    # 最后一个 bool 表示是否与汉达匹配成功
                    bone_to_shapes[(book_name, char)] = []
                bone_to_shapes[(book_name, char)].append(
                    [page, row, col, ch_idx, age, os.path.join(folder, file), False])
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
    all_books = []

    for part in handa_parts:
        meta = load_json(f'{handa_base}/{part}/oracle_meta_{part}.json')
        for book in tqdm(meta, desc=part):
            book_name, row_order, modern_text, category, url, l_bone_img, r_bone_img = \
                book['book_name'], book['row_order'], book['modern_text'], book['category'], \
                book['url'], book['l_bone_img'], book['r_bone_img']
            l_bone_img = f'{part}/bones/{l_bone_img}'
            r_bone_img = f'{part}/bones/{r_bone_img}'
            cur_book_shape_ids = []
            for char in book['l_chars']:
                ch, coords, img = char['char'], char['coords'], f'{part}/characters/{char["img"]}'
                font = re.findall(r"<font face='([^']+)'>", ch)
                font = list(set(font))
                assert len(font) <= 1
                font = font[0] if len(font) > 0 else ''
                ch = re.sub(r'</?[^>]+>|[ ]', '', ch).strip()
                ch = converter.convert(ch)
                # 创建字形，新增对数量的检查，去除前导 0，数量不对就一个都不匹配
                key = (book_name, ch)
                if key not in bone_to_shapes or len(bone_to_shapes[key]) != book_char_to_count[(book_name, ch)]:
                    if key in bone_to_shapes:
                        err_log_match.write(f'not match: [{book_name}] [{ch}] '
                                            f'[{len(bone_to_shapes[key])}] [{book_char_to_count[(book_name, ch)]}]\n')
                    else:
                        err_log_match.write(f'not match: [{book_name}] [{ch}]\n')
                    # 首先创建单字
                    if (ch, font, -1) not in char_font_index_to_id:
                        new_ch = Character.create(wzb_character_number=-1,
                                                  modern_character=ch, chant_font_label=font)
                        char_font_index_to_id[(ch, font, -1)] = new_ch.id
                        # char_font_index_to_id[(ch, font, -1)] = cur_new_char_id
                        # cur_new_char_id += 1
                    # 考虑不匹配的情况，match_case == 0
                    match_count['match_handa'] += 1
                    new_shape = CharFace.create(
                        match_case=0,
                        liding_character=char_font_index_to_id[(ch, font, -1)],
                        chant_coordinates=coords,
                        chant_authentic_face=img,
                        wzb_handcopy_face="",
                        published_collection_number=book_name,
                        wzb_calligraphy="",
                        wzb_page_number=-1,
                        wzb_row_number=-1,
                        wzb_col_number=-1,
                    )
                else:
                    # 考虑匹配的情况，match_case == 2，按条目顺序匹配
                    candidate_list = bone_to_shapes[key]
                    if len(candidate_list) == 1:
                        all_books.append(book_name)
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
                        new_ch = Character.create(wzb_character_number=ch_idx,
                                                  modern_character=ch, chant_font_label=font)
                        char_font_index_to_id[(ch, font, ch_idx)] = new_ch.id
                        # char_font_index_to_id[(ch, font, ch_idx)] = cur_new_char_id
                        # cur_new_char_id += 1
                    new_shape = CharFace.create(
                        match_case=match_case,
                        liding_character=char_font_index_to_id[(ch, font, ch_idx)],
                        chant_coordinates=coords,
                        chant_authentic_face=img,
                        wzb_handcopy_face=shape_img,
                        published_collection_number=book_name,
                        wzb_calligraphy=age,
                        wzb_page_number=page,
                        wzb_row_number=row,
                        wzb_col_number=col,
                    )
                cur_book_shape_ids.append(str(new_shape.id))
            # 创建新甲片
            ChantOracleBoneItems.create(
                published_collection_number=book_name,
                chant_notation_number=row_order,
                chant_transcription_text=modern_text,
                chant_calligraphy=category,
                chant_url=url,
                characters='\t'.join(cur_book_shape_ids),
                chant_processed_rubbing=l_bone_img,
                chant_mapped_character_image=r_bone_img,
            )
    print(match_count)
    save_json(all_books, 'output/all_books.json')
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
                new_ch = Character.create(wzb_character_number=ch_idx, modern_character=char, chant_font_label='')
                char_font_index_to_id[(char, '', ch_idx)] = new_ch.id
            CharFace.create(
                match_case=1,
                liding_character=char_font_index_to_id[(char, '', ch_idx)],
                chant_coordinates="",
                chant_authentic_face="",
                wzb_handcopy_face=shape_img,
                published_collection_number=book_name,
                wzb_calligraphy=age,
                wzb_page_number=page,
                wzb_row_number=row,
                wzb_col_number=col,
            )
    # {'match_handa': 433321, 'match_shape': 21619, 'all_match_1': 6737, 'all_match_more': 869}
    print(match_count)
    err_log_match.close()
    print(ChantOracleBoneItems.select().count())  # 125697
    print(Character.select().count())  # 4718
    print(CharFace.select().count())  # 462546
    print('count2:', CharFace.select().where(CharFace.match_case == 2).count())  # 6737
    print(CharFace.select().where(CharFace.match_case == 3).count())  # 869
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


def gen_sharpen_dataset():
    """
    train: 6055, valid: 682
    """
    random.seed(100)
    handa_base = '/var/lib/shared_volume/data/private/songchenyang/hanzi_filter/handa'
    char_base = '/var/lib/shared_volume/home/linbiyuan/yolov5/ocr_res_png_324/ocr_char'
    train_path = ['handa_sharpen2/train/noise', 'handa_sharpen2/train/label']
    valid_path = ['handa_sharpen2/valid/noise', 'handa_sharpen2/valid/label']
    os.makedirs(train_path[0], exist_ok=True)
    os.makedirs(train_path[1], exist_ok=True)
    os.makedirs(valid_path[0], exist_ok=True)
    os.makedirs(valid_path[1], exist_ok=True)
    init_db()
    for shape in tqdm(CharFace.select().where(CharFace.match_case == 2)):
        assert isinstance(shape, CharFace)
        ch = Character.select().where(Character.id == shape.char_belong)
        assert len(ch) == 1 and isinstance(ch[0], Character)
        ch = ch[0].char_byte
        file_name = f'{shape.book_name}_{shape.page_code}_{shape.row_number}_{shape.col_number}_{ch}.png'
        if random.random() <= 0.9:
            target_path = train_path
        else:
            target_path = valid_path
        shutil.copy(os.path.join(handa_base, str(shape.noise_image)), os.path.join(target_path[0], file_name))
        shutil.copy(os.path.join(char_base, str(shape.shape_image)), os.path.join(target_path[1], file_name))


if __name__ == '__main__':
    # gen_sharpen_dataset()
    # exit()
    # check_data()
    # exit()
    # gen_book_char_stat()
    # gen_char_to_indexes()
    # exit()
    init_db()
    dump_data()
