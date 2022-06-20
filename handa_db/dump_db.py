from initialize import *


def zero_padding(src: str, pad_len=5) -> str:
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


def book_name_convert(wzb_book_name: str, book_name_map: dict) -> str:
    """《甲骨文字编》著录号转汉达著录号"""
    if wzb_book_name[0].isdigit():
        return 'H' + zero_padding(wzb_book_name)
    pos = -1
    for pos in range(len(wzb_book_name)):
        if wzb_book_name[pos].isdigit():
            break
    if not (pos > 0 and wzb_book_name[pos].isdigit()):
        return 'error: invalid WZB format'
    wzb_src, suffix = wzb_book_name[:pos], wzb_book_name[pos:]
    if wzb_src in book_name_map:
        return book_name_map[wzb_src] + zero_padding(suffix)
    else:
        return 'error: invalid WZB source'


def dump_data_new():
    """
    将现有数据导入数据库
    """
    err_log = open('output/err_db_060616.txt', 'w', encoding='utf-8')
    wzb_chant_map = {
        '合': 'H', '合補': 'B', '東大': 'D', '屯': 'T', '英': 'Y', '日天': 'L', '花': 'HD', '懷': 'W',
    }

    # sample: 甲骨文字編-李宗焜_0002_0001_人_2_[26907正(A11)].png
    # format: 甲骨文字編-李宗焜_页数_行数_个数_序号_汉字_字义_著录号.png
    char_base = '/var/lib/shared_volume/home/linbiyuan/corpus/wenbian/labels_页数+序号+著录号+字形_校对版_060616/char'

    print('loading from char_base:', char_base)
    bone_to_shapes = {}
    png_list = sorted(os.listdir(char_base))
    for img in tqdm(png_list, desc='images'):
        assert img.startswith("甲骨文字編-") and img.endswith(".png")
        img = img[6:-4]
        _, page, ch_idx, char, sub_idx, book_age = img.split('_')
        if sub_idx[-1] == '.':
            sub_idx = sub_idx[:-1]
        page, ch_idx, sub_idx = int(page), int(ch_idx), int(sub_idx)
        assert book_age[0] == '[' and book_age[-1] == ']'
        if book_age[:3] in ('[洹寶', '[續補', '[掇三', '[上博', '[山東'):
            err_log.write(f'error: irregular character, {img}\n')
            bone_to_shapes.setdefault((book_age[1:-1], char), [])
            bone_to_shapes[(book_age[1:-1], char)].append([page, ch_idx, sub_idx, -1, img, False])
            continue
        book_age = book_age.replace('（', '(')
        book_age = book_age.replace('）', ')')
        left_pos, right_pos = book_age.find('('), book_age.find(')')
        try:
            assert 1 < left_pos < right_pos - 1
        except AssertionError as err:
            print(img)
            raise err
        book_name, age = book_age[1:left_pos], book_age[left_pos+1:right_pos]
        book_name = book_name_convert(book_name, wzb_chant_map)
        if book_name.startswith('error'):
            err_log.write(f'{book_name}, {img}\n')
            bone_to_shapes.setdefault((book_age[1:-1], char), [])
            bone_to_shapes[(book_age[1:-1], char)].append([page, ch_idx, sub_idx, -1, img, False])
            continue
        if (book_name, char) not in bone_to_shapes:
            # 最后一个 bool 表示是否与汉达匹配成功
            bone_to_shapes[(book_name, char)] = []
        bone_to_shapes[(book_name, char)].append([page, ch_idx, sub_idx, age, img, False])
    save_json([key + tuple(val) for key, val in bone_to_shapes.items()], 'output/bone_to_shapes.json')
    err_log.close()

    handa_base = '/var/lib/shared_volume/data/private/songchenyang/hanzi_filter/handa'
    handa_parts = ['H', 'B', 'D', 'L', 'S', 'T', 'W', 'Y', 'HD']
    match_count = {'match_handa': 0, 'match_shape': 0, 'all_match_1': 0, 'all_match_more': 0}
    book_char_to_count = {(book_name, ch): count for
                          book_name, ch, count in load_json('output/tra_book_char_to_count.json')}
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
                base_img = os.path.basename(img)
                base_img_tokens = base_img[:-4].split('-')
                assert len(base_img_tokens) == 3 and base_img_tokens[2].isdigit() and base_img[-4:] == '.png'
                font = re.findall(r"<font face='([^']+)'>", ch)
                font = list(set(font))
                assert len(font) <= 1
                font = font[0] if len(font) > 0 else ''
                ch = re.sub(r'</?[^>]+>| ', '', ch).strip()
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
                                                  standard_liding_character=ch,
                                                  chant_font_label=font,
                                                  data_sources="chant.org")
                        char_font_index_to_id[(ch, font, -1)] = new_ch.id
                        # char_font_index_to_id[(ch, font, -1)] = cur_new_char_id
                        # cur_new_char_id += 1
                    # 考虑不匹配的情况，match_case == 0
                    match_count['match_handa'] += 1
                    new_shape = CharFace.create(
                        match_case=0,
                        standard_character_id=char_font_index_to_id[(ch, font, -1)],
                        first_order_standard_character=ch,
                        second_order_standard_character=-1,
                        chant_coordinates=coords,
                        chant_authentic_face=img,
                        chant_notation_number=row_order,
                        chant_face_index=int(base_img[:-4].split('-')[2]),
                        wzb_handcopy_face="",
                        published_collection_number=book_name,
                        wzb_calligraphy="",
                        wzb_page_number=-1,
                        wzb_row_number=-1,
                        wzb_col_number=-1,
                        data_sources="chant.org",
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
                    page, ch_idx, sub_idx, age, shape_img, _ = candidate
                    # 首先创建单字
                    if (ch, font, ch_idx) not in char_font_index_to_id:
                        new_ch = Character.create(wzb_character_number=ch_idx,
                                                  standard_liding_character=ch,
                                                  chant_font_label=font,
                                                  data_sources="chant.org\t《甲骨文字编》2012，正文")
                        char_font_index_to_id[(ch, font, ch_idx)] = new_ch.id
                        # char_font_index_to_id[(ch, font, ch_idx)] = cur_new_char_id
                        # cur_new_char_id += 1
                    new_shape = CharFace.create(
                        match_case=match_case,
                        standard_character_id=char_font_index_to_id[(ch, font, ch_idx)],
                        first_order_standard_character=ch,
                        second_order_standard_character=sub_idx,
                        chant_coordinates=coords,
                        chant_authentic_face=img,
                        chant_notation_number=row_order,
                        chant_face_index=int(base_img[:-4].split('-')[2]),
                        wzb_handcopy_face=shape_img,
                        published_collection_number=book_name,
                        wzb_calligraphy=age,
                        wzb_page_number=page,
                        wzb_row_number=-1,
                        wzb_col_number=-1,
                        data_sources="chant.org\t《甲骨文字编》2012，正文",
                    )
                cur_book_shape_ids.append(str(new_shape.id))
            # 创建新甲片
            ChantOracleBoneItems.create(
                chant_published_collection_number=book_name,
                chant_notation_number=row_order,
                chant_transcription_text=modern_text,
                chant_calligraphy=category,
                chant_url=url,
                characters='\t'.join(cur_book_shape_ids),
                chant_processed_rubbing=l_bone_img,
                chant_mapped_character_image=r_bone_img,
                data_sources="chant.org"
            )
    print(match_count)
    save_json(all_books, 'output/all_books.json')
    # {'match_handa': 428817, 'match_shape': 0, 'all_match_1': 11964, 'all_match_more': 146}
    # exit()

    # 处理剩下的字形, match_case == 1
    for book_name, char in tqdm(bone_to_shapes.keys(), desc='last'):
        check_sum = sum(int(candidate[-1]) for candidate in bone_to_shapes[(book_name, char)])
        assert check_sum == 0 or check_sum == len(bone_to_shapes[(book_name, char)])
        if check_sum == len(bone_to_shapes[(book_name, char)]):
            continue
        for page, ch_idx, sub_idx, age, shape_img, flag in bone_to_shapes[(book_name, char)]:
            assert not flag
            match_count['match_shape'] += 1
            # 创建新字
            if (char, '', ch_idx) not in char_font_index_to_id:
                new_ch = Character.create(wzb_character_number=ch_idx,
                                          standard_liding_character=char,
                                          chant_font_label='',
                                          data_sources="《甲骨文字编》2012，正文")
                char_font_index_to_id[(char, '', ch_idx)] = new_ch.id
            CharFace.create(
                match_case=1,
                standard_character_id=char_font_index_to_id[(char, '', ch_idx)],
                first_order_standard_character=char,
                second_order_standard_character=sub_idx,
                chant_coordinates="",
                chant_authentic_face="",
                chant_notation_number=-1,
                chant_face_index=-1,
                wzb_handcopy_face=shape_img,
                published_collection_number=book_name,
                wzb_calligraphy=age,
                wzb_page_number=page,
                wzb_row_number=-1,
                wzb_col_number=-1,
                data_sources="《甲骨文字编》2012，正文",
            )
    print(match_count)
    err_log_match.close()
    print(ChantOracleBoneItems.select().count())  # 125697
    print(Character.select().count())  # 8071
    print(CharFace.select().count())  # 468754
    print('count2:', CharFace.select().where(CharFace.match_case == 2).count())  # 11964
    print(CharFace.select().where(CharFace.match_case == 3).count())  # 146
    save_json([key + (val,) for key, val in char_font_index_to_id.items()], 'output/char_font_index_to_id.json')


if __name__ == '__main__':
    init_db()
    dump_data_new()
