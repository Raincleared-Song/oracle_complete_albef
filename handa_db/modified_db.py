from peewee import IntegerField, TextField, AutoField, Model, SqliteDatabase, CharField

db = SqliteDatabase('chant_oracle.db')


class BaseModel(Model):
    class Meta:
        database = db


class ChantOracleBoneItems(BaseModel):
    """
    甲片条目表，主要接驳汉达数据库原布局，每条数据代表甲片中的一个条目，(著录号-条号) 唯一对应一条数据。约125697行数据。
    """
    # 自增 id 域，主键
    id = AutoField()
    # 著录号（甲片的唯一标识），最大长度 511 字符, 原 book_name
    chant_published_collection_number = CharField(null=False, max_length=511,
                                                  column_name='chant_published_collection_number')
    # 汉达条号（正整数，甲片下某个甲骨文句子的序号），原 row_order
    chant_notation_number = IntegerField(null=False, column_name='chant_notation_number')
    # 汉达释文，甲骨文句子（繁体汉字），带 font 标签，原 modern_text
    chant_transcription_text = TextField(null=False, column_name='chant_transcription_text')
    # 汉达字体分类，最大长度 7，原 category
    chant_calligraphy = CharField(null=False, max_length=7, column_name='chant_calligraphy')
    # 字形列表，单条释文中包含单字或合文的列表，以 '\t' 分隔的字符串，每个元素都是汉达字表 ChantAuthenticFace.id
    characters = TextField(null=False, column_name='characters')
    # 汉达甲骨图片的路径，最大长度 511 字符，原 l_bone_img
    chant_processed_rubbing = CharField(null=False, max_length=511, column_name='chant_processed_rubbing')
    # 汉字排布图的路径，最大长度 511 字符，原 r_bone_img
    chant_mapped_character_image = CharField(null=False, max_length=511, column_name='chant_mapped_character_image')
    # 源数据在汉达文库中的 url 后缀，最大长度 511 字符，原 url
    chant_url = CharField(null=False, max_length=511, column_name='chant_url')
    # 数据来源，新增列，如有多个来源，请用 \t 分割，默认值为 chant.org
    data_sources = TextField(null=False, default='chant.org', column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 著录号-条号 联合唯一索引
        indexes = (
            (('book_name', 'row_order'), True),
        )


class ChantAuthenticFace(BaseModel):
    """
    字形表（原形），主要接驳汉达数据库原布局，每条数据对应汉达文库的一个字形，包含单字+合文。约440927行数据。
    """
    # 自增 id 域，主键
    id = AutoField()
    # 所属的著录号，最大长度 511 字符，原 book_name
    chant_published_collection_number = CharField(null=False, max_length=511,
                                                  column_name='chant_published_collection_number')
    # 所属汉达条号（正整数，甲片下某个甲骨文句子的序号），原 row_order
    chant_notation_number = IntegerField(null=False, column_name='chant_notation_number')
    # 所属字形列表索引，该字形在 OracleBoneItems 表中 characters 列的索引位置，新增列，-1 表示未知
    characters_index = IntegerField(null=False, default=-1, column_name='characters_index')
    # todo 属于哪一个标准字 ChantCharacter.id，索引字段，原 char_belong
    standard_character = IntegerField(null=False, index=True, column_name='standard_character')
    # 在汉达文库甲片图中的坐标信息，可能为空（match_case == 1），原 coords
    chant_coordinates = CharField(null=False, max_length=63, column_name='chant_coordinates')
    # 原形，即汉达文库中带背景的噪声图片路径，最大长度 511 字符，可能为空（match_case == 1），原 noise_image
    chant_authentic_face = CharField(null=False, max_length=511, column_name='chant_authentic_face')
    # 甲骨分类标识，原 category
    chant_calligraphy = CharField(null=False, max_length=7, column_name='chant_calligraphy')
    # 数据来源，新增列，如有多个来源，请用 \t 分割，默认值为 chant.org
    data_sources = TextField(null=False, default='chant.org', column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # todo 新增这一部分。著录号-条号-字形列表索引号，联合唯一索引
        indexes = (
            (('chant_published_collection_number', 'chant_notation_number', 'characters_index'), True),
        )


class ChantCharacter(BaseModel):
    """
    汉达字表，包含单字+合文。约4-5千行数据。（由于汉达与《文字编》释字法不完全一致，分为2张表更利于数据维护）
    """
    # 自增 id 域，主键
    id = AutoField()
    # 对应的标准字，utf-8 编码，原 char_byte
    standard_character = CharField(null=False, max_length=7, column_name='standard_character')
    # 汉达字体标签 name，原 font
    chant_font_label = CharField(null=False, max_length=7, column_name='chant_font_label')
    # 数据来源，新增列，如有多个来源，请用 \t 分割，默认值为 chant.org
    data_sources = TextField(null=False, default='chant.org', column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 编码-字体 联合唯一索引 #todo index中名称是否要更新？
        indexes = (
            (('char_byte', 'font'), False),
        )


class WzbCharacter(BaseModel):
    """
    《文字编》字表，包含单字+合文。单字部分，对应于《文字编·字表目录》的每一行数据；合文部分，需另外提取，暂留空。约5000行数据。
    """
    # 自增 id 域，主键
    id = AutoField()
    # 《字表目录》第3列“字号”，索引字段，原 char_index
    wzb_character_number = IntegerField(null=False, index=True, column_name='wzb_character_number')
    # todo 《字表目录》第1列“字形”，新增列
    standard_form_of_inscription = CharField(null=False, max_length=511, column_name='standard_form_of_inscription')
    # todo 《字表目录》第2列“隶定”，utf-8 编码，原 char_byte
    standard_form_of_liding_character = CharField(null=False, max_length=7,
                                                  column_name='standard_form_of_liding_character')
    # todo 一级字头，新增列
    first_order_standard_character = CharField(null=False, max_length=511, column_name='first_order_standard_character')
    # todo 二级字头，可能为空，新增列
    second_order_standard_character = CharField(null=False, max_length=511,
                                                column_name='second_order_standard_character')
    # todo 《字表目录》第4列“页码”，可能为空，原 page_number
    wzb_page_number = IntegerField(null=False, column_name='wzb_page_number')
    # 部首编号，未指定时为 -1
    wzb_radical_number = IntegerField(null=False, default=-1, column_name='wzb_radical_number')
    # todo 部首，新增列
    wzb_radical = CharField(null=False, max_length=7, column_name='wzb_radical')
    # todo 拼音，新增列
    wzb_spelling = CharField(null=False, max_length=511, column_name='wzb_spelling')
    # todo 笔画数，新增列
    wzb_stroke_count = IntegerField(null=False, default=-1, column_name='wzb_stroke_count')
    # 数据来源，新增列，如有多个来源，请用 \t 分割，默认值为 《甲骨文字编》2012，正文
    data_sources = TextField(null=False, default='《甲骨文字编》2012，正文', column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:  # todo 可能需要重新规划一下此处的meta，似乎要以文字编字号为索引
        # 编码-字体 联合唯一索引
        indexes = (
            (('char_byte', 'font'), False),
        )


class WzbHandcopyFace(BaseModel):
    """
    《文字编》字形表（摹形），对应李老师摹本中的每一个字形，包含单字+合文。约4万行数据。
    """
    # 自增 id 域，主键
    id = AutoField()
    # todo 属于哪一个字号 WzbCharacter.id，原 char_belong 这里需要跟踪的是“WzbCharacter.id"，下面一行不知道写得对不对，请查验
    wzb_character_id = IntegerField(null=False, index=True, column_name='wzb_character_id')
    # todo 所属一级字头，新增列 如果能跟踪到WzbCharacter.id，也就知道一级字头与二级字头了，那么此表中”first_order_standard_character”,“second_order_standard_character”也可删除，你来决定吧
    first_order_standard_character = CharField(null=False, max_length=511,
                                               column_name='first_order_standard_character')  # todo: 是否删除？
    # todo 二级字头，可能为空，新增列
    second_order_standard_character = CharField(null=False, max_length=511,
                                                column_name='second_order_standard_character')  # todo: 是否删除？
    # 摹写字形图片路径，最大长度 511 字符，可能为空（match_case == 0），原 shape_image
    handcopy_face = CharField(null=False, max_length=511, column_name='wzb_handcopy_face')
    # 所属著录号，原 book_name
    wzb_published_collection_number = CharField(null=False, max_length=511,
                                                column_name='wzb_published_collection_number')
    # 李老师摹本类别码，最大长度 7，可能为空（match_case == 0），missing 表示找不到有效 ocr 编码，原 category
    wzb_calligraphy = CharField(null=False, max_length=7, column_name='wzb_calligraphy')
    # todo:【这条请重写】结构类似汉达的字形列表，是对上的汉达原形图片的“著录号-条号-列表索引号”所组成的list。可能为空，可以唯一（是我们需要的），可以有多个（有歧义），中间用"\t"分开？
    # todo 李霜洁to宋这条写不写。对上的数据一直在更新，是通过数据库的形式保留（放在WzbHandcopyFace表，还是ChantAuthenticFace表），还是查询调用，怎么合适
    chant_authentic_face_id = None  # todo
    # 页码号，可能为 -1（match_case == 0），原 page_number
    wzb_page_number = IntegerField(null=False, column_name='wzb_page_number')
    # 第几行，可能为 -1（match_case == 0）
    wzb_row_number = IntegerField(null=False, column_name='wzb_row_number')
    # 第几列，可能为 -1（match_case == 0）
    wzb_col_number = IntegerField(null=False, column_name='wzb_col_number')
    # 数据来源，新增列，如有多个来源，请用 \t 分割，默认值为 《甲骨文字编》2012，正文
    data_sources = TextField(null=False, default='《甲骨文字编》2012，正文', column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')
