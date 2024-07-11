# -*- coding: utf-8 -*-

import random
import time
import traceback
import requests
import io
import json
import base64
import urllib

import re
from qanything_kernel.configs.model_config import MAX_TOKENS_FOR_CHUNK_SUMMARY_GEN, MAX_CHARS_FOR_CHUNK_TRANSLATION
from qanything_kernel.utils.general_utils import cur_func_name, num_tokens_from_messages, sent_tokenize
from qanything_kernel.utils.custom_log import debug_logger, qa_logger

# block_type = block["blockType"] # TEXT, TITLE, IMAGE_CAPTION, TABLE_CAPTION, CAPTION, REFERENCE, BIB_INFO, HEADER, FOOTER, CONTENT, CODE, OTHER, FORMULA, IMAGE, TABLE, SEPARATELINE

#  WATCH_PAGE_IDS = [1]
WATCH_PAGE_IDS = []


def delete_space_in_Chinese(text):
    return re.sub(r'\s?([\u4e00-\u9fa5])\s?', r'\1', text)

def call_pdf_parse_service(b64):
    data = {
        'img': b64,
        'uid': 'youdaoocr',
        'type': '10012',
        #  'options': 'input_is_pdf,check_scanned_pdf,only_parse'
        'options': 'input_is_pdf,only_parse'
    }
    d = urllib.parse.urlencode(data).encode(encoding='UTF8')  # 字典 data 中保存请求数据，包括输入图片和请求参数等

    f = urllib.request.urlopen(
        # url = 'http://gpu98:8088/ocr',
        # url = 'https://vertifytest.youdao.com/pdf2pdf',
        #  url = 'http://gpu98:8401/ocr',
        url = 'https://pdf-split-parse.corp.youdao.com/split_parse',
        # url = 'http://103.74.51.95:49382/split_parse',
        # url = 'http://10.55.162.112:49382/split_parse', # gpu23
        data = d
    )
    s = f.read()
    js = json.loads(str(s, 'utf-8'))
    try:
        if 'code' in js:
            debug_logger.info(f'[call_pdf_parse_service] "code" in js, js = {js}')

        elif js['Status'] == 'success':
            js['Result'] = json.loads(js['Result'])
        else:
            debug_logger.info(f'[call_pdf_parse_service] status = {js["Status"]}, js = {js}')
    except:
        debug_logger.error(f'[call_pdf_parse_service] failed with no Status, detail: {traceback.format_exc()}, js = {js}')

    # if DEBUG_PDF_INCOMPLETE:
    #     now = datetime.now()
    #     timestamp = datetime.timestamp(now)
    #     with open(f'pdf_{int(timestamp)}.json', 'w') as f:
    #         json.dump(js['Result'], f, indent=4)
    return js

def is_cn_ja(char):
    res = False
    c = ord(char)
    if ((c >= 0x3000 and c <= 0x303f) or
            (c >= 0x3040 and c <= 0x30ff) or (c >= 0x3400 and c <= 0x4dbf) or
            (c >= 0x4e00 and c <= 0x9fff) or (c >= 0xe000 and c <= 0xf8ff) or
            (c >= 0xff01 and c <= 0xffef) or (c >= 0x20000 and c <= 0x2ebef) or
            (c >= 0x2f800 and c <= 0x2fa1f) or (c >= 0x30000 and c<= 0x3134f)):
        res = True
    return res


def do_formula_request(imb64):
    FORMULA_URL="http://api.ocr.youdao.com/ocr_formula"
    #  FORMULA_URL="https://api2.ocr.youdao.com/accurate_ocr"
    data = {
            "img": imb64,
            "uid": "youdaoocr",
            "type": "10012",
            "options": "formula"
            }
    d = urllib.parse.urlencode(data).encode(encoding='UTF8')
    response = None
    try:
        response = urllib.request.urlopen(
                url=FORMULA_URL,
                data=d
            )
    except:
        debug_logger.error("formula ocr failed!!!")
        pass
    out = None
    if response:
        out = json.loads(response.read().decode())
        out = json.loads(out["Result"])
        text = ""
        for region in out["regions"]:
            for line in region["lines"]:
                pre_w = text[-1] if len(text) > 0 else "\u3000"
                if isinstance(line, dict):
                    is_pre_cn_ja = is_cn_ja(pre_w)
                    is_post_cn_ja = is_cn_ja(line["text"][0])
                    if is_pre_cn_ja or is_post_cn_ja:
                        text += line["text"]
                    else:
                        text += " "
                        text += line["text"]
                else:
                    for subline in line:
                        is_pre_cn_ja = is_cn_ja(pre_w)
                        is_post_cn_ja = is_cn_ja(subline["text"][0])
                        if is_pre_cn_ja or is_post_cn_ja:
                            text += subline["text"]
                        else:
                            text += " "
                            text += subline["text"]
            # text += "\n"
        out = text
    return out


def block_types_2_chunk_types(jr, block_type_str):
    if block_type_str == "TITLE":
        return "title"
    return "normal"

# [20231102][deprecated] 新规则：对于一个 block 内的文本，就合成为一个大块，给到前后端的就是一整块文本。请求摘要也是用着一整块文本来请求的。所以就消灭以前结果里的 paragraph 概念即可。
# [20231103] TODO 新规则：对于一个 block 内的多个 para，如果有过短的 para，就尝试往前后合并得到新的para，bbox以合并后的总bbox为准。
def extract_paras_text_from_1_block(jr, block, page_id, block_type):
    paras = block["paragraphs"]
    # if "splicing_info" in block:
        # splicing_info = block["splicing_info"]
        # if splicing_info: # TODO 如果这个字段非空，则说明当前段落后面需要拼接一个段落
            # _a, _b, _c, _d = [int(x) for x in splicing_info.split(",")]
            # next_block = jr["pages"][_a]["sections"][_b]["columns"][_c]["blocks"][_d]
    local_para_text_strs = []
    need_merge_last_para_and_cur_para = False # 标志，指出当前已处理过的最后一段（即，已添加进local_para_text_strs的最后一段，是否过短、需要与后续段落合并）
    block_bbox = block["boundingBox"]
    # line_bboxes = []
    #  lines_to_output = []
    page = jr["pages"][page_id]
    page_w = page["pageWidth"]
    page_h = page["pageHeight"]

    for para in paras:
        para_text = ""
        para_bbox = para['boundingBox']
        lines_to_output = []
        lines = para["lines"]
        for line in lines:
            words = line["words"]
            line_bbox = line["boundingBox"]
            # line_bboxes.append(line_bbox)
            line_text = ""
            line_fontsizes = []
            for w_i, word in enumerate(words):
                word_text = word["text"]
                if len(word_text) > 1 and word_text[-1] == '-' and (w_i == len(words) - 1):
                    word_text = word_text[:-1]
                elif "hasSpaceAfter" in word and word["hasSpaceAfter"]:
                    # TODO [20231012] 需要特殊规则：当全都是大写字母，但首字母是大号字体时，不能加空格。判定规则为：当当前单词是单字母且大写字母，判断其下一个word的首字母，如果也是大写字母，且fontSize与当前字母的 fontSize 相差5%以上，说明这俩word应该拼起来，则这里不要添加额外的空格。
                    if  len(words)> w_i + 1 : # 首先得有下个单词啊
                        next_word = words[w_i+1]
                        next_word_chars = next_word["chars"]
                        if len(next_word_chars) > 0:
                            #  # TODO [20231025] 如果下个单词有多个字母，就可以根据字母之间 x 轴的空隙大小，判断后继单词与当前单词是不是需要合并。
                            #  next_word__1st_char_bbox = next_word_chars[0]['boundingBox']

                            next_word_first_char_font_size = float(next_word_chars[0]['fontSize'])
                            if len(word_text) == 1 and word_text.isupper() and next_word["text"].isupper():
                                cur_char_font_size = float(word['chars'][0]['fontSize'])
                                #  print(f'VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV')
                                #  print(f'word_text = {word_text}, next_word_text = {next_word["text"]}, cur_char_font_size = {cur_char_font_size}, next_word_first_char_font_size = {next_word_first_char_font_size}')
                                #  print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
                                if cur_char_font_size and next_word_first_char_font_size / cur_char_font_size < 0.95: # 如果相邻俩大写单词的字号一致，甚至后继单词字号还大于前驱字号，说明这俩不是一个词，需要插入空格。而如果后继比前驱的字号小，则说明是同一个词，不要插入空格！
                                    word_text += ""
                                else:
                                    word_text += " "
                            else:
                                word_text += " "
                        else:
                            word_text += " "
                    else:
                        word_text += " "
                para_text += word_text
                line_text += word_text

                # 记录该行的行高即字体大小
                chars = word["chars"]
                for char in chars:
                    # print(f'fontSize = {type(char["fontSize"])}')
                    line_fontsizes.append(float(char["fontSize"]))
            line_fontsize = round(sum(line_fontsizes) / len(line_fontsizes)) if line_fontsizes else 12
            #  print(f'[{cur_func_name()}] before delete space in Chinese, line_text = {line_text}')
            line_text = delete_space_in_Chinese(line_text)
            #  print(f'[{cur_func_name()}] after delete space in Chinese, line_text = {line_text}')
            line_to_output = {
                "line_bbox": line_bbox,
                "line_text": line_text,
                "line_fontsize": line_fontsize
            }
            lines_to_output.append(line_to_output)

        locations = [
            {
                "page_id": page_id,
                "page_w": page_w,
                "page_h": page_h,
                #  "bbox": block_bbox,
                "bbox": para_bbox,
                # "line_bboxes": line_bboxes
                "lines": lines_to_output
            }
        ]
        para_text = delete_space_in_Chinese(para_text)
        para_dict = {
            "text": para_text,
            "page_ids": [int(page_id)],
            "chunk_type": block_types_2_chunk_types(jr, block_type),
            "locations": locations
        }

        # TODO DEBUG
        # if page_id in [2]:
        if page_id in WATCH_PAGE_IDS:
            debug_logger.info(f'[{cur_func_name()}] = = = = = = = = = = = = =VVVVVVVVVVV')
            debug_logger.info(f'para_dict = {json.dumps(para_dict, ensure_ascii=False, indent=4)}')
            debug_logger.info(f'[{cur_func_name()}] = = = = = = = = = = = = =AAAAAAAAAAA')

        if (need_merge_last_para_and_cur_para or para_too_short_and_need_merge(para_dict)) and len(local_para_text_strs) > 0 and chunk_types_compatiable_to_merge(local_para_text_strs[-1], para_dict): # TODO 如果需要把当前段落拼到已处理过的最后一段上，就拼。否则把新段落作为单独一段，加入 local_para_text_strs。
            last_para = local_para_text_strs[-1]
            last_para['text'] += "\n\n"
            last_para['text'] += para_dict['text']
            # last_para['page_ids'].extend(para_dict['page_ids'])
            # last_para['locations'].extend(para_dict['locations'])
            merged_locations = merge_para_locations(last_para, para_dict)
            last_para['locations'] = merged_locations
            # TODO 合并后，看看新的段落是否够长了。如果还不够长，need_merge_last_para_and_cur_para 赋值为 True
            local_para_text_strs[-1] = last_para
        else:
            local_para_text_strs.append(para_dict)
        if para_too_short_and_need_merge(local_para_text_strs[-1]):
            need_merge_last_para_and_cur_para = True
        else:
            need_merge_last_para_and_cur_para = False
    return local_para_text_strs

def merge_para_locations(last_para, para_dict):
    last_locations = last_para['locations']
    cur_locations = para_dict['locations']
    last_para_bbox_xywh_str = last_locations[0]['bbox']
    cur_para_bbox_xywh_str = cur_locations[0]['bbox']
    last_para_bbox = list(map(int, last_para_bbox_xywh_str.split(',')))
    cur_para_bbox = list(map(int, cur_para_bbox_xywh_str.split(',')))
    last_x, last_y, last_w, last_h = last_para_bbox
    cur_x, cur_y, cur_w, cur_h = cur_para_bbox
    merged_x = min(last_x, cur_x)
    merged_y = min(last_y, cur_y)
    merged_x2 = max((last_x + last_w), (cur_x + cur_w))
    merged_y2 = max((last_y + last_h), (cur_y + cur_h))
    merged_w = merged_x2 - merged_x
    merged_h = merged_y2 - merged_y
    merged_para_bbox = [merged_x,
                        merged_y,
                        merged_w,
                        merged_h
                        ]
    merged_locations = last_locations
    merged_locations[0]['bbox'] = ','.join(list(map(str, merged_para_bbox)))
    merged_locations[0]['lines'].extend(cur_locations[0]['lines'])
    return merged_locations


def para_too_short_and_need_merge(para_dict):
    # if len(para_dict['text']) < 500:
    if num_tokens_from_messages([para_dict['text']]) < 200:
        return True
    return False

def chunk_types_compatiable_to_merge(last_para_dict, cur_para_dict):
    if last_para_dict['chunk_type'] == cur_para_dict['chunk_type']:
        return True
    return False

def extract_paras_text_from_1_block__old(jr, block, page_id, block_type):
    paras = block["paragraphs"]
    # if "splicing_info" in block:
        # splicing_info = block["splicing_info"]
        # if splicing_info: # TODO 如果这个字段非空，则说明当前段落后面需要拼接一个段落
            # _a, _b, _c, _d = [int(x) for x in splicing_info.split(",")]
            # next_block = jr["pages"][_a]["sections"][_b]["columns"][_c]["blocks"][_d]
    local_para_text_strs = []
    block_bbox = block["boundingBox"]
    # line_bboxes = []
    #  lines_to_output = []
    page = jr["pages"][page_id]
    page_w = page["pageWidth"]
    page_h = page["pageHeight"]

    for para in paras:
        para_text = ""
        para_bbox = para['boundingBox']
        lines_to_output = []
        lines = para["lines"]
        for line in lines:
            words = line["words"]
            line_bbox = line["boundingBox"]
            # line_bboxes.append(line_bbox)
            line_text = ""
            line_fontsizes = []
            for w_i, word in enumerate(words):
                word_text = word["text"]
                if len(word_text) > 1 and word_text[-1] == '-' and (w_i == len(words) - 1):
                    word_text = word_text[:-1]
                elif "hasSpaceAfter" in word and word["hasSpaceAfter"]:
                    # TODO [20231012] 需要特殊规则：当全都是大写字母，但首字母是大号字体时，不能加空格。判定规则为：当当前单词是单字母且大写字母，判断其下一个word的首字母，如果也是大写字母，且fontSize与当前字母的 fontSize 相差5%以上，说明这俩word应该拼起来，则这里不要添加额外的空格。
                    if  len(words)> w_i + 1 : # 首先得有下个单词啊
                        next_word = words[w_i+1]
                        next_word_chars = next_word["chars"]
                        if len(next_word_chars) > 0:
                            #  # TODO [20231025] 如果下个单词有多个字母，就可以根据字母之间 x 轴的空隙大小，判断后继单词与当前单词是不是需要合并。
                            #  next_word__1st_char_bbox = next_word_chars[0]['boundingBox']

                            next_word_first_char_font_size = float(next_word_chars[0]['fontSize'])
                            if len(word_text) == 1 and word_text.isupper() and next_word["text"].isupper():
                                cur_char_font_size = float(word['chars'][0]['fontSize'])
                                #  print(f'VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV')
                                #  print(f'word_text = {word_text}, next_word_text = {next_word["text"]}, cur_char_font_size = {cur_char_font_size}, next_word_first_char_font_size = {next_word_first_char_font_size}')
                                #  print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
                                if cur_char_font_size and next_word_first_char_font_size / cur_char_font_size < 0.95: # 如果相邻俩大写单词的字号一致，甚至后继单词字号还大于前驱字号，说明这俩不是一个词，需要插入空格。而如果后继比前驱的字号小，则说明是同一个词，不要插入空格！
                                    word_text += ""
                                else:
                                    word_text += " "
                            else:
                                word_text += " "
                        else:
                            word_text += " "
                    else:
                        word_text += " "
                para_text += word_text
                line_text += word_text

                # 记录该行的行高即字体大小
                chars = word["chars"]
                for char in chars:
                    # print(f'fontSize = {type(char["fontSize"])}')
                    line_fontsizes.append(float(char["fontSize"]))
            line_fontsize = round(sum(line_fontsizes) / len(line_fontsizes)) if line_fontsizes else 12
            #  print(f'[{cur_func_name()}] before delete space in Chinese, line_text = {line_text}')
            line_text = delete_space_in_Chinese(line_text)
            #  print(f'[{cur_func_name()}] after delete space in Chinese, line_text = {line_text}')
            line_to_output = {
                "line_bbox": line_bbox,
                "line_text": line_text,
                "line_fontsize": line_fontsize
            }
            lines_to_output.append(line_to_output)

        locations = [
            {
                "page_id": page_id,
                "page_w": page_w,
                "page_h": page_h,
                #  "bbox": block_bbox,
                "bbox": para_bbox,
                # "line_bboxes": line_bboxes
                "lines": lines_to_output
            }
        ]
        para_text = delete_space_in_Chinese(para_text)
        para_dict = {
            "text": para_text,
            "page_ids": [int(page_id)],
            "chunk_type": block_types_2_chunk_types(jr, block_type),
            "locations": locations
        }

        # TODO DEBUG
        # if page_id in [2]:
        if page_id in WATCH_PAGE_IDS:
            debug_logger.info(f'[{cur_func_name()}] = = = = = = = = = = = = =')
            debug_logger.info(f'para_dict = {json.dumps(para_dict, ensure_ascii=False, indent=4)}')
            debug_logger.info(f'[{cur_func_name()}] = = = = = = = = = = = = =')

        local_para_text_strs.append(para_dict)
    return local_para_text_strs

def extend_paras_of_cur_and_next_block(cur_paras, next_paras):
    # logger = Logger.get_logger()
    if next_paras:
        next_first_para = next_paras[0]
        if cur_paras:
            # print(f'[extend_paras_of_cur_and_next_block] ------------VVVVVV-------------')
            # print(f'cur_paras[-1] = {cur_paras[-1]}')
            # print(f'next_first_para = {next_first_para}')
            # print(f'[extend_paras_of_cur_and_next_block] ------------AAAAAA-------------')
            #  logger.info(f'[extend_paras_of_cur_and_next_block] ------------VVVVVV-------------')
            #  logger.info(f'cur_paras[-1] = {cur_paras[-1]}')
            #  logger.info(f'next_first_para = {next_first_para}')
            #  logger.info(f'[extend_paras_of_cur_and_next_block] ------------AAAAAA-------------')
            cur_paras[-1]["text"] += next_first_para["text"]
            cur_paras[-1]["locations"].extend(next_first_para["locations"])
            cur_paras[-1]["page_ids"].extend(next_first_para["page_ids"])
        next_paras = next_paras[1:]
    cur_paras.extend(next_paras)
    return cur_paras

def para_long_enough_to_be_valid_chunk(paras_of_cur_block):
    return True
    # chunk_type = paras_of_cur_block["chunk_type"]
    # if chunk_type == "normal":
        # all_text = paras_of_cur_block["text"]
        # if len(all_text) < 160:
            # return False
    # return True

def extract_paras_text_from_formula_block(jr, block, page_id, block_type):
    # 进来的一定是公式block，则直接把整个公式解析结果塞进 text 里吧
    formula_text = do_formula_request(block['imageMat'])

    local_para_text_strs = []
    block_bbox = block["boundingBox"]
    # lines_to_output = []
    page = jr["pages"][page_id]
    page_w = page["pageWidth"]
    page_h = page["pageHeight"]

    locations = [
        {
            "page_id": page_id,
            "page_w": page_w,
            "page_h": page_h,
            "bbox": block_bbox,
            "lines": [
                {
                    "line_bbox": block_bbox,
                    "line_text": formula_text if formula_text else "",
                    "line_fontsize": 12
                }
            ]
        }
    ]
    para_dict = {
        "text": formula_text if formula_text else "",
        "page_ids": [int(page_id)],
        "chunk_type": block_types_2_chunk_types(jr, block_type),
        "is_formula": True,
        "locations": locations
    }
    # print(f'para_dict = {para_dict}')
    local_para_text_strs.append(para_dict)
    return local_para_text_strs

def process_1_block(jr, block, para_text_strs, page_id, cur_block_abcd=None):
    # block["processed"] = True # 因为有可能跳着访问后续的block，所以如果有被处理过的，就要标记一下，避免重复处理
    if "processed" in block and block["processed"]:
        return

    block_type = block["blockType"]
    if block_type not in ["TEXT", "TITLE", "FORMULA"]:
        # print(f'[{cur_func_name()}] block_type = {block_type} , page_id = {page_id}')
        return
    if block["isVertical"] != "0":
        # print(f'[{cur_func_name()}] isVertical = {block["isVertical"]} , page_id = {page_id}')
        return
    if "rotation" in block and block["rotation"] != "0":
        # print(f'[{cur_func_name()}] rotation = {block["rotation"]} , page_id = {page_id}')
        return
    # paras = block["paragraphs"]
    paras_of_next_block = []
    if "splicing_info" in block:
        splicing_info = block["splicing_info"]
        if splicing_info: # TODO 如果这个字段非空，则说明当前段落后面需要拼接一个段落
            _a, _b, _c, _d = [int(x) for x in splicing_info.split(",")]
            # debug_logger.info(f"[{cur_func_name()}] _a = {_a}, len(pages) = {len(jr['pages'])}, _b = {_b}, _c = {_c}, _d = {_d}")
            next_block = jr["pages"][_a]["sections"][_b]["columns"][_c]["blocks"][_d]
            # [20230818] 发现奇怪的现象：论文的Abstract往往会被拼接到前面的作者信息后面。这里加个特判吧，检查 next_block 是否以 "Abstract" 或 "ABSTRACT" 开头，是的话，就取消拼接。
            if check_next_block_valid_for_block_concat_by_splicing_info(next_block):
                paras_of_next_block  = extract_paras_text_from_1_block(jr, next_block, _a, block_type)
                if page_id in WATCH_PAGE_IDS:
                    debug_logger.info(f"paras_of_next_block = {json.dumps(paras_of_next_block, indent=4, ensure_ascii=False)}")
                    debug_logger.info('--------------------------')
                next_block["processed"] = True

    if block_type == "FORMULA":
        paras_of_cur_block = extract_paras_text_from_formula_block(jr, block, page_id, block_type)
        # print(f'[{cur_func_name()}] got formula: {json.dumps(paras_of_cur_block, indent=4, ensure_ascii=False)}')
    else:
        paras_of_cur_block = extract_paras_text_from_1_block(jr, block, page_id, block_type)
    if page_id in WATCH_PAGE_IDS:
        debug_logger.info(f"paras_of_cur_block = {json.dumps(paras_of_cur_block, indent=4, ensure_ascii=False)}")
        debug_logger.info('--------------------------')

    # # TODO [20231106][deprecated][改为全部block处理完后，再从头遍历一遍chunks，合并跨chunk的短段落]
    # 新增跨 block 的短段落拼接
    # if cur_block_abcd:
        # cur_a, cur_b, cur_c, cur_d = cur_block_abcd
        # # TODO 如果当前block的最后一个 para 过短，尝试把下一个 block 合并过来。
        # if block_too_short_and_need_merge(paras_of_cur_block):
            # blocks_of_cur_column = jr["pages"][cur_a]["sections"][cur_b]["columns"][cur_c]["blocks"]
            # if cur_d + 1 < len(blocks_of_cur_column):
                # next_block = blocks_of_cur_column[cur_d+1]
                # paras_of_next_block  = extract_paras_text_from_1_block(jr, next_block, _a, block_type)
                # if page_id in WATCH_PAGE_IDS:
                    # print(f"paras_of_next_block = {json.dumps(paras_of_next_block, indent=4, ensure_ascii=False)}")
                    # print('--------------------------')
                # next_block["processed"] = True

    paras_of_cur_block = extend_paras_of_cur_and_next_block(paras_of_cur_block, paras_of_next_block)
    block["processed"] = True
    if para_long_enough_to_be_valid_chunk(paras_of_cur_block[0]):
        para_text_strs.extend(paras_of_cur_block)
    return

def block_too_short_and_need_merge(block):
    para = block[-1]
    if len(para['text']) < 500:
        return True
    return False
    # chunk_type = paras_of_cur_block["chunk_type"]
    # if chunk_type == "normal":
        # all_text = paras_of_cur_block["text"]
        # if len(all_text) < 160:
            # return False
    # return True

def check_next_block_valid_for_block_concat_by_splicing_info(block):
    if "paragraphs" not in block:
        return False

    # found_abstract_as_begining = False
    paras = block["paragraphs"]
    if len(paras) > 0:
        lines = paras[0]["lines"]
        if len(lines) > 0:
            words = lines[0]["words"]
            line_text = ""
            for w_i, word in enumerate(words):
                word_text = word["text"]
                line_text += word_text
            if line_text.strip().lower().startswith("abstract"):
                return False
    return True

def extract_text_blocks_from_groupblocks(parsed_pdf_json):
    status = parsed_pdf_json["Status"]

    if status == "success":
        jr = parsed_pdf_json["Result"]
        pages = jr["pages"]
        # print(f'[pdf2paras] len(pages) = {len(pages)}')
        for p_i, p in enumerate(pages):
            # debug_logger.info(f'[{cur_func_name()}] Processing page: {p_i}')
            sections = p["sections"]
            for sec_i, sec in enumerate(sections):
                section_type = sec["sectionType"]
                if section_type not in ["TEXTBOX", "NORMAL"]:
                    continue
                cols = sec["columns"]
                for col_i, column in enumerate(cols):
                    blocks = column["blocks"]
                    new_blocks_without_groupblocks = []
                    for block_i, block in enumerate(blocks):
                        new_blocks_without_groupblocks.append(block)
                        # TODO 看看有没有 groupBlocks，有的话，把其中内容取出来，放到与 blocks 平级。
                        if 'groupBlocks' in block:
                            # print(f'[{cur_func_name()}] yes yes YES! groupBlocks found!')
                            new_blocks_without_groupblocks.extend(block['groupBlocks'])
                    jr['pages'][p_i]['sections'][sec_i]['columns'][col_i]['blocks'] = new_blocks_without_groupblocks

    return parsed_pdf_json


"""
把pdf解析服务返回的东西，转换成文档总结需要的 chunks。

https://note.youdao.com/ynoteshare/index.html?id=ba84eec349087b624510d6b84b4c4eb0&type=note&_time=1678700026260

pages
sections
columns
blocks
paragraphs
lines
words
text, hasSpaceAfter
"""
def pdf2paras(parsed_pdf_json):
    status = parsed_pdf_json["Status"]
    para_text_strs = []
    # TODO [20231109] 先把 groupblocks 处理成普通 blocks
    parsed_pdf_json = extract_text_blocks_from_groupblocks(parsed_pdf_json)

    if status == "success":
        jr = parsed_pdf_json["Result"]
        pages = jr["pages"]
        # print(f'[pdf2paras] len(pages) = {len(pages)}')
        for p_i, p in enumerate(pages):
            # debug_logger.info(f'Processing page: {p_i}')
            sections = p["sections"]
            for sec_i, sec in enumerate(sections):
                section_type = sec["sectionType"]
                if section_type not in ["TEXTBOX", "NORMAL"]:
                    continue
                cols = sec["columns"]
                for col_i, column in enumerate(cols):
                    blocks = column["blocks"]
                    for block_i, block in enumerate(blocks):
                        # 一旦遇到了参考文献，就直接中止后续的解析。避免大量参考文献被解析为正文，搞一大堆chunk出来干扰摘要。
                        if block['blockType'] == "REFERENCE":
                            #  print(f'Met REFERENCE, stop parsing remained pages of cur pdf, block = {block}')
                            #  return para_text_strs
                            continue

                        # # TODO DEBUG [20230818]
                        # print(f'[pdf2paras][block text]')
                        # print_block(block)
                        # print(f'[pdf2paras]============== 1block finished =====================')

                        cur_block_abcd = (p_i, sec_i, col_i, block_i)
                        process_1_block(jr, block, para_text_strs, p_i, cur_block_abcd)

    # TODO [20231106] 再从头遍历 para_text_strs，把短段落合并起来
    para_text_strs = merge_conj_short_paras(para_text_strs)


    # print(f'len(para_text_strs) = {len(para_text_strs)}')
    #  print(f'[pdf2paras] para_text_strs = {para_text_strs}')
    return para_text_strs

def if_merged_paras_len_will_not_exceed_thresh(last_para_dict, cur_para_dict):
    if num_tokens_from_messages([last_para_dict['text'], cur_para_dict['text']]) < MAX_TOKENS_FOR_CHUNK_SUMMARY_GEN and (len(last_para_dict['text']) + len(cur_para_dict['text']) < MAX_CHARS_FOR_CHUNK_TRANSLATION):
        return True
    return False

def merge_conj_short_paras(para_text_strs):
    need_merge_last_para_and_cur_para = False
    merged_para_text_strs = []
    for para_dict in para_text_strs:
        #  print(f'[FUCK---FUCK] -=-=-=-=-=- {json.dumps(para, indent=4, ensure_ascii=False)}')
        if (need_merge_last_para_and_cur_para or (para_too_short_and_need_merge(para_dict) and len(merged_para_text_strs) > 0 and if_merged_paras_len_will_not_exceed_thresh(merged_para_text_strs[-1], para_dict))) and len(merged_para_text_strs) > 0 and chunk_types_compatiable_to_merge(merged_para_text_strs[-1], para_dict): # TODO 如果需要把当前段落拼到已处理过的最后一段上，就拼。否则把新段落作为单独一段，加入 merged_para_text_strs。
            last_para = merged_para_text_strs[-1]
            last_para['text'] += "\n"
            last_para['text'] += para_dict['text']
            last_para['page_ids'].extend(para_dict['page_ids'])
            last_para['locations'].extend(para_dict['locations'])
            #  merged_locations = merge_para_locations(last_para, para_dict)
            #  last_para['locations'] = merged_locations
            # TODO 合并后，看看新的段落是否够长了。如果还不够长，need_merge_last_para_and_cur_para 赋值为 True
            merged_para_text_strs[-1] = last_para
        else:
            merged_para_text_strs.append(para_dict)
        if para_too_short_and_need_merge(merged_para_text_strs[-1]):
            need_merge_last_para_and_cur_para = True
        else:
            need_merge_last_para_and_cur_para = False
    #  return para_text_strs
    return merged_para_text_strs

def print_block(block):
    if "paragraphs" not in block:
        return
    paras = block["paragraphs"]
    for para in paras:
        lines = para["lines"]
        for line in lines:
            words = line["words"]
            line_text = ""
            for w_i, word in enumerate(words):
                word_text = word["text"] + " "
                line_text += word_text
            # debug_logger.info(line_text)

def paras2chunks(para_text_strs):
    first_title_found = False
    for i, p in enumerate(para_text_strs):
        if p["chunk_type"] == "title":
            if (not first_title_found):
                p["chunk_type"] = "title"
                first_title_found = True
            else:
                p["chunk_type"] = "h1"

        p["chunk_id"] = str(i)
        p["summary"] = {}
        p["summary_status"] = 0 if (p["chunk_type"] == "title" or p["chunk_type"] == "h1") else 1

    return para_text_strs


def generate_fake_chunks():
    # 构造结果
    chunk_0 = {
        "text": "Chunk 0",
        "page_ids": [0],
        "chunk_id": "0",
        "chunk_type": "h1",
        "summary_status": 0, # 0 表示不用做总结；1 表示需要总结，还没总结；2 表示已经有总结
        "summary": {}
    }

    chunk_1 = {
        "text": "Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. Chunk 1 text. ",
        "page_ids": [1, 2],
        "chunk_id": "1",
        "chunk_type": "normal",
        "summary_status": 1,
        "summary": {}
    }

    chunk_2 = {
        "text": "Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. Chunk 2 text. ",
        "page_ids": [2],
        "chunk_id": "2",
        "chunk_type": "normal",
        "summary_status": 2,
        "summary": {
            "tldr": "hahahaha, this is chunk2"
        }
    }

    return [chunk_0, chunk_1, chunk_2]


def save_chunks_to_db(redis_client, doc_id, chunks):
    max_page_id = -1
    chunk_ids = []
    for chunk in chunks:
        if chunk["summary_status"] != 0:
            chunk_ids.append(chunk["chunk_id"])
    #  redis_client.set(str((doc_id, 'TEXT_CHUNK_IDS')), json.dumps(chunk_ids))

    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        page_id = int(chunk['locations'][0]['page_id'])
        max_page_id = max(max_page_id, page_id)

        # TODO [20230602] 需要扩充一下 chunk 内容。从原来的 text 字段，变更为  real_text 和 text_for_prompt 两部分，前者是真实内容，后者是向上下扩展一两段的结果
        chunk["real_text"] = chunk["text"]
        enriched_text = ""
        if i - 2 >= 0: # 能往前扩充两段。则需要判断这两段是不是太长
            pre_2_chunk = chunks[i-2]["text"]
            pre_1_chunk = chunks[i-1]["text"]
            enriched_text = pre_2_chunk + pre_1_chunk
        elif i - 1 >= 0:
            pre_1_chunk = chunks[i-1]["text"]
            enriched_text = pre_1_chunk
        enriched_text += chunk["text"]
        if i + 2 < len(chunks):
            enriched_text += chunks[i+1]["text"]
            enriched_text += chunks[i+2]["text"]
        elif i + 1 < len(chunks):
            enriched_text += chunks[i+1]["text"]


        chunk["enriched_text"] = enriched_text
        chunk_json_str = json.dumps(chunk)
        redis_client.set(str((doc_id, chunk_id)), chunk_json_str)

    if len(chunk_ids) < 1200 and max_page_id <= 40: # 超过40页的，连VIRT都可以不要了。
        # TODO [20230519] 接下来要把段落们合并后构成的虚拟段落也加入数据库，用于问答时的检索，提高正确参考信息的召回率。TODO 包括 它们的 chunk id，要都写进 VIRTUAL_CHUNK_IDS
        #  MIN_SINGLE_CHUNK_LEN = 200
        # MAX_MERGED_CHUNK_LEN = 3000
        MAX_MERGED_CHUNK_LEN_LIST = [100, 200, 300, 500, 1000]
        virtual_chunk_ids = []
        virtual_index = 0
        used_orig_chunkids_set = set()
        for MAX_MERGED_CHUNK_LEN in MAX_MERGED_CHUNK_LEN_LIST:
            for i in range(0, len(chunks)):
                start_id = i
                start_chunk = chunks[i]
                #  # TODO 检查一下后续段落长度，低于阈值的，就合并进来。
                # TODO 当合并后的总长大于阈值了，就停止合并。
                merged_chunk_ids = [i]
                merged_text = start_chunk["text"]
                for j in range(i + 1, len(chunks)):
                    cur_chunk = chunks[j]
                    cur_chunk_len = len(cur_chunk['text'])
                    if len(merged_text) + cur_chunk_len < MAX_MERGED_CHUNK_LEN:
                        merged_chunk_ids.append(j)
                        merged_text += cur_chunk['text']
                    else:
                        break
                if len(merged_chunk_ids) <= 1: # 即当前段落够长了，未进行有效合并。
                    continue
                if tuple(merged_chunk_ids) in used_orig_chunkids_set:
                    continue
                # 以 virtual chunk 的名字，加入 redis
                virtual_index += 1
                virtual_chunk_id = f'VIRT-{virtual_index}'
                virtual_chunk_ids.append(virtual_chunk_id)
                virtual_chunk = {
                    "text": merged_text,
                    "real_text": merged_text,
                    "enriched_text": merged_text,
                    "real_chunk_ids": merged_chunk_ids
                }
                redis_client.set(str((doc_id, virtual_chunk_id)), json.dumps(virtual_chunk))
                used_orig_chunkids_set.add(tuple(merged_chunk_ids))
        redis_client.set(str((doc_id, 'VIRTUAL_CHUNK_IDS')), json.dumps(virtual_chunk_ids))
        # 把真实和虚拟chunk ID列表合并到一起，方便后面检索
        chunk_ids.extend(virtual_chunk_ids)

    # [20230607] 得限制一下太长的文档。传一本几百页的书的话，下面的 SUB 段落就他妈得有上万个。那 embedding 得算好久好久。实际上，遇到特别长的文档，就先不要算 SUB 的段落了。
    if len(chunk_ids) < 2000 and max_page_id <= 100: # 超过100页的，都不要SUB了。

        # TODO [20230602] 新增大段落拆分成小份的，命名为 SUB-{int}
        sub_chunk_id_from_0 = 0
        sub_chunk_ids = []
        for i, orig_chunk in enumerate(chunks):
            orig_chunk_id = orig_chunk["chunk_id"]
            # TODO 对于 SUB 型的 chunk，其 enriched_text 反正都是一样的。目的都是，只要有一小句话命中了，就把一大块上下关联的段落加进 prompt 里。
            enriched_text = ""
            if i - 2 >= 0: # 能往前扩充两段。则需要判断这两段是不是太长
                pre_2_chunk = chunks[i-2]["text"]
                pre_1_chunk = chunks[i-1]["text"]
                enriched_text = pre_2_chunk + pre_1_chunk
            elif i - 1 >= 0:
                pre_1_chunk = chunks[i-1]["text"]
                enriched_text = pre_1_chunk
            enriched_text += orig_chunk["text"]
            if i + 2 < len(chunks):
                enriched_text += chunks[i+1]["text"]
                enriched_text += chunks[i+2]["text"]
            elif i + 1 < len(chunks):
                enriched_text += chunks[i+1]["text"]

            # [20230602] 需要扩充一下 chunk 内容。从原来的 text 字段，变更为 real_text 和 text_for_prompt 两部分，前者是真实内容，后者是向上下扩展一两段的结果
            # TODO [20230602] 每个 real_text 都应该是一句话？
            sent_tokenize_list = sent_tokenize(orig_chunk["text"]) # 切成小句子
            # TODO 以1句、2句、3句两种粒度
            for j, sent in enumerate(sent_tokenize_list):
                if j + 2 < len(sent_tokenize_list):
                    # TODO 把双句的构造出来
                    chunk = {
                        "text": sent + sent_tokenize_list[j+1] + sent_tokenize_list[j+2],
                        "real_text": sent + sent_tokenize_list[j+1] + sent_tokenize_list[j+2],
                        "enriched_text": enriched_text,
                        "real_chunk_ids": [orig_chunk_id]
                    }
                    sub_chunk_id = f"SUB-{sub_chunk_id_from_0}-{orig_chunk_id}"
                    sub_chunk_ids.append(sub_chunk_id)
                    chunk_json_str = json.dumps(chunk)
                    redis_client.set(str((doc_id, sub_chunk_id)), chunk_json_str)
                    sub_chunk_id_from_0 += 1
                if j + 1 < len(sent_tokenize_list):
                    # TODO 把双句的构造出来
                    chunk = {
                        "text": sent + sent_tokenize_list[j+1],
                        "real_text": sent + sent_tokenize_list[j+1],
                        "enriched_text": enriched_text,
                        "real_chunk_ids": [orig_chunk_id]
                    }
                    sub_chunk_id = f"SUB-{sub_chunk_id_from_0}-{orig_chunk_id}"
                    sub_chunk_ids.append(sub_chunk_id)
                    chunk_json_str = json.dumps(chunk)
                    redis_client.set(str((doc_id, sub_chunk_id)), chunk_json_str)
                    sub_chunk_id_from_0 += 1
                # TODO 把单句的构造出来
                chunk = {
                    "text": sent,
                    "real_text": sent,
                    "enriched_text": enriched_text,
                    "real_chunk_ids": [orig_chunk_id]
                }
                sub_chunk_id = f"SUB-{sub_chunk_id_from_0}-{orig_chunk_id}"
                sub_chunk_ids.append(sub_chunk_id)
                chunk_json_str = json.dumps(chunk)
                redis_client.set(str((doc_id, sub_chunk_id)), chunk_json_str)
                sub_chunk_id_from_0 += 1




        chunk_ids.extend(sub_chunk_ids)
    # debug_logger.info(f'[save_chunks_to_db] chunk_ids = {chunk_ids}')

    redis_client.set(str((doc_id, 'TEXT_CHUNK_IDS')), json.dumps(chunk_ids))

    # 全文加进 redis
    redis_client.set(doc_id, json.dumps(chunks))




"""
把公式干掉。
为了在问答时能有公式参与检索，但在前端渲染时没有公式的段落。进而前端不会拿公式段落来请求其摘要。
"""
def filter_chunks_json(chunks_json):
    res = []
    for i, p in enumerate(chunks_json):
        if "is_formula" in p and p["is_formula"]:
            continue
        if "text" in p and "chunk_type" in p and p['chunk_type'] == "normal" and ((i != len(chunks_json) - 1 and len(p["text"]) < 160) or (i == len(chunks_json) - 1 and len(p["text"]) < 40)):
            continue
        # [20231219] 为了发布会演示好看(《ReAct》)，临时加个丑陋规则，跳过部分论文的作者信息。
        # 首页的、abstract标题(按付凯的norm_str方法做一下归一化后，内容==abstract的)之前的、正文段落。干掉。
        if "text" in p and "chunk_type" in p and p['chunk_type'] == "normal":
            if i+1 < len(chunks_json):
                next_chunk = chunks_json[i+1]
                if "text" in next_chunk and "chunk_type" in next_chunk and next_chunk['chunk_type'] == "h1":
                    if norm_str_(next_chunk['text']) == 'abstract':
                        # print(f'[{cur_func_name()}] Author info: {json.dumps(p, indent=4, ensure_ascii=False)}')
                        # 确实是 abstract 之前的段落了。那如果当前页码是0或1，就直接跳过当前段落。
                        if 'page_ids' in p and p['page_ids'] and p['page_ids'][0] in [0, 1, '0', '1']:
                            continue

        res.append(p)
    return res

def norm_str_(line):
    # 统一小写化，去除空格，只保留英文字符
    new_line = []
    line = line.lower()
    for ch in list(line):
        if ch.isalpha() or ch.isnumeric():
            new_line.append(ch)
    return ''.join(new_line)










