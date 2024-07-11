import random
from langchain.schema.document import Document
from qanything_kernel.utils.custom_log import debug_logger, insert_logger
from copy import deepcopy
import re
import mistune

RANDOM_NUMBER_SET = set()


def remove_escapes(markdown_text):
    # 使用正则表达式匹配转义字符
    pattern = r'\\(.)'

    # 使用re.sub函数替换匹配的字符
    cleaned_text = re.sub(pattern, r'\1', markdown_text)

    return cleaned_text


def contains_table(text):
    lines = text.split('\n')
    if len(lines) < 2:
        return False
    has_separator = False
    for i in range(len(lines) - 1):
        if '|' in lines[i] and '|' in lines[i + 1]:
            separator_line = lines[i + 1].strip()
            if separator_line.startswith('|') and separator_line.endswith('|'):
                separator_parts = separator_line[1:-1].split('|')
                if all(part.strip().startswith('-') and len(part.strip()) >= 3 for part in separator_parts):
                    has_separator = True
                    break
    return has_separator


def _get_heading_level_offset(document):
    total_levels = []
    for block in document:
        if not isinstance(block, list): continue
        total_levels += [item['attrs']['level'] for item in block if item['type'] == 'heading']
    # print(total_levels)
    offset = min(total_levels) - 1 if len(total_levels) > 0 else 0
    max_depth = max(total_levels) - min(total_levels) + 1 if len(total_levels) > 0 else 0

    for i, block in enumerate(document):
        if not isinstance(block, list): continue
        for j, item in enumerate(block):
            if item['type'] != 'heading': continue
            document[i][j]['attrs']['level'] -= offset

    # with open('markdown_test2.json','w',encoding='utf-8') as f_w:
    #     json.dump(document[0],f_w,ensure_ascii=False)

    return document, offset, max_depth


def _init_node(node_type, title, coord=None, id_len=4):
    while True:
        random_number = random.randint(0, 16 ** id_len - 1)
        if random_number in RANDOM_NUMBER_SET: continue
        RANDOM_NUMBER_SET.add(random_number)
        node_id = format(random_number, 'x').zfill(id_len)
        break

    return {
        'node_id': node_id,
        'node_type': node_type,
        'title': title,
        'coord': coord,
        'blocks': []
    }


def _get_content_dfs(item):
    def dfs_child(child, lines):
        if child['type'] == 'image':
            if 'title' in child['attrs']:
                lines.append("![figure]" + '(' + child['attrs']['url'] + ' ' + child['attrs']['title'] + ' ' + ')')
            else:
                lines.append("![figure]" + '(' + child['attrs']['url']+ ')')
        else:
            if 'children' in child:
                for c in child['children']:
                    dfs_child(c, lines)
            else:
                if 'raw' in child:
                    lines.append(child['raw'])
        return lines

    text_lines = dfs_child(item, [])
    content = '\n'.join(text_lines) + '\n'
    coord = [item['page_id'], item['bbox'], 'content']
    return content, coord


def _add_content_to_block(content, block, coord):
    while content.endswith('\n'):
        content = content[:-1]

    if len(content) > 0:
        content_node = _init_node('ContentNode', '文字内容')
        content_node['content'] = content
        content_node['coord'] = coord
        del content_node['blocks']
        block['blocks'].append(content_node)

    return block


def _update_heading_recursive(hierarchy_blocks, heading_depth, content, coord):
    upper_level = max([heading_depth - i for i in range(1, heading_depth + 1) \
                       if hierarchy_blocks[heading_depth - i] is not None])

    # add content
    if heading_depth == len(hierarchy_blocks) or hierarchy_blocks[heading_depth] is None:
        hierarchy_blocks[upper_level] = _add_content_to_block(
            content, hierarchy_blocks[upper_level], coord)
    # update heading tree
    else:
        _update_heading_recursive(hierarchy_blocks, heading_depth + 1, content, coord)
        hierarchy_blocks[upper_level]['blocks'].append(hierarchy_blocks[heading_depth])
        hierarchy_blocks[heading_depth] = None

    return hierarchy_blocks


def get_raw(item):
    if 'raw' in item['children'][0].keys():
        return item['children'][0]['raw']
    else:
        return get_raw(item['children'][0])


def _process_heading(item, hierarchy_blocks, content, coord):
    # print(item)
    heading_depth = item['attrs']['level']
    # heading_text = item['children'][0]['raw']
    heading_text = get_raw(item)

    if heading_depth < len(hierarchy_blocks):
        hierarchy_blocks = _update_heading_recursive(hierarchy_blocks, heading_depth, content, coord)
        hierarchy_blocks[heading_depth] = _init_node('Level#%dHeadingNode' % heading_depth, heading_text,
                                                     coord=[item['page_id'], item['bbox'], 'heading'])
        # hierarchy_blocks[heading_depth] = _init_node('Level#%dHeadingNode' %heading_depth, heading_text)
        content = ''
        coord = []
        # coord.append([item['page_id'],item['paragraph_id'],'heading'])

    else:
        content += '#' * heading_depth + ' ' + heading_text + '\n'

    return hierarchy_blocks, content, coord


def _process_block(block, doc_json, max_heading_depth):
    content = ''
    coord = []
    hierarchy_blocks = [doc_json] + [None] * max_heading_depth

    for item in block:
        if item['type'] == 'heading':
            hierarchy_blocks, content, coord = _process_heading(item, hierarchy_blocks, content, coord)
        elif item['type'] in ['blank_line', 'thematic_break']:
            continue
        elif item['type'] in ['paragraph', 'list', 'block_quote']:
            # content += _get_content_dfs(item)
            content_item, coord_item = _get_content_dfs(item)
            content += content_item
            coord.append(coord_item)
        elif item['type'] in ['block_code', 'block_html']:
            content += item['raw'] + '\n'
            coord.append([item['page_id'], item['bbox'], 'content'])
        else:
            raise ValueError('Unknown Type %s !!!' % item['type'])

    hierarchy_blocks = _update_heading_recursive(hierarchy_blocks, 1, content, coord)

    return hierarchy_blocks[0]


def _update_node_id_title_dfs(doc_json):
    def dfs_recursive(node, node_id_list=[], title_list=[], coord_list=[]):
        node_id_list.append(node['node_id'])
        node['node_id'] = '-'.join(node_id_list)
        title_list.append(node['title'])
        coord_list.append(node['coord'])
        # node['title'] = '-'.join(title_list)
        node['title'] = title_list.copy()
        node['coord'] = coord_list.copy()
        if 'blocks' in node:
            for block in node['blocks']:
                node_id_list, title_list, coord_list = dfs_recursive(
                    block, node_id_list, title_list, coord_list)
        node_id_list = node_id_list[:-1]
        title_list = title_list[:-1]
        coord_list = coord_list[:-1]

        return node_id_list, title_list, coord_list

    dfs_recursive(doc_json)


def extract_paragraph(page_content):
    # print(page_content)
    bbox_lst = []
    full_page_content = ''
    for item in page_content:
        if item['text'] == '\n\n':
            continue
        # print(item)
        bbox = item['boundingBox']
        # bbox = [int(float(_bb_cord)) for _bb_cord in  item['boundingBox'].split(',')]
        bbox_lst.append(bbox)
        full_page_content += item['text']
    return bbox_lst, full_page_content


def parse_markdown_mistune(file_name, pages, doc_title=None, max_heading_depth=5):
    mistune_parser = mistune.Markdown()
    all_document = []
    for page in pages:
        # insert_logger.info(page)
        page_content = page['page_content']
        page_coord, page_content = extract_paragraph(page_content)
        page_id = page['page_id']
        page_content = remove_escapes(page_content)
        document = mistune_parser.parse(page_content)
        paragraph_id = 0
        for item in document[0]:
            if item['type'] != "blank_line":
                item['page_id'] = page_id
                paragraph_id += 1
        assert len(page_coord) == paragraph_id
        paragraph_id = 0
        for item in document[0]:
            if item['type'] != "blank_line":
                item['bbox'] = page_coord[paragraph_id]
                for k,v in item.items():
                    if k != 'children':
                        pass
                    else:
                        lines = ''
                        for child in item['children']:
                            if child['type'] == 'text':
                                lines += child['raw']
                            elif child['type'] == 'image':
                                if 'title' in child['attrs']:
                                    lines += "![figure]" + '(' + child['attrs']['url'] + ' ' + child['attrs']['title'] + ' ' + ')'
                                else:
                                    lines += "![figure]" + '(' + child['attrs']['url']+ ')'

                        item['children'] = [{'type':'text','raw':lines}]
                paragraph_id += 1
        all_document.extend(document[0])

    # with open('./markdown_page/markdown_test.json','w',encoding='utf-8') as f_w:
    #     json.dump(all_document,f_w,ensure_ascii=False)
    # document = [document,None]
    # print(document)
    document = all_document
    document = [document, None]
    document, level_offset, max_depth = _get_heading_level_offset(document)
    # print(max_depth)
    if max_heading_depth is None or max_heading_depth <= 0:
        max_heading_depth = max_depth

    doc_title = file_name if doc_title is None else '-'.join([doc_title, file_name])
    doc_json = _init_node('DocumentNode', doc_title, id_len=8)

    # with open('./markdown_page/markdown_test2.json','w',encoding='utf-8') as f_w:
    #     json.dump(doc_json,f_w,ensure_ascii=False)

    for block in document:
        # print(block)
        # print('***********************')
        if not isinstance(block, list): continue
        doc_json = _process_block(block, doc_json, max_heading_depth)
    # print(doc_json)
    _update_node_id_title_dfs(doc_json)
    # print(doc_json)
    # with open('./markdown_page/markdown_test2.json','w',encoding='utf-8') as f_w:
    #     json.dump(doc_json,f_w,ensure_ascii=False)
    print('Tree building done.')
    return doc_json


def _convert_to_node_lists_dfs(parsing_json):
    node_lists = {}

    def traverse_and_group_by_depth(node, depth, group):
        node_info = node.copy()
        if 'blocks' in node_info:
            node_info['child_id_list'] = [block['node_id'] for block in node_info['blocks']]
            del node_info['blocks']
        if depth not in node_lists.keys():
            node_lists[depth] = [node_info]
        else:
            node_lists[depth].append(node_info)
        if 'blocks' not in node.keys():
            return
        for child in node['blocks']:
            traverse_and_group_by_depth(child, depth + 1, node_lists)

    for doc_json in parsing_json:
        traverse_and_group_by_depth(doc_json, 0, node_lists)

    return node_lists


def extract_inner_lists(nested_list):
    result = []
    for i in nested_list:
        if isinstance(i, list):
            if len(i) > 0 and isinstance(i[0], list):
                result.extend(extract_inner_lists(i))
            else:
                result.append(i)
    return result


def convert_node_to_document(node_lists):
    doc_lst = []
    for k, v in node_lists.items():
        for item_ori in v:
            item = deepcopy(item_ori)
            if item['node_type'].startswith('Level'):
                if len(item['child_id_list']) == 0:  # 是一个单独的标题，并且没有子节点，可能是markdown解析出现了问题
                    title_lst = []
                    coord = extract_inner_lists(item['coord'])
                    for coord_item in coord:
                        bbox = [int(float(_bb_cord)) for _bb_cord in coord_item[1].split(',')]
                        coord_item[1] = bbox
                    coord.sort(key=lambda l: (l[1][1], l[1][0]))
                    for index, title in enumerate(item['title']):
                        title_lst.append('#' * (index + 1) + ' ' + title)
                    doc = Document(page_content='',
                                   metadata={'title_lst': title_lst, 'has_table': False, 'coord_lst': coord})
                    doc_lst.append(doc)
            if item['node_type'] == 'ContentNode':
                title_lst = []
                coord = extract_inner_lists(item['coord'])
                for coord_item in coord:
                    bbox = [int(float(_bb_cord)) for _bb_cord in coord_item[1].split(',')]
                    coord_item[1] = bbox
                coord.sort(key=lambda l: (l[1][1], l[1][0]))
                for index, title in enumerate(item['title'][:-1]):
                    title_lst.append('#' * (index + 1) + ' ' + title)
                has_table = contains_table(item['content'])
                doc = Document(page_content=item['content'],
                               metadata={'title_lst': title_lst, 'has_table': has_table, 'coord_lst': coord})
                doc_lst.append(doc)
    return doc_lst

def create_node_dfs(item,doc_lst,all_nodes):
    if item['node_type'].startswith('Level'):
        if len(item['child_id_list']) == 0:
            title_lst = []
            coord = extract_inner_lists(item['coord'])
            new_coord = []
            for coord_item in coord:
                bbox = [int(float(_bb_cord)) for _bb_cord in coord_item[1].split(',')]
                new_coord_item = coord_item.copy()
                new_coord_item[1] = bbox
                new_coord.append(new_coord_item)
            new_coord.sort(key=lambda l: (l[1][1], l[1][0]))
            for index,title in enumerate(item['title']):
                title_lst.append('#'*(index+1)+' '+ title)
            doc = Document(page_content='',metadata={'title_lst':title_lst,'has_table':False,'coord_lst':new_coord})
            doc_lst.append(doc)
        else:
            for child_id in item['child_id_list']:
                create_node_dfs(all_nodes[child_id],doc_lst,all_nodes)
    elif item['node_type'] == 'ContentNode':
        title_lst = []
        coord = extract_inner_lists(item['coord'])
        new_coord = []
        for coord_item in coord:
            bbox = [int(float(_bb_cord)) for _bb_cord in coord_item[1].split(',')]
            new_coord_item = coord_item.copy()
            new_coord_item[1] = bbox
            new_coord.append(new_coord_item)
        new_coord.sort(key=lambda l: (l[1][1], l[1][0]))
        for index,title in enumerate(item['title'][:-1]):
            title_lst.append('#'*(index+1)+' '+title)
        has_table = contains_table(item['content'])
        doc = Document(page_content=item['content'],metadata={'title_lst':title_lst,'has_table':has_table,'coord_lst':new_coord})
        doc_lst.append(doc)
    else:
        for child_id in item['child_id_list']:
            create_node_dfs(all_nodes[child_id],doc_lst,all_nodes)


def convert_node_to_document_dfs(node_lists):
    doc_lst = [ ]
    all_nodes = {}
    for k,v in node_lists.items():
        for item_ori in v:
            all_nodes[item_ori['node_id']] = item_ori
    first_key = next(iter(all_nodes))  # 获取第一个键
    root_node = all_nodes[first_key]
    create_node_dfs(root_node,doc_lst,all_nodes)
    return doc_lst




def convert_markdown_to_langchaindoc(file_name, pages):
    doc_json = parse_markdown_mistune(file_name, pages)
    node_lists = _convert_to_node_lists_dfs([doc_json])
    # doc_lst = convert_node_to_document(node_lists)
    doc_lst = convert_node_to_document_dfs(node_lists)
    return doc_lst

def recover_langchaindoc_to_page(doc_lst):
    """
    将doc_list 恢复为原始的markdown的string
    """
    title_dict = {}
    markdown_str = ''
    for doc in doc_lst:
        # print(doc)
        for title in doc.metadata['title_lst']:
            if title not in title_dict.keys():
                title_dict[title] = 1
                markdown_str += title + '\n'
    with open('./markdown_page/recover.md','w',encoding='utf-8') as f_w:
        f_w.write(markdown_str)
    




if __name__ == '__main__':
    doc_lst = convert_markdown_to_langchaindoc(
        '/ssd8/exec/qinhaibo/code/RAG/release/git/document-layout-parser/markdown_page/test_1.md')
    # print(doc_lst)
    for doc in doc_lst:
        print(doc)
        print('*************************')

    # convert_md('/ssd8/exec/qinhaibo/code/RAG/release/git/document-layout-parser/markdown_page/0a1e5f79bb75425cba8adbbf051996ac.md')
