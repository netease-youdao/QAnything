import random
from langchain.schema.document import Document
import re

RANDOM_NUMBER_SET = set()


def remove_escapes(markdown_text):
    pattern = r'\\(.)'
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

    offset = min(total_levels) - 1 if len(total_levels) > 0 else 0
    max_depth = max(total_levels) - min(total_levels) + 1 if len(total_levels) > 0 else 0

    for i, block in enumerate(document):
        if not isinstance(block, list): continue
        for j, item in enumerate(block):
            if item['type'] != 'heading': continue
            document[i][j]['attrs']['level'] -= offset

    return document, offset, max_depth


def _init_node(node_type, title, id_len=4):
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
        'blocks': []
    }


def _get_content_dfs(item):
    def dfs_child(child, lines):
        if 'children' in child:
            for c in child['children']:
                dfs_child(c, lines)
        else:
            if 'raw' in child:
                lines.append(child['raw'])
        return lines

    text_lines = dfs_child(item, [])
    content = '\n'.join(text_lines) + '\n'

    return content


def _add_content_to_block(content, block):
    while content.endswith('\n'):
        content = content[:-1]

    if len(content) > 0:
        content_node = _init_node('ContentNode', '文字内容')
        content_node['content'] = content
        del content_node['blocks']
        block['blocks'].append(content_node)

    return block


def _update_heading_recursive(hierarchy_blocks, heading_depth, content):
    upper_level = max([heading_depth - i for i in range(1, heading_depth + 1) \
                       if hierarchy_blocks[heading_depth - i] is not None])

    # add content
    if heading_depth == len(hierarchy_blocks) or hierarchy_blocks[heading_depth] is None:
        hierarchy_blocks[upper_level] = _add_content_to_block(
            content, hierarchy_blocks[upper_level])
    # update heading tree
    else:
        _update_heading_recursive(hierarchy_blocks, heading_depth + 1, content)
        hierarchy_blocks[upper_level]['blocks'].append(hierarchy_blocks[heading_depth])
        hierarchy_blocks[heading_depth] = None

    return hierarchy_blocks


def get_raw(item):
    if 'raw' in item['children'][0].keys():
        return item['children'][0]['raw']
    else:
        return get_raw(item['children'][0])


def _process_heading(item, hierarchy_blocks, content):
    # print(item)
    heading_depth = item['attrs']['level']
    # heading_text = item['children'][0]['raw']
    heading_text = get_raw(item)

    if heading_depth < len(hierarchy_blocks):
        hierarchy_blocks = _update_heading_recursive(hierarchy_blocks, heading_depth, content)
        hierarchy_blocks[heading_depth] = _init_node('Level#%dHeadingNode' % heading_depth, heading_text)
        content = ''
    else:
        content += '#' * heading_depth + ' ' + heading_text + '\n'

    return hierarchy_blocks, content


def _process_block(block, doc_json, max_heading_depth):
    content = ''
    hierarchy_blocks = [doc_json] + [None] * max_heading_depth

    for item in block:
        if item['type'] == 'heading':
            hierarchy_blocks, content = _process_heading(item, hierarchy_blocks, content)
        elif item['type'] in ['blank_line', 'thematic_break']:
            continue
        elif item['type'] in ['paragraph', 'list', 'block_quote']:
            content += _get_content_dfs(item)
        elif item['type'] in ['block_code', 'block_html']:
            content += item['raw'] + '\n'
        else:
            raise ValueError('Unknown Type %s !!!' % item['type'])

    hierarchy_blocks = _update_heading_recursive(hierarchy_blocks, 1, content)

    return hierarchy_blocks[0]


def _update_node_id_title_dfs(doc_json):
    def dfs_recursive(node, node_id_list=[], title_list=[]):

        node_id_list.append(node['node_id'])
        node['node_id'] = '-'.join(node_id_list)

        title_list.append(node['title'])
        # node['title'] = '-'.join(title_list)
        node['title'] = title_list.copy()

        if 'blocks' in node:
            for block in node['blocks']:
                node_id_list, title_list = dfs_recursive(
                    block, node_id_list, title_list)

        node_id_list = node_id_list[:-1]
        title_list = title_list[:-1]

        return node_id_list, title_list

    dfs_recursive(doc_json)


def parse_markdown_mistune(file_path, doc_title=None, max_heading_depth=2):
    import mistune

    if not file_path.endswith('.md'):
        raise ValueError('Not a markdown file !!!')

    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_content = file.read()
    markdown_content = remove_escapes(markdown_content)
    mistune_parser = mistune.Markdown()
    document = mistune_parser.parse(markdown_content)
    print('Markdown parsing done.')
    document, level_offset, max_depth = _get_heading_level_offset(document)
    if max_heading_depth is None or max_heading_depth <= 0:
        max_heading_depth = max_depth

    file_tile = file_path.split('/')[-1].replace('.md', '')
    doc_title = file_tile if doc_title is None else '-'.join([doc_title, file_tile])
    doc_json = _init_node('DocumentNode', doc_title, id_len=8)

    for block in document:
        if not isinstance(block, list): continue
        doc_json = _process_block(block, doc_json, max_heading_depth)
    # print(doc_json)
    _update_node_id_title_dfs(doc_json)
    # print(doc_json)
    # with open('markdown_test.json','w',encoding='utf-8') as f_w:
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


def convert_node_to_document(node_lists):
    doc_lst = []
    for k, v in node_lists.items():
        for item in v:
            if item['node_type'].startswith('Level'):
                if len(item['child_id_list']) == 0:  #是一个单独的标题，并且没有子节点，可能是markdown解析出现了问题
                    title_lst = []
                    for index, title in enumerate(item['title']):
                        title_lst.append('#' * (index + 1) + ' ' + title)
                    doc = Document(page_content='', metadata={'title_lst': title_lst, 'has_table': False})
                    doc_lst.append(doc)
            if item['node_type'] == 'ContentNode':
                title_lst = []
                for index, title in enumerate(item['title'][:-1]):
                    title_lst.append('#' * (index + 1) + ' ' + title)
                has_table = contains_table(item['content'])
                doc = Document(page_content=item['content'], metadata={'title_lst': title_lst, 'has_table': has_table})
                doc_lst.append(doc)
    return doc_lst


def convert_markdown_to_langchaindoc(md_file):
    doc_json = parse_markdown_mistune(md_file)
    node_lists = _convert_to_node_lists_dfs([doc_json])
    doc_lst = convert_node_to_document(node_lists)
    return doc_lst


if __name__ == '__main__':
    doc_lst = convert_markdown_to_langchaindoc(
        '/ssd8/exec/qinhaibo/code/RAG/release/git/document-layout-parser/results/樊昊天个人简历_1715841225/樊昊天个人简历_md/樊昊天个人简历.md')
    print(doc_lst)
