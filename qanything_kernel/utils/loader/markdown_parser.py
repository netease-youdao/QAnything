from fsspec.implementations.local import LocalFileSystem
import re
from typing import Dict, List, Optional, Tuple, cast
from langchain.schema.document import Document
import uuid


def get_tqdm_iterable(items, show_progress, desc):
    _iterator = items
    if show_progress:
        try:
            from tqdm.auto import tqdm

            return tqdm(items, desc=desc)
        except ImportError:
            pass
    return _iterator


def default_id_func(i, doc):
    return str(uuid.uuid4())


class PageStruct(object):
    def __init__(self, text='', metadata=None):
        self.text = text
        self.metadata = metadata


class MarkdownReader(object):
    def __init__(self, remove_hyperlinks=True, remove_images=True) -> None:
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        """Convert a markdown file to a dictionary.
        The keys are the headers and the values are the text under each header.
        """
        markdown_tups: List[Tuple[Optional[str], str]] = []
        lines = markdown_text.split("\n")
        current_header = None
        current_text = ""
        for line in lines:
            header_match = re.match(r"^#\s", line)  # 每一个一级标题下方的内容构成一个doc
            if header_match:
                if current_header is not None:
                    if current_text == "" or None:
                        continue
                    markdown_tups.append((current_header, current_text))

                current_header = line
                current_text = ""
            else:
                current_text += line + "\n"
        markdown_tups.append((current_header, current_text))
        if current_header is not None:
            markdown_tups = [
                (cast(str, key).strip(), value)
                for key, value in markdown_tups
            ]
        else:
            markdown_tups = [
                (key, value) for key, value in markdown_tups
            ]
        return markdown_tups

    def remove_images(self, content: str) -> str:
        """Remove images in markdown content."""
        pattern = r"!{1}\[\[(.*)\]\]"
        return re.sub(pattern, "", content)

    def remove_hyperlinks(self, content: str) -> str:
        """Remove hyperlinks in markdown content."""
        pattern = r"\[(.*?)\]\((.*?)\)"
        return re.sub(pattern, r"\1", content)

    def parse_tups(self, filepath):
        """Parse file into tuples"""
        fs = LocalFileSystem()
        with fs.open(filepath, encoding="utf-8") as f:
            content = f.read().decode(encoding="utf-8")
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        return self.markdown_to_tups(content)

    def load_file(self, filepath, meta_info=None):
        tups = self.parse_tups(filepath)
        single_results = []
        for header, value in tups:
            if header is None:
                single_results.append(PageStruct(text=value, metadata={}))
            else:
                single_results.append(
                    PageStruct(text=f"\n\n{header}\n{value}", metadata={})
                )
        return single_results

    def load_data(self, filepaths):
        results = []
        for filepath in filepaths:
            results.extend(self.load_file(filepath))
        return results


class MarkdownParser(object):

    def __init__(self) -> None:
        self.include_metadata = True
        self.id_func = default_id_func
        pass

    def parse_markdown(self, md):
        lines = md.strip().split("\n")
        result = []
        headings = {0: ""}
        content = ""

        for line in lines:
            stripped_line = line.strip()
            match_obj = re.match(r'(#+)\s?(.*?)\s*$', stripped_line)
            if match_obj:  # the line is a heading line
                # if there is already content under the current heading
                if content:
                    # combine current heading with the content
                    full_heading = " ".join(v for k, v in sorted(headings.items()))
                    result.append(full_heading + " " + content)
                    content = ""  # reset before processing next heading

                level = len(match_obj.group(1))
                title = match_obj.group(2)
                headings[level] = title
                # remove all saved subheadings
                headings = {k: v for k, v in headings.items() if k <= level}
            elif stripped_line:  # the line belongs to content
                # combine all content lines
                content = content + " " + stripped_line if content else stripped_line

        # in case the content under the last heading has not been appended
        if content:
            full_heading = " ".join(v for k, v in sorted(headings.items()))
            result.append(full_heading + " " + content)
        return result

    def get_nodes_from_node(self, node):
        """Get nodes from document."""
        # text = node.get_content(metadata_mode=MetadataMode.NONE)
        # print(text)
        # print('**************************')
        text = node.text
        markdown_nodes = []
        metadata: Dict[str, str] = {}
        res = self.parse_markdown(text)
        for node_text in res:
            markdown_nodes.append(
                self._build_node_from_split(node_text.strip(), node, metadata)
            )
        return markdown_nodes

    def _parse_nodes(self, nodes):
        all_nodes = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress=False, desc="Parsing nodes")
        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)
        return all_nodes

    def _build_node_from_split(
            self,
            text_split,
            node,
            metadata, ):
        """Build node from single text split."""
        # node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]
        node = Document(page_content=text_split, metadata={})
        return node

    def get_nodes_from_documents(self, documents):
        nodes = self._parse_nodes(documents)
        return nodes


# DEFAULT_TEXT_QA_PROMPT_TMPL = (
# """参考信息：
# {context_str}
# ---
# 我的问题或指令：
# {query_str}
# ---
# 请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据，原文中有答案的尽量用原文中的表述进行回答。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复
# 你的回复：""")
# new_summary_tmpl = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)
# llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

# reader = MarkdownReader()
# parser = MarkdownParser()
# pages = reader.load_data(['./智慧教育问题.md', './杭研问答总结.md'])
# docs = parser.get_nodes_from_documents(pages)
# for doc in docs:
#     # print(node.node_id)
#     print(doc.text)
#     print('********')
