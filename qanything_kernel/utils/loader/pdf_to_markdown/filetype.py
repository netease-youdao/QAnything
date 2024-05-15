import pandas as pd
import markdownify


def html2markdown(html_text):
    markdown_text = markdownify.markdownify(html_text)
    return markdown_text


def excel2html(xlsx_path):
    """
    convert excel file to html
    """
    df = pd.read_excel(xlsx_path)
    html = df.to_html()
    return html


def excel2markdown(xlsx_path):
    df = pd.read_excel(xlsx_path)
    html = df.to_html()
    markdown = html2markdown(html)
    return markdown


if __name__ =='__main__':
    excel2html('/ssd8/exec/qinhaibo/code/RAG/release/git/document-layout-parser/第三次培训_数字人产品销售素材20240314.xlsx')