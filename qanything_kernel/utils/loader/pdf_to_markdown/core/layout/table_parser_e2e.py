from .table_rec.pipeline import TableParser
from .table_cls.infer_onnx import TableCls


class TableRecognizer():
    def __init__(self, device):
        print('table model initing...')
        self.table_cls = TableCls(device)
        self.table_parse = TableParser(device)
        print('table model inited...')

    def extract_table(self, table_image, ocr_result):
        table_type = self.table_cls.process(table_image.copy())
        table_html, table_markdown = self.table_parse.process(table_image.copy(), table_type, ocr_result=ocr_result,
                                                              convert2markdown=True)
        return table_html, table_markdown
