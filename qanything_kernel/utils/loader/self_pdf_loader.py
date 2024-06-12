import json
import re
from qanything_kernel.utils.loader.pdf_to_markdown.core.parser import PdfParser
from qanything_kernel.utils.loader.pdf_to_markdown.convert2markdown import json2markdown
from qanything_kernel.utils.custom_log import debug_logger
from timeit import default_timer as timer
import numpy as np
import os
import time


class PdfLoader(PdfParser):
    def __init__(self, filename, binary=None, from_page=0, to_page=10000, zoomin=3, save_dir='results/', callback=None):
        super().__init__()
        timestamp = int(time.time())
        # save_dir = os.path.join(root_dir, filename.split('.')[0] + '_' + str(timestamp))
        os.makedirs(save_dir, exist_ok=True)

        self.json_dir = os.path.join(save_dir, os.path.basename(filename)[:-4]) + '.json'
        basedir = os.path.dirname(self.json_dir)
        basename = os.path.basename(self.json_dir)
        self.markdown_path = os.path.join(basedir, basename.split('.')[0] + '_md')
        os.makedirs(self.markdown_path, exist_ok=True)
        self.markdown_dir = os.path.join(self.markdown_path, basename.split('.')[0] + '.md')

        self.filename = filename
        self.binary = binary
        self.from_page = from_page
        self.to_page = to_page
        self.zoomin = zoomin
        self.callback = callback

    def load_to_markdown(self):
        ocr_start = timer()
        self.__images__(
            self.filename if self.binary is None else self.binary,
            self.zoomin,
            self.from_page,
            self.to_page,
            self.callback
        )
        debug_logger.info("OCR finished in %s seconds" % (timer() - ocr_start))

        np.set_printoptions(threshold=np.inf)
        start = timer()
        self._layouts_rec(self.zoomin)

        self._text_merge()
        tbls = self._extract_table_figure(True, self.zoomin, True, True, self.markdown_path)
        try:
            page_width = max([b["x1"] for b in self.boxes if b['layout_type'] == 'text']) - min(
                [b["x0"] for b in self.boxes if b['layout_type'] == 'text'])
            self._concat_downward()
            self._filter_forpages()
            column_width = np.median([b["x1"] - b["x0"] for b in self.boxes if b['layout_type'] == 'text'])
            text_width = np.argmax(np.bincount([b["x1"] - b["x0"] for b in self.boxes if b['layout_type'] == 'text']))

            # clean mess
            if column_width < page_width / 2 and text_width < page_width / 2:
                self.boxes = self.sort_X_by_page(self.boxes, 0.9 * column_width)

            for b in self.boxes:
                b["text"] = re.sub(r"([\t 　]|\u3000){2,}", " ", b["text"].strip())

            if self.from_page > 0:
                return {
                    "title": "",
                    "authors": "",
                    "abstract": "",
                    "sections": [(b["text"] + self._line_tag(b, self.zoomin), b.get("layoutno", "")) for b in self.boxes if
                                 re.match(r"(text|title)", b.get("layoutno", "text"))],
                    "tables": tbls
                }
        except Exception as e:
            debug_logger.warning("Error in Powerful PDF parsing: %s" % e)
        i = 0
        sections = [(b["text"] + self._line_tag(b, self.zoomin), b.get("layoutno", "")) for b in self.boxes[i:] if
                    re.match(r"(text|title|author|reference|content)", b.get("layoutno", "text"))]
        new_sections = {}

        for sec in sections:
            i = 0
            pn = int(sec[0].split('@@')[-1].split('\t')[0])
            top = float(sec[0].split('@@')[-1].split('\t')[3]) + self.page_cum_height[pn - 1]
            right = float(sec[0].split('@@')[-1].split('\t')[2])
            sec_no = str(pn) + '-' + sec[1]
            while i < len(tbls):
                tbl = tbls[i]
                t_pn = int(tbl[1][0][0]) + 1
                t_bottom = float(tbl[1][0][4]) + self.page_cum_height[t_pn - 1]
                t_left = float(tbl[1][0][1])
                tbl_no = tbl[0][1]
                if t_pn > pn:
                    i += 1
                    continue
                if t_bottom < top and t_left < right:
                    new_sections[tbl_no] = {'text': tbl[0][0], 'type': tbl[0][1]}
                    tbls.pop(i)
                    continue
                if t_bottom < top and t_left > right and t_pn < pn:
                    new_sections[tbl_no] = {'text': tbl[0][0], 'type': tbl[0][1]}
                    tbls.pop(i)
                    continue
                i += 1
            if sec_no not in new_sections.keys():
                new_sections[sec_no] = {'text': sec[0].split('@@')[0], 'type': sec[1]}
            else:
                new_sections[sec_no]['text'] += sec[0].split('@@')[0]
        if tbls:
            for tbl in tbls:
                tbl_no = tbl[0][1]
                new_sections[tbl_no] = {'text': tbl[0][0], 'type': tbl[0][1]}

        json.dump(new_sections, open(self.json_dir, 'w'))
        markdown_str = json2markdown(self.json_dir, self.markdown_dir)
        debug_logger.info("PDF Parse finished in %s seconds" % (timer() - start))
        # print(new_sections, flush=True)
        return self.markdown_dir

