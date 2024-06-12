import json
import re

from qanything_kernel.utils.loader.pdf_to_markdown.core.parser import PdfParser, PlainParser
from qanything_kernel.utils.loader.pdf_to_markdown.convert2markdown import json2markdown
import numpy as np
import os

import time 


class Pdf(PdfParser):
    def __init__(self):
        super().__init__()

    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None):
        res_dir = 'results/'
        timestamp = int(time.time())
        save_dir = os.path.join(res_dir,filename.split('.')[0]+'_'+str(timestamp))
        os.makedirs(save_dir,exist_ok=True)

        json_dir = os.path.join(save_dir,os.path.basename(filename)[:-4]) + '.json'
        basedir = os.path.dirname(json_dir)
        basename = os.path.basename(json_dir)
        markdown_path = os.path.join(basedir,basename.split('.')[0]+'_md')
        os.makedirs(markdown_path,exist_ok=True)
        markdown_dir = os.path.join(markdown_path, basename.split('.')[0]+'.md')
        # image_dir = os.path.join(markdown_path, 'image')
        # os.makedirs(image_dir,exist_ok=True)


        callback(msg="OCR is  running...")
        print("OCR is  running...")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        callback(msg="OCR finished.")
        print("OCR finished.")

        np.set_printoptions(threshold=np.inf)
        from timeit import default_timer as timer
        start = timer()
        self._layouts_rec(zoomin)

        callback(0.63, "Layout analysis finished")
        # self._table_transformer_job(zoomin)
        callback(0.68, "Table analysis finished")
        self._text_merge()
        tbls = self._extract_table_figure(True, zoomin, True, True, markdown_path)
        page_width = max([b["x1"] for b in self.boxes if b['layout_type'] == 'text']) - min([b["x0"] for b in self.boxes if b['layout_type'] == 'text'])
        # self._naive_vertical_merge()
        self._concat_downward()
        self._filter_forpages()
        callback(0.75, "Text merging finished.")
        column_width = np.median([b["x1"] - b["x0"] for b in self.boxes if b['layout_type'] == 'text'])
        text_width = np.argmax(np.bincount([b["x1"] - b["x0"] for b in self.boxes if b['layout_type'] == 'text']))


        # clean mess
        if column_width < page_width / 2 and text_width < page_width / 2:
        # if column_width < self.page_images[0].size[0] / zoomin / 2:
            # print("two_column...................", column_width,
            #       self.page_images[0].size[0] / zoomin / 2)
            self.boxes = self.sort_X_by_page(self.boxes, 0.9 * column_width)
        

        for b in self.boxes:
            b["text"] = re.sub(r"([\t 　]|\u3000){2,}", " ", b["text"].strip())

        def _begin(txt):
            return re.match(
                "[0-9. 一、i]*(introduction|abstract|摘要|引言|keywords|key words|关键词|background|背景|目录|前言|contents)",
                txt.lower().strip())

        if from_page > 0:
            return {
                "title": "",
                "authors": "",
                "abstract": "",
                "sections": [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", "")) for b in self.boxes if
                             re.match(r"(text|title)", b.get("layoutno", "text"))],
                "tables": tbls
            }
        i = 0
        sections = [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", "")) for b in self.boxes[i:] if
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
        
        json.dump(new_sections,open(json_dir,'w'))
        json2markdown(json_dir,markdown_dir)
        print("pdf parser:", timer() - start)


        return new_sections



def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
        Only pdf is supported.
    """
    pdf_parser = None
    if re.search(r"\.pdf$", filename, re.IGNORECASE):
        if not kwargs.get("parser_config", {}).get("layout_recognize", True):
            pdf_parser = PlainParser()
            paper = {
                "title": filename,
                "authors": " ",
                "abstract": "",
                "sections": pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page)[0],
                "tables": []
            }
        else:
            pdf_parser = Pdf()
            paper = pdf_parser(filename if not binary else binary,
                               from_page=from_page, to_page=to_page, callback=callback)
    else:
        raise NotImplementedError("file type not supported yet(pdf supported)")
    return paper


if __name__ == "__main__":
    import sys

    def dummy(prog=None, msg=""):
        pass
    chunk(sys.argv[1], callback=dummy, layout_recognize=True)
