from qanything_kernel.utils.loader import PdfLoader

# 创建 PdfLoader 类的实例
# 可以自由更改路径
pdf_loader = PdfLoader(filename='/Users/admin/QA/QAnything/xxx.pdf', save_dir_='/Users/admin/QA/QAnything/')

# 调用 load_to_markdown 方法进行转换
markdown_directory = pdf_loader.load_to_markdown()
print(f"Markdown文件在: {markdown_directory}")