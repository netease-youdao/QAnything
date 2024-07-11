import json
import logging

search_result_example = {
    'url': "https://github.com/lhao499/RingAttention",
    'title': "Transformers with Arbitrarily Large Context - GitHub",
    'content': "Navigation Menu\nSearch code, repositories, users, issues, pull requests...\nProvide feedback\nWe read every piece of feedback, and take your input very seriously.\nSaved searches\nUse saved searches to filter your results more quickly\nTo see all available qualifiers, see our documentation.\nTransformers with Arbitrarily Large Context\nLicense\nlhao499/RingAttention\nFolders and files\nLatest commit\nHistory\nbpt\nbpt\ndocs\ndocs\nscripts\nscripts\ntests\ntests\n.gitignore\n.gitignore\nLICENSE\nLICENSE\nREADME.md\nREADME.md\nRepository files navigation\nRing Attention with Blockwise Transformers for Near-Infinite Context\nHao Liu, Matei Zaharia, Pieter Abbeel\nPaper: https://arxiv.org/abs/2310.01889\nBlockwise Parallel Transformer for Large Context Models\nHao Liu, Pieter Abbeel\nPaper: https://arxiv.org/abs/2305.19370\nThis codebase provides the implementation of the Ring Attention with Blockwise Transformers. The model is described in the paper Ring Attention with Blockwise Transformers for Near-Infinite Context and Blockwise Parallel Transformer for Large Context Models.\nBlockwise Parallel Transformers (BPT) compute attention and feedforward in a blockwise manner, allowing for the training and inference of sequences up to four times longer than those manageable by standard memory-efficient attention methods, such as flash attention.\nRing Attention with Blockwise Parallel Transformers enables training sequences up to a length of 'number of devices' times longer than those possible with BPT.",
}


async def article_generation(**payload):
    """__payload_example__
    {
        "topic": str, # 文章标题
        "raw_outline": str, # outline_generation 接口生成的原始大纲(包含所有层级)
        "usr_outline": str, # 用户直接提供或者修改的大纲(仅包含一级标题# 和二级标题## )
        "kb_ids": List[str], # 知识库列表（包含用户创建和联网搜索接口创建的知识库）
        "user_id": str, # 用户Id
        "related_docs": Dict[Dict[List[str]]], # 大纲二级标题及其相关文档[doc_id]的映射
        "urls": {file_id: url}, # 生成大纲联网检索返回的所有相关URL
    }
    """

    async def article_generator(sections):
        for section in sections:
            logging.info(f"[debug] article_generation section={section}")
            yield section

    """
    # 按章节顺序依次返回正文部分内容, 最后一次返回引用来源
    # section_content 字段返回 [一级标题, 二级标题+\n\n+正文内容]
    # section_index 字段返回 [一级标题 Index, 二级标题 Index]
    sections = [
        {
            'section_content': ['# Challenges in Long-Sequence Processing',
                                "## Quadratic Complexity in Attention Mechanisms\n\nThe Transformer architecture, introduced in 2017 by Vaswani et al. ([1]), relies heavily on self-attention mechanisms, which are at the core of its success in sequence modeling tasks. Self-attention connects every position of the input sequence, allowing for global interactions between elements ([2]). However, this comes with a computational challenge: the quadratic complexity of self-attention ([2]).\n\nIn mathematical terms, the computational complexity of self-attention is expressed as O(n^2 * d), where n is the sequence length and d is the representation dimension ([2]). This means that as the sequence length increases, the computational load grows exponentially, making it impractical for tasks requiring long contexts, such as document summarization or DNA sequence analysis ([3]). The quadratic dependency on the sequence length limits the length of the input that can be processed, often necessitating fixed-length inputs like BERT's 512-token limit ([4]).\n\nThis inherent limitation poses a significant obstacle for tasks that demand longer context, as it can lead to information loss when the sequence length is truncated ([5]). To address this, researchers have proposed various approximation techniques to reduce the computational complexity from quadratic to linear, enabling the efficient processing of long sequences ([6])."],
            'section_index': [0, 0],
            'reference': {}
        },
        {
            'section_content': ['# Challenges in Long-Sequence Processing',
                                '## Memory Constraints in Serving Long-Context LLMs\n\nLong-context LLMs, with their ever-increasing context window sizes, present significant memory management challenges for serving systems (1). The primary issue lies in the storage and retrieval of the intermediate states, known as the key-value cache (2). As the sequence length grows, the size of the key-value cache grows linearly, often surpassing the GPU memory capacity (3). For instance, processing a 1M-token input with a Large World Model (LWM) requires a cache size of 488GB, which is unfeasible with current GPU hardware (4).\n\nExisting approaches, such as model parallelism (5) and sequence parallelism (6), are insufficient for addressing the issue effectively. Model parallelism partitions the model across GPUs, while sequence parallelism divides input sequences, but both rely on static configurations that do not adapt to varying sequence lengths or the distinct requirements of the prefill and decoding phases (7). This static allocation can lead to underutilization of resources or memory fragmentation (8).\n\nIn the preprocessing phase, the memory demand is high due to the computation of all input tokens, while the decoding phase requires less memory as only a few tokens are processed (9). Patel et al. (2023) and Holmes et al. (2024) propose chunked prefetching, but this method introduces interference between phases (10). Disaggregation of the phases (Zhong et al., 2024) mitigates interference but incurs high communication overhead due to cache migrations (11) and exacerbates GPU memory fragmentation (12).\n\nThese memory constraints motivate the need for a dynamic solution, like Elastic Sequence Parallelism (ESP), that can adapt to the varying demands of the varying-length requests and the distinct phases, thus overcoming the limitations of existing methods.'],
            'section_index': [0, 1],
            'reference': {}
        },
        {
            'section_content': ['# Challenges in Long-Sequence Processing',
                                "## Information Loss in the Middle: The Challenge of Long Sequence Understanding\n\nIn the realm of long-context language modeling, a crucial concern arises from the potential loss of information when processing sequences with thousands or even millions of tokens[1]. The challenge lies in understanding that, despite the advancements in architectures and embedding techniques[2], current LLMs struggle to maintain coherence and reasoning over extended input lengths. This is particularly evident in the LongICLBench study, where models were found to struggle with tasks requiring longer demonstrations[1].\n\nAs demonstrated by the analysis, models tend to favor predictions from later parts of the sequence, potentially overlooking crucial information in the middle[1]. This phenomenon indicates a deficiency in the model's ability to retain and reason about context when the input becomes excessively long. The issue can be exacerbated by the presence of dense, fine-grained classification tasks, where understanding the nuances of the input is essential[1].\n\nThe sensitivity to instance position in the prompt further exacerbates the problem[1]. For example, GPT4-turbo exhibited a decline in performance when instances were positioned in certain areas of the prompt. This highlights the need for better mechanisms to handle the middle part of the sequence, as it might be crucial for maintaining contextual understanding and avoiding biases in prediction.\n\nTo overcome this issue, researchers have proposed memory augmentation and extrapolation techniques[3], as well as architectural innovations[4]. However, the study shows that these approaches have not yet fully addressed the challenge of long-context understanding, suggesting that further research is necessary to ensure that LLMs can effectively process and reason over the entirety of long sequences."],
            'section_index': [0, 2],
            'reference': {}
        },
        # 最后一次返回内容:
        # section_content 和 section_index字段为空
        # reference 字段返回引用来源 (引用来源可以是URL来源或者知识库doc_id来源)
        {
            'section_content': [],
            'section_index': [],
            'reference': {
                # url_to_info 引用来源Index
                'url_to_unified_index': {
                    'https://medium.com/@lukas.noebauer/the-big-picture-transformers-for-long-sequences-890cc0e7613b': 1,
                    'https://arxiv.org/html/2402.15290v1': 2,
                    'https://arxiv.org/html/2210.09298v1': 3,
                    '3062cf1c02c143b2ade31a158c6fdbb8_1': 4,
                    'https://arxiv.org/html/2404.02060v1': 5
                },
                # url_to_info 引用来源 url/doc_id 需要根据实际情况判断:
                # 1. url以http://或者https://开头
                # 2. 请求 https://qanything-online-test.site.youdao.com/api/local_doc_qa/get_doc 入参: payload = {'doc_id': 'url_or_doc_id'} 返回500 Error则表示为url来源, 否则为知识库来源
                'url_to_info': {
                    'https://medium.com/@lukas.noebauer/the-big-picture-transformers-for-long-sequences-890cc0e7613b':
                        {
                            'url': 'https://medium.com/@lukas.noebauer/the-big-picture-transformers-for-long-sequences-890cc0e7613b',
                            'metadata': {'user_id': 'storm_test__26229', 'kb_id': 'KBxxx_240328', 'file_id': 'xxx',
                                         'file_name': 'https://medium.com/@lukas.noebauer/the-big-picture-transformers-for-long-sequences-890cc0e7613b',
                                         'nos_key': '', 'faq_dict': {}, 'source_info': {}, 'doc_id': ''}},
                    'https://arxiv.org/html/2402.15290v1':
                        {'url': 'https://arxiv.org/html/2402.15290v1',
                         'metadata': {'user_id': 'storm_test__26229', 'kb_id': 'KBxxx_240328', 'file_id': 'xxx',
                                      'file_name': 'https://arxiv.org/html/2402.15290v1', 'nos_key': '', 'faq_dict': {},
                                      'source_info': {}, 'doc_id': ''}},
                    'https://arxiv.org/html/2210.09298v1':
                        {'url': 'https://arxiv.org/html/2210.09298v1',
                         'metadata': {'user_id': 'storm_test__26229', 'kb_id': 'KBxxx_240328', 'file_id': 'xxx',
                                      'file_name': 'https://arxiv.org/html/2210.09298v1', 'nos_key': '', 'faq_dict': {},
                                      'source_info': {}, 'doc_id': ''}},
                    '3062cf1c02c143b2ade31a158c6fdbb8_1':
                        {'url': '3062cf1c02c143b2ade31a158c6fdbb8_1', 'metadata': {'user_id': 'storm_test__26229',
                                                                                   'kb_id': 'KBc604f86bd2724c5e98025d641499627e_240328',
                                                                                   'file_id': '3062cf1c02c143b2ade31a158c6fdbb8',
                                                                                   'file_name': 'Index51_search_result.txt',
                                                                                   'nos_key': 'zhiyun/docqa/qanything/local_file/storm_test__26229/KBc604f86bd2724c5e98025d641499627e_240328/3062cf1c02c143b2ade31a158c6fdbb8/Index51_search_result.txt',
                                                                                   'faq_dict': {}, 'source_info': {},
                                                                                   'doc_id': '3062cf1c02c143b2ade31a158c6fdbb8_1'}},
                    'https://arxiv.org/html/2404.02060v1':
                        {'url': 'https://arxiv.org/html/2404.02060v1',
                         'metadata': {'user_id': 'storm_test__26229', 'kb_id': 'KBxxx_240328', 'file_id': 'xxx',
                                      'file_name': 'https://arxiv.org/html/2404.02060v1', 'nos_key': '', 'faq_dict': {},
                                      'source_info': {}, 'doc_id': ''}}
                }
            }
        }
    ]
    """

    sections = [
        {'code': 200, 'msg': 'success', 'section_content': ['# 1.概述', '## 1.1 QAnything: 网易有道开源的知识库问答系统\n\nQAnything是由网易有道自主研发并开源的知识库问答引擎，其设计初衷在于实现“万物皆可问”的目标，通过集成先进的检索增强生成技术(RAG)[1][3]，使得用户能够针对多种格式的文档，如PDF、Word、PPT、Excel和图片等，进行直接的问答交互[1][3][5]。该系统的核心优势在于结合了用户私有数据与大型语言模型的力量，即使在纯本地部署的情况下，也能提供类似ChatGPT的使用体验，且系统对硬件要求相对亲民，所有核心组件的显存占用控制在16G以内[1][3]。\n\nQAnything的架构设计围绕RAG展开，内置优化的自研embedding和rerank模型，支持文档内容的精确检索与理解[1][5]。在技术细节上，该系统采用了高效的指令微调策略，包括LoRA（Low-Rank Adaptation）方法来降低成本并加速模型的训练过程，特别是在处理大规模指令微调数据时，通过特定的参数配置提升了模型效果[2]。\n\n自2023年起，QAnything迅速在开发者社区获得认可，不仅在GitHub上收获超过7000星标，其语义嵌入模型BCEmbedding更是每月获得超60万次下载，彰显了其在实际应用中的高价值和广泛需求[4][5]。随着系统的不断迭代，最新的1.3.\n\nQAnything的成功开源及其在多个产品中的应用，如有道翻译、有道速读和内部客服系统，不仅展示了网易有道在文档处理和自然语言处理技术的深厚积累，也为行业提供了探索大模型本地化应用的新范例[1][3]。'], 'section_index': [0, 0]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 1.概述', '## 1.2 基于检索增强生成 (RAG) 技术\n\nQAnything，由网易有道自主研发的知识库问答引擎，巧妙地融合了检索增强生成技术（RAG），代表了当前自然语言处理领域的一大进步[5][3]。这一技术的核心在于结合大规模语言模型（LLM）与高效文档解析能力，特别是在处理PDF、Word、Excel、PowerPoint、图片等多种格式文档时展现出了卓越的问答功能[1][3]。QAnything的构建基于深厚的文档翻译技术积累，利用先进的Transformer架构模型，与当前研究领域的LLM紧密相连，确保了翻译与问答服务的高质量[1]。\n\nRAG系统的应用并非简单的信息检索与模型输出叠加，而是面对文档多样性和复杂布局的挑战，通过精准的文档解析和OCR技术，确保信息提取的准确性和完整性[1]。这一系统特别注重在处理含有逻辑关联的文本片段和图表数据时，维持信息的原貌，从而提高问答的准确性[1]。通过Retrieval Augmented Generation，QAnything能够利用外部知识源，即便在存在不相关噪声的情况下，仍展现出较强的鲁棒性，这一点在对比评估中得到了证实，如Qwen-7B-QAnything模型的评测结果显示[6]。\n\n此外，QAnything在技术实现上，不仅依赖于强大的语言处理核心，还通过指令微调策略优化模型，包括利用LoRA方法进行低成本快速探索，随后采用全参数微调以获得最佳效果，这些策略显著提升了模型针对多样化指令的适应性和问答的准确性[2]。其系统设计支持本地部署，提供了FasterTransformer、vLLM和Huggingface Transformers作为推理框架选项，满足不同部署需求，展现了高度的灵活性和实用性[2]。\n\n综上所述，QAnything通过其创新的RAG实施，不仅在技术层面推动了文档问答领域的界限，还在实际应用中，如有道翻译、速读产品及内部业务中，证明了其强大效能和广泛适用性，成为市场上一个值得关注的先进解决方案[5][7]。'], 'section_index': [0, 1]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 2.技术特点', '## 2.1 两阶段检索机制\n\nQAnything本地知识库问答系统采纳了一种创新的两阶段检索机制，该机制显著提高了从海量数据中精准提取信息的能力[8]。这一机制分步进行，首先在第一阶段通过高效的索引和匹配算法迅速筛选出与用户问题相关的文档或段落，确保快速召回潜在答案来源[8]。此步骤侧重于广度，旨在最大限度地包容可能含有所需信息的候选资料。\n\n随后的第二阶段，则专注于深度分析，利用生成式应用对初步筛选出的候选文档进行细致处理[8]。此阶段融合了自然语言处理和机器学习技术，以精准理解问题意图，并从中提炼出最贴合问题的答案[8]。这种设计不仅强化了系统的精确性，还通过结合快速检索与深度分析，优化了信息提取的效率与准确性[9]。\n\n两阶段检索的策略有效地解决了数据互相干扰的问题，实验证明，在数据不变的情况下，问答准确率由52.8%提升到了65.'], 'section_index': [1, 0]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 2.技术特点', '## 2.2 支持海量数据处理\n\nQAnything本地知识库问答系统的一大突出特点是其对海量数据处理的能力。该系统设计用于应对数字化时代信息爆炸的挑战，无论数据来源是书籍、论文、网页还是其他格式的知识资源，QAnything都能够迅速进行索引和存储，构建起一个内容丰富的知识库[8][11]。这背后的技术支撑，尤其是在处理大规模数据方面，是通过高效的索引技术和两阶段检索机制实现的。在第一阶段，系统利用先进的索引和匹配算法，能够在短时间内筛选出与用户问题相关的文档或段落，即便是面对海量数据集也能够保持高效运行[8]。这种能力确保了即使数据量巨大，系统也能快速响应，为用户提供即时的答案检索服务[11]。\n\nQAnything通过其独特的两阶段检索机制，解决了大规模数据检索时常见的效率与准确性退化问题。随着数据量的增加，传统的单一阶段检索方法可能会遭遇检索效果的下降，而QAnything通过后续的reranking阶段显著提升了准确性，实现了数据量与检索效果正相关的优化目标[12]。这一机制的高效运作，结合BCEmbedding模型提供的强大双语和跨语种语义表征能力，保证了系统在处理多语言、多领域数据时的灵活性和准确性[10]。因此，QAnything不仅支持广泛的数据格式，还能确保在数据规模不断增长的情况下，维持甚至提升其问答服务的质量[11]。'], 'section_index': [1, 1]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 2.技术特点', '## 2.3 双语和跨语种问答能力\n\nQAnything在双语和跨语种问答方面展现出了显著的技术优势，这得益于其核心组件BCEmbedding的强力支持。BCEmbedding通过整合大规模的开源数据集，包括摘要、翻译、语义改写和问答等多种类型的数据，经过网易有道成熟的翻译引擎处理，形成了一个既能处理中文又能处理英文，乃至实现跨语种检索的强大模型[10]。这一特性使得QAnything不仅限于单一语言环境，而是跨越语言障碍，为用户提供全球化的问题解答服务。特别是在处理包含专业术语或特定领域知识的问题时，它的跨语种能力确保了广泛适用性和准确性，增强了系统的国际化交互体验[8]。\n\n通过结合中英双语和跨语种数据集，BCEmbedding训练出的模型能够覆盖教育、医疗、法律、金融等多个领域的应用场景，实现了多业务场景的开箱即用功能。这种设计策略不仅提升了模型的泛化能力，还确保了在处理跨文化知识查询时的高效和精准[10]。此外，QAnything的两阶段检索机制进一步加强了其跨语种问答的效能，尤其是在大规模数据集中，通过高效的初步检索和后续的精确排序，确保了即使在语言差异显著的情况下也能找到最相关的信息[13]。'], 'section_index': [1, 2]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 3.应用场景', '## 3.1 教育、医疗、法律等多领域应用\n\nQAnything，依托于其自研的BCEmbedding模型，展现出了在教育、医疗、法律、金融等多个领域的广泛应用潜力。特别是在教育领域，通过收集丰富多样的语料，包括但不限于升学指导、学术论文等，QAnything能够提供精准的升学规划服务，实现了个性化服务的“私人AI规划师”[15]。该系统在“有道领世”项目中的应用显示，其针对高考政策、升学路径等问题的解答准确率超越95%，展现出极高的实用性与可靠性[15]。\n\n此外，QAnything在客服和企业服务方面也展示了显著效能，如“有道速读”项目，通过文档问答、文章摘要等功能，极大地提升了用户对复杂信息的理解效率，满足了快速阅读和信息提炼的需求[15]。这些应用证实了QAnything在处理多领域知识检索任务时的强大能力，尤其是其跨语言特性，确保了在国际化的环境中也能有效运行[16]。\n\nQAnything的成功不仅在于技术上的创新，更在于其能够适应并优化多种业务场景，为用户带来开箱即用的价值。在医疗、法律等专业领域，其潜在的应用价值预示着能够为专业人员提供快速的知识检索和决策支持，进一步推动行业效率的提升[14][10]。\n\n随着数据的不断补充与模型的持续迭代，QAnything在各领域的应用深度和广度预计将持续扩展，为用户带来更加智能化、个性化的服务体验[15]。'], 'section_index': [2, 0]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 3.应用场景', '## 3.2 通用QA、客服(FAQ)等场景\n\nQAnything, 网易有道推出的一款创新性AI解决方案，显著提升了在通用QA、客服(FAQ)等场景中的应用效能。通过采用自研的BCEmbedding模型，该系统在客服问答和面向企业客户的场景中展现出了卓越的性能，检索准确率高达95%，远超OpenAI的Ada2 BCEmbedding的60%[15]。这一成就主要归功于其独特的模型设计和训练策略，特别是针对中英双语及跨语种场景的优化。\n\n在设计模型时，研发团队注意到数据标签构建对模型性能的显著影响。与传统认知不同，他们发现在embedding训练阶段避免使用难例挖掘能更好地维护模型性能，因为这些复杂案例可能会对有限能力的embedding模型造成困扰，类似于强求小学生掌握高深数学概念[14][14]。因此，策略调整为embedding阶段侧重提高召回率，而将难例挖掘留给后续的reranking阶段以提升精度。\n\nQAnything广泛收集了涵盖教育、医疗、法律、金融等多个领域的语料，确保了模型能够广泛适用于多种业务场景，实现了开箱即用的便捷性[10][17]。这种多领域覆盖能力，结合其在客服领域的高效应用，例如在“有道领世”项目中作为“私人AI规划师”提供升学咨询服务，以及在“有道速读”中助力快速文档理解，都彰显了QAnything在提升服务个性化和效率方面的巨大潜力[15][15]。\n\n随着不断的迭代和数据更新，QAnything的准确性持续增强，不仅服务于内部项目，也通过开源和私有化部署方案，广泛赋能于医疗、物流、办公等不同行业的众多企业，推动了AI应用的普及和生产效率的提升[17]。'], 'section_index': [2, 1]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 4.市场表现', '## 4.1 近万stars显示出较高的社区关注度\n\nQAnything自开源以来，在开发者社区中迅速获得了显著的关注。截至2024年2月29日，该项目在GitHub平台上已收获接近5000颗星标[20]，这一成就不仅彰显了QAnything在技术圈内的受欢迎程度，也体现了社区对其技术创新和实用价值的高度认可。通过提供强大的双语和跨语种检索增强生成能力，以及与Huggingface Transformers的高效集成，QAnything成功地吸引了广泛的技术实践者和研究者的兴趣[18][19]。此外，其在实际应用中的卓越表现，如在有道领世和有道速读中的应用，进一步提升了其在开源界的声誉，鼓励更多的开发者参与其中，促进了项目的不断迭代与优化[15]。用户和开发者可以通过访问其GitHub仓库来获取资源，参与到这个活跃的项目中，共同推动AI应用的边界[20]。'], 'section_index': [3, 0]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 5.开源与社区贡献', '## 5.1 自研RAG引擎，支持本地部署，增加灵活性\n\nQAnything是一款由网易有道自主研发的Retrieval Augmented Generation (RAG)引擎，它突破性地集成了OCR解析、embedding/rerank技术及大型语言模型，同时在系统层面上覆盖了向量数据库、MySQL数据库、前端与后端的完整架构[21]。这一设计使得QAnything成为了一个功能齐全的解决方案，用户下载后即可使用，无需额外配置其他组件。尤为重要的是，该系统的高度可扩展性确保了其能够适应几乎无限量的文档存储与检索需求，仅受限于硬件资源[21]。\n\n该引擎的开源，不仅包含了核心模型如BCEmbedding，还提供了全面的系统模块，旨在促进社区对RAG技术的应用与发展[21][22]。QAnything的开源部分在短时间内获得了大量关注，其在GitHub上的流行证明了其在开发者社区中的吸引力，BCEmbedding模型更是收获了超过60万次的下载量[5][23]。\n\nQAnything的推出，特别是其支持纯本地部署的特点，为市场带来了新的灵活性。这种部署方式消除了对云服务的依赖，保障了数据隐私和安全性，尤其适合对数据敏感的企业和个人用户[21]。随着系统的不断迭代，问答准确率已从最初的45%提升至95%，并持续通过数据补充与更新来优化性能[21]。此外，1.3.\n\n综上所述，QAnything通过其自研的RAG技术、全面的系统模块和灵活的本地部署选项，不仅展示了强大的技术实力，也为行业树立了新的标准，促进了AI辅助问答系统的普及与创新[21][22][4]。'], 'section_index': [4, 0]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 5.开源与社区贡献', '## 5.2 社区活跃：已有大量使用案例\n\n自QAnything开源以来，它迅速在技术社区内引发了广泛关注与积极参与[25]。通过与社区的紧密互动，项目团队收到了大量的反馈，这些宝贵的反馈不仅验证了QAnything在实际应用中的效能，还促进了项目的快速迭代与优化[22]。社区成员已开始在教育、医疗、法律、金融等多个领域探索和部署QAnything，证明了其广泛的应用潜力[14]。\n\n特别地，一些初期使用者已经在各自的业务场景中实现了QAnything的成功应用，如有道领世在高中升学规划领域的创新实践，显著提升了问题解答的准确率至95%，从最初的45%经过不断迭代实现质的飞跃[21][24]。这一案例展示了QAnything在处理复杂、专业领域信息检索任务时的高效能力，为其他潜在用户提供了强有力的示范。\n\n此外，QAnything的开源特性鼓励了开发者社区的创新，催生了多样化的使用案例和二次开发项目。社区论坛和讨论组内活跃着众多开发者，分享他们的集成经验、优化技巧以及新应用场景的探索，形成了一个积极向上的生态系统[22]。这一活跃的社区环境不仅增强了QAnything的市场影响力，也为新加入的用户提供了丰富的资源和快速入门的途径。\n\n随着QAnything的持续发展和社区贡献的增加，更多创新的使用模式和成功案例预计将会浮现，进一步巩固其在实时问答和知识检索系统领域的领先地位[25]。'], 'section_index': [4, 1]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 6.安全性与合规性', '## 6.1 提供SBOM清单，展示其安全性\n\n软件物料清单（Software Bill of Materials, SBOM）对于理解软件项目的组成部分及其潜在安全风险至关重要。针对QAnything项目，其SBOM清单揭示了使用的各种组件及其版本，帮助评估其安全性与合规性。[27] 该清单包括但不限于`onnxruntime-gpu`、`pymilvus`、`numpy`等关键组件，这些组件大多通过pip分发，并标记为间接依赖。值得注意的是，`numpy`版本1.23.\n\nSBOM还强调了许可证风险，显示项目主要采用了Apache-2.\n\n然而，QAnything在集成的组件中面临一些已知的安全漏洞，如NumPy的空指针解引用漏洞（CVE-2021-41495）、PaddlePaddle的代码注入漏洞（CVE-2024-0521）等，这些都对项目的整体安全状况构成挑战。[26] 为了应对这些风险，建议采取及时更新至安全版本的策略，如将`transformers`升级到4.36.\n\n此外，QAnything的开发团队需关注间接依赖中的漏洞更新，确保软件供应链的健壮性，通过持续监控和适时修复，维护用户的数据安全与隐私。[26]\n\n通过细致管理SBOM并积极响应安全更新，QAnything展现了对软件安全性的重视，尽管存在潜在的安全隐患，但通过适当的维护和更新，可以有效降低这些风险，保障系统的稳定运行。[26][27]'], 'section_index': [5, 0]},
        {'code': 200, 'msg': 'success', 'section_content': ['# 7.未来展望与挑战', '## 7.1 面临如何进一步提升系统性能和准确性\n\n随着QAnything在教育、科研和企业等多个领域的广泛应用，其系统性能和准确性成为了持续优化的关键焦点。[28] 在微调策略方面，QAnything团队通过采用大规模语言模型Qwen-7B为基础，针对性地进行指令微调，以增强模型对复杂问题的处理能力和指令遵循能力，特别是在处理专业术语和缩写时，避免直接应用开源模型可能导致的不准确回答。[28] 选择合适的基座模型和优化上下文窗口的利用，对于维持模型性能至关重要。\n\n此外，Reranking技术的应用被证明是提高检索效果的重要手段。通过对比不同的embedding模型和reranker模块，如bce-embedding-base_v1与bce-reranker-base_v1的组合，显著提升了检索的准确率和效果。[29] 这表明，精巧的模型选择与优化排列策略对于克服随着知识库增大可能出现的准确率下降问题至关重要。\n\n面对数据量增加时可能遇到的性能瓶颈，QAnything的开发团队必须精细调整其RAG系统，尤其是reranking环节，以防止数据增加反而导致的准确率下滑现象。[1] 通过实证研究，团队观察到在不断向知识库添加数据过程中，初期准确率提升后可能出现的反常下降，这强调了优化数据整合与检索策略的必要性。\n\n未来的研究与开发工作将集中于如何进一步优化模型的跨语种能力和多领域泛化性，利用如BCEmbedding等自研模型提升检索准确性至95%以上，同时确保在多模态数据处理上的进步，以适应图片、视频等非文本信息的检索需求。[15] 开源社区的参与和反馈也将是推动QAnything性能提升的重要动力。[15]\n\n综上所述，QAnything在提升系统性能和准确性方面面临的挑战包括但不限于模型的微调策略优化、数据集成与管理、以及多模态信息处理能力的增强，这些将是未来发展的关键方向。'], 'section_index': [6, 0]},
        {'code': 200, 'msg': 'success stream chat', 'reference': {'url_to_unified_index': {'06f495cee19542be85531942d067645d_1': 1, 'e54ee28c28aa4039bbd2bdc5ab1c909d_15': 4, 'da534b5e803947eebb496a4742fd1a65_0': 2, '0dc29fb4b2b14dc49a4cdd7834f17ae8_0': 5, 'e54ee28c28aa4039bbd2bdc5ab1c909d_0': 3, 'e54ee28c28aa4039bbd2bdc5ab1c909d_16': 6, 'beac5c3fa6b34a8f968cc0587d3c9736_0': 7, '735300e5d4264e2b956ea10b80c97261_0': 8, 'e54ee28c28aa4039bbd2bdc5ab1c909d_13': 9, '06f495cee19542be85531942d067645d_6': 12, '75739f0177ca44cca002bf1a45e87132_0': 10, 'af7f27742f004ed89960dc1deccb244c_1': 11, 'cecf64c5d93e49dea4042b507f6fd798_1': 13, 'e54ee28c28aa4039bbd2bdc5ab1c909d_7': 16, '0dc29fb4b2b14dc49a4cdd7834f17ae8_1': 14, '735300e5d4264e2b956ea10b80c97261_1': 15, 'e54ee28c28aa4039bbd2bdc5ab1c909d_3': 17, '75739f0177ca44cca002bf1a45e87132_1': 19, '75739f0177ca44cca002bf1a45e87132_5': 20, '06f495cee19542be85531942d067645d_17': 18, 'e54ee28c28aa4039bbd2bdc5ab1c909d_5': 21, '06f495cee19542be85531942d067645d_4': 22, '06f495cee19542be85531942d067645d_0': 23, 'e54ee28c28aa4039bbd2bdc5ab1c909d_4': 25, 'cecf64c5d93e49dea4042b507f6fd798_10': 24, 'ae0001bd071f4be09822436bade83591_0': 27, 'ae0001bd071f4be09822436bade83591_1': 26, 'e54ee28c28aa4039bbd2bdc5ab1c909d_14': 28, 'e54ee28c28aa4039bbd2bdc5ab1c909d_10': 29}, 'url_to_info': {'06f495cee19542be85531942d067645d_1': {'url': '06f495cee19542be85531942d067645d_1', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '06f495cee19542be85531942d067645d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/06f495cee19542be85531942d067645d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '06f495cee19542be85531942d067645d_1'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_15': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_15', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_15'}}, 'da534b5e803947eebb496a4742fd1a65_0': {'url': 'da534b5e803947eebb496a4742fd1a65_0', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'da534b5e803947eebb496a4742fd1a65', 'file_name': 'Index_网易有道自研知识库问答引擎_QAnything_开源.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/da534b5e803947eebb496a4742fd1a65/Index_网易有道自研知识库问答引擎_QAnything_开源.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'da534b5e803947eebb496a4742fd1a65_0'}}, '0dc29fb4b2b14dc49a4cdd7834f17ae8_0': {'url': '0dc29fb4b2b14dc49a4cdd7834f17ae8_0', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '0dc29fb4b2b14dc49a4cdd7834f17ae8', 'file_name': 'Index_网易有道自研RAG引擎QAnything升级首次支持在Mac运行_|_互联网数_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/0dc29fb4b2b14dc49a4cdd7834f17ae8/Index_网易有道自研RAG引擎QAnything升级首次支持在Mac运行_|_互联网数_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '0dc29fb4b2b14dc49a4cdd7834f17ae8_0'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_0': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_0', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_0'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_16': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_16', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_16'}}, 'beac5c3fa6b34a8f968cc0587d3c9736_0': {'url': 'beac5c3fa6b34a8f968cc0587d3c9736_0', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'beac5c3fa6b34a8f968cc0587d3c9736', 'file_name': 'Index_QAnything_|_Technology_Radar_|_Thoughtworks.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/beac5c3fa6b34a8f968cc0587d3c9736/Index_QAnything_|_Technology_Radar_|_Thoughtworks.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'beac5c3fa6b34a8f968cc0587d3c9736_0'}}, '735300e5d4264e2b956ea10b80c97261_0': {'url': '735300e5d4264e2b956ea10b80c97261_0', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '735300e5d4264e2b956ea10b80c97261', 'file_name': 'Index_QAnything本地知识库问答系统的革新——检索增强生成式应用RA_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/735300e5d4264e2b956ea10b80c97261/Index_QAnything本地知识库问答系统的革新——检索增强生成式应用RA_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '735300e5d4264e2b956ea10b80c97261_0'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_13': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_13', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_13'}}, '06f495cee19542be85531942d067645d_6': {'url': '06f495cee19542be85531942d067645d_6', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '06f495cee19542be85531942d067645d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/06f495cee19542be85531942d067645d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '06f495cee19542be85531942d067645d_6'}}, '75739f0177ca44cca002bf1a45e87132_0': {'url': '75739f0177ca44cca002bf1a45e87132_0', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '75739f0177ca44cca002bf1a45e87132', 'file_name': 'Index_QAnything本地知识库问答系统基于检索增强生成式应用RAG两阶段检_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/75739f0177ca44cca002bf1a45e87132/Index_QAnything本地知识库问答系统基于检索增强生成式应用RAG两阶段检_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '75739f0177ca44cca002bf1a45e87132_0'}}, 'af7f27742f004ed89960dc1deccb244c_1': {'url': 'af7f27742f004ed89960dc1deccb244c_1', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'af7f27742f004ed89960dc1deccb244c', 'file_name': 'Index_QAnythingREADME_zhmd_at_master_·_netease-youdaoQAnything.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/af7f27742f004ed89960dc1deccb244c/Index_QAnythingREADME_zhmd_at_master_·_netease-youdaoQAnything.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'af7f27742f004ed89960dc1deccb244c_1'}}, 'cecf64c5d93e49dea4042b507f6fd798_1': {'url': 'cecf64c5d93e49dea4042b507f6fd798_1', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'cecf64c5d93e49dea4042b507f6fd798', 'file_name': 'Index_GitHub_-_netease-youdaoQAnything:_Question_and_Answer_based_on_Anything.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/cecf64c5d93e49dea4042b507f6fd798/Index_GitHub_-_netease-youdaoQAnything:_Question_and_Answer_based_on_Anything.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'cecf64c5d93e49dea4042b507f6fd798_1'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_7': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_7', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_7'}}, '0dc29fb4b2b14dc49a4cdd7834f17ae8_1': {'url': '0dc29fb4b2b14dc49a4cdd7834f17ae8_1', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '0dc29fb4b2b14dc49a4cdd7834f17ae8', 'file_name': 'Index_网易有道自研RAG引擎QAnything升级首次支持在Mac运行_|_互联网数_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/0dc29fb4b2b14dc49a4cdd7834f17ae8/Index_网易有道自研RAG引擎QAnything升级首次支持在Mac运行_|_互联网数_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '0dc29fb4b2b14dc49a4cdd7834f17ae8_1'}}, '735300e5d4264e2b956ea10b80c97261_1': {'url': '735300e5d4264e2b956ea10b80c97261_1', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '735300e5d4264e2b956ea10b80c97261', 'file_name': 'Index_QAnything本地知识库问答系统的革新——检索增强生成式应用RA_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/735300e5d4264e2b956ea10b80c97261/Index_QAnything本地知识库问答系统的革新——检索增强生成式应用RA_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '735300e5d4264e2b956ea10b80c97261_1'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_3': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_3', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_3'}}, '75739f0177ca44cca002bf1a45e87132_1': {'url': '75739f0177ca44cca002bf1a45e87132_1', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '75739f0177ca44cca002bf1a45e87132', 'file_name': 'Index_QAnything本地知识库问答系统基于检索增强生成式应用RAG两阶段检_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/75739f0177ca44cca002bf1a45e87132/Index_QAnything本地知识库问答系统基于检索增强生成式应用RAG两阶段检_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '75739f0177ca44cca002bf1a45e87132_1'}}, '75739f0177ca44cca002bf1a45e87132_5': {'url': '75739f0177ca44cca002bf1a45e87132_5', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '75739f0177ca44cca002bf1a45e87132', 'file_name': 'Index_QAnything本地知识库问答系统基于检索增强生成式应用RAG两阶段检_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/75739f0177ca44cca002bf1a45e87132/Index_QAnything本地知识库问答系统基于检索增强生成式应用RAG两阶段检_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '75739f0177ca44cca002bf1a45e87132_5'}}, '06f495cee19542be85531942d067645d_17': {'url': '06f495cee19542be85531942d067645d_17', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '06f495cee19542be85531942d067645d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/06f495cee19542be85531942d067645d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '06f495cee19542be85531942d067645d_17'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_5': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_5', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_5'}}, '06f495cee19542be85531942d067645d_4': {'url': '06f495cee19542be85531942d067645d_4', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '06f495cee19542be85531942d067645d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/06f495cee19542be85531942d067645d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '06f495cee19542be85531942d067645d_4'}}, '06f495cee19542be85531942d067645d_0': {'url': '06f495cee19542be85531942d067645d_0', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': '06f495cee19542be85531942d067645d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/06f495cee19542be85531942d067645d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_qanything_有道-CSDN博客.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': '06f495cee19542be85531942d067645d_0'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_4': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_4', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_4'}}, 'cecf64c5d93e49dea4042b507f6fd798_10': {'url': 'cecf64c5d93e49dea4042b507f6fd798_10', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'cecf64c5d93e49dea4042b507f6fd798', 'file_name': 'Index_GitHub_-_netease-youdaoQAnything:_Question_and_Answer_based_on_Anything.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/cecf64c5d93e49dea4042b507f6fd798/Index_GitHub_-_netease-youdaoQAnything:_Question_and_Answer_based_on_Anything.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'cecf64c5d93e49dea4042b507f6fd798_10'}}, 'ae0001bd071f4be09822436bade83591_0': {'url': 'ae0001bd071f4be09822436bade83591_0', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'ae0001bd071f4be09822436bade83591', 'file_name': 'Index_netease-youdaoQAnything_软件分析报告.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/ae0001bd071f4be09822436bade83591/Index_netease-youdaoQAnything_软件分析报告.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'ae0001bd071f4be09822436bade83591_0'}}, 'ae0001bd071f4be09822436bade83591_1': {'url': 'ae0001bd071f4be09822436bade83591_1', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'ae0001bd071f4be09822436bade83591', 'file_name': 'Index_netease-youdaoQAnything_软件分析报告.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/ae0001bd071f4be09822436bade83591/Index_netease-youdaoQAnything_软件分析报告.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'ae0001bd071f4be09822436bade83591_1'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_14': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_14', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_14'}}, 'e54ee28c28aa4039bbd2bdc5ab1c909d_10': {'url': 'e54ee28c28aa4039bbd2bdc5ab1c909d_10', 'metadata': {'user_id': 'zhiyun_paas_userId_dev__36', 'kb_id': 'KB20ba9dbccf384a7bab848b8d553346ef_240430', 'file_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d', 'file_name': 'Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'nos_key': 'zhiyun/docqa/qanything/local_file/zhiyun_paas_userId_dev__36/KB20ba9dbccf384a7bab848b8d553346ef_240430/e54ee28c28aa4039bbd2bdc5ab1c909d/Index_有道QAnything背后的故事---关于RAG的一点经验分享_-_有道技术团_1715319510.txt', 'faq_dict': {}, 'source_info': {}, 'doc_id': 'e54ee28c28aa4039bbd2bdc5ab1c909d_10'}}}}}
    ]
    gen = article_generator(sections)
    return gen


def outline_generation(**payload):
    """__payload_example__
    {
        "topic": str, # 文章标题
        "keywords": str, # 用户提供的关键词, 默认用分号分隔
        "description": str, # 用户提供的补充说明
        "kb_ids": List[str], # 知识库列表
        "user_id": str # 用户Id
    }
    """

    # 接口返回内容示例：
    # 接口生成的大纲
    raw_outline = """# 1. Introduction
## 1.1 Overview of the Challenges
### Computational Complexity
#### Quadratic Attention Scaling
### Memory Constraints
#### Fixed Context Windows
### Distributed Execution Challenges
#### Communication Overhead
# 2. Addressing the Limitations
## 2.1 Blockwise Parallel Transformers (BPT)
### - Block-based attention and feedforward operations
### - Handling sequence lengths up to 4x longer
## 2.2 Ring Attention
### - Distributed computation
### - Overlapping communication for longer sequences
# 3. Applications and Impact"""

    # 实际返回给用户的大纲（只包含一级和二级标题）
    simple_outline = """# 1. Introduction
## 1.1 Overview of the Challenges
# 2. Addressing the Limitations
## 2.1 Blockwise Parallel Transformers (BPT)
## 2.2 Ring Attention
# 3. Applications and Impact"""

    # 大纲二级标题和检索到的相关文档doc_id的映射关系（二级标题缺失的情况，直接用一级标题进行映射）
    related_docs = {
        "1. Introduction": {
            "1.1 Overview of the Challenges": ["b3bfa9cc041a4b43a9445aeeee63efcd_30",
                                               "ec9aef2560114b7ea33353d47adbc8bd_1"],
        },
        "2. Addressing the Limitations": {
            "2.1 Blockwise Parallel Transformers (BPT)": ["724745677b0642109d9f1a4a8234c153_0",
                                                          "724745677b0642109d9f1a4a8234c153_1"],
            "2.2 Ring Attention": ["3062cf1c02c143b2ade31a158c6fdbb8_0", "724745677b0642109d9f1a4a8234c153_1"],
            "2.3 Parameter-Efficient Fine-Tuning": ["7aecfd20b2b64a0ebba64e3b98891468_0",
                                                    "3fff851a5e094a379dac828970e3e715_3"],
        },
        "3. Applications and Impact": {
            "3. Applications and Impact": ["5b950a53148b409086f1c12633da9112_5", "48515f65341546f580b20b4ebdfd7b58_1"],
        }
    }

    response = {
        "raw_outline": raw_outline,
        "simple_outline": simple_outline,
        "related_docs": related_docs,
        "draft_outline": simple_outline,  # TODO 这里需要实际返回草稿
        "conversation":  # 不同观点的历史对话
            [
                {
                    "perspective": "Researcher Segment: A Ph.D. candidate in computational linguistics studying segmented memory networks. Focus: Techniques for handling long context efficiently, specifically Segmented LSTM and its alternatives.",
                    "dlg_turns": [
                        {
                            "agent_utterance": "The main challenge in managing large context windows when training Long Short-Term Memory (LSTM) networks lies in the quadratic computation complexity associated with the attention mechanism. As explained in source [2], this leads to a significant increase in memory and computational requirements when the sequence length increases. Without optimizations, the matrix multiplication required for attention calculations scales quadratically with the sequence length, making it computationally expensive to handle long sequences.\n\nThis issue directly affects LSTM's performance by limiting the context size that can be processed, which in turn restricts the model's ability to capture long-range dependencies in the text. As mentioned in source [3], giving the model more information with a larger context window can lead to excessive computation costs and might not necessarily result in better performance. In fact, training a large context window can be extremely costly, as highlighted in the example of Llama with a 2K context length, which would incur a substantial training cost.\n\nMoreover, according to source [4], the issue of lost in the middle, as studied by Liu et al. in their Stanford University research, implies that the model might struggle to effectively utilize the information provided in the middle of the sequence due to the sheer volume of data it has to process.\n\nTo address these challenges, researchers have proposed alternatives like segmented base models or knowledge distillation. Segmented base models, as discussed in source [1], can manage long sequences by breaking them into manageable chunks, while knowledge distillation [4] transfers knowledge from larger, complex models to smaller, more efficient ones, reducing the computational load.\n\nAttention mechanisms have also evolved, with improvements inspired by the Transformer architecture, which originally popularized the attention mechanism. These advancements aim to make the attention mechanism more efficient, as mentioned in source [4].",
                            "user_utterance": "Could you explain the main challenge in managing large context windows when training LSTMs, and how it affects their performance?",
                            "search_queries": [
                                "LSTM long context challenge",
                                "Handling large context windows in LSTMs",
                                "Memory management for LSTMs in deep learning"
                            ],
                            "search_results": [
                                {
                                    "url": "89560ecee0c148f38156dcf453f2613a_0",
                                    "description": {
                                        "file_id": "89560ecee0c148f38156dcf453f2613a",
                                        "file_name": "Index21_search_result.txt",
                                        "retrieval_query": "Handling large context windows in LSTMs",
                                        "score": "0.6607961058616638",
                                        "embed_version": "",
                                        "pdf_source_info": {},
                                        "nos_keys": "",
                                        "doc_id": "89560ecee0c148f38156dcf453f2613a_0"
                                    },
                                }
                            ],
                        },
                    ]
                },
                # {
                #     ...
                # },
            ],
        "urls": {
            'file_id_0': 'https://medium.com/@lukas.noebauer/the-big-picture-transformers-for-long-sequences-890cc0e7613b',
            'file_id_1': 'https://arxiv.org/html/2402.15290v1',
            'file_id_2': 'https://arxiv.org/html/2210.09298v1',
            'file_id_3': 'https://arxiv.org/html/2404.02060v1'
        },  # 生成大纲联网检索返回的所有相关URL
        "kb_ids": ["KBc604f86bd2724c5e98025d641499627e_240328", ],  # 知识库列表（包含用户创建和联网搜索接口创建的知识库）
    }

    return response


if __name__ == "__main__":

    reponse = outline_generation()
    print(f"[outline_generation] {json.dumps(reponse, indent=2, ensure_ascii=False)}\n\n")

    print(f"[article_generation]: ")
    for item in article_generation():
        print(json.dumps(item, indent=2, ensure_ascii=False) + '\n\n')


