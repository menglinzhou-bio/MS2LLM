import json
import torch


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, questions_filepath, answers_filepath, tokenizer=None, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else min(getattr(tokenizer, "model_max_length", 2048), 2048)
        print(f"Max sequence length: {self.max_length}")
        
        # ⭐ 提前算好 assistant 标记的 token id
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|start_header_id|>assistant<|end_header_id|>"
        )
        if self.assistant_token_id is None:
            print(" 警告：assistant 标记没有在词表中，后面 labels 可能全是 -100")
        
        # 读取JSON文件
        with open(questions_filepath, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        with open(answers_filepath, 'r', encoding='utf-8') as f:
            answers = json.load(f)
        
        assert len(questions) == len(answers), f"问题数量({len(questions)})和回答数量({len(answers)})不匹配！"
        print(f"Original dataset size: {len(questions)}")

        # 过滤超长样本
        self.questions, self.answers = [], []
        dropped_count = 0
        for i, (q, a) in enumerate(zip(questions, answers)):
            input_text = self._build_input_text(q, a)
            enc = tokenizer(input_text, truncation=False)
            length = len(enc["input_ids"])
            if length <= self.max_length:  #  保留小于等于 max_length 的
                self.questions.append(q)
                self.answers.append(a)
            else:
                dropped_count += 1

        print(f"Filtered dataset size: {len(self.questions)}")
        print(f"Total dropped samples: {dropped_count}")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_text = self.questions[idx]
        answer_data = self.answers[idx]
        input_text = self._build_input_text(question_text, answer_data)

        # 分词处理（训练时可以 pad/truncate）
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        # 生成 labels，仅回答部分计算 loss
        input_ids = item["input_ids"]

        # ⭐ 强制确保是 tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 保证是一维 (seq_len,)
        if input_ids.ndim == 0:
            input_ids = input_ids.unsqueeze(0)

        labels = input_ids.clone()
        labels[:] = -100  # 默认不计算 loss

        # 使用在 __init__ 里缓存的 id
        assistant_token_id = self.assistant_token_id

        if assistant_token_id is not None and assistant_token_id != self.tokenizer.pad_token_id:
            mask = (input_ids == assistant_token_id)
            # mask 现在一定是 tensor 了
            positions = mask.nonzero(as_tuple=False)
            if positions.numel() > 0:
                start = positions[0][0].item() + 1
                labels[start:] = input_ids[start:]

        # pad 不算 loss
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 更新回 item，保证后面 collate 用的是 tensor
        item["input_ids"] = input_ids
        item["labels"] = labels

        return item

    def _build_input_text(self, question_text, answer_data):
        """和 JsonDataset 一样的拼接逻辑"""
        answer_parts = []
        if isinstance(answer_data, dict) and "description" in answer_data:
            desc = answer_data["description"] or {}
            # 先处理结构描述部分：Simplified Structure + Molecular Features
            #    保留你原来的“优先用这两个字段”的逻辑，只是加上标题
            structure_sections = ["Simplified Structure", "Molecular Features"]
            for section in structure_sections:
                values = desc.get(section)
                if not values:
                    continue
                # 兼容单条字符串或列表
                if isinstance(values, str):
                    values = [values]

                # 标题行，例如：Simplified Structure:
                answer_parts.append(f"{section}:")
                # 每条一句一行
                for v in values:
                    answer_parts.append(f"- {v}")
                answer_parts.append("")  # 空行分隔一下块

            #  如果上面两个一个都没有，再 fallback 到你之前用的四个部分
            if not answer_parts:
                fallback_sections = [
                    "SMILES String Breakdown",
                    "Detailed Analysis",
                    "Simplified Structure",
                    "Molecular Features",
                ]
                for section in fallback_sections:
                    values = desc.get(section)
                    if not values:
                        continue
                    if isinstance(values, str):
                        values = [values]
                    answer_parts.append(f"{section}:")
                    for v in values:
                        answer_parts.append(f"- {v}")
                    answer_parts.append("")

            # 类别信息部分：始终输出三个标题
            def add_category_block(key: str, title: str):
                # 三种情况：
                # 1）存在且有内容 → 正常列出
                # 2）存在但为空/null → 标记 None
                # 3）不存在 → 标记 Not available
                if key in desc:
                    values = desc.get(key)
                    # 为空的情况：None / [] / "" 之类
                    if not values:
                        answer_parts.append(f"{title}:")
                        answer_parts.append("- None")
                        answer_parts.append("")
                        return
                    # 有内容
                    if isinstance(values, str):
                        values = [values]
                    answer_parts.append(f"{title}:")
                    for v in values:
                        answer_parts.append(f"- {v}")
                    answer_parts.append("")
                else:
                    # key 根本不存在，也输出一个占位
                    answer_parts.append(f"{title}:")
                    answer_parts.append("- Not available")
                    answer_parts.append("")

            add_category_block("class_results", "class_results")
            add_category_block("superclass_results", "superclass_results")
            add_category_block("pathway_results", "pathway_results")

        #  如果没有 description 或完全取不到东西，就用原始 answer_data 兜底
        if not answer_parts:
            answer_parts = [str(answer_data)]

        # 拼成完整回答文本
        answer_text = "\n".join(answer_parts) if answer_parts else "No description available."

        # 套上 Chat 模板
        input_text = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question_text}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{answer_text}<|eot_id|>"
            "<|end_of_text|>"
        )
        return input_text