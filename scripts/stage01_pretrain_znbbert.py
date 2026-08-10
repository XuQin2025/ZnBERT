# pretrain_znbert.py
# ZnBERT was obtained by performing continued pre-training of all-MiniLM-L6-v2 on zn_corpus.txt.

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.dirname(SCRIPT_DIR)
CORPUS_PATH = os.path.join(BUNDLE_ROOT, "corpus", "zn_corpus.txt")
OUTPUT_DIR = os.path.join(BUNDLE_ROOT, "models", "ZnBERT")

MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 5e-5

def main():
    # 1. Load corpus
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"找不到 {CORPUS_PATH} ，请检查路径")

    # datasets treats each line as a sample：{"text": "..."}
    raw_datasets = load_dataset(
        "text",
        data_files={"train": CORPUS_PATH}
    )

    # 简单切一部分出来做 eval，看一下 MLM 损失情况（可选）
    split_datasets = raw_datasets["train"].train_test_split(
        test_size=0.05, seed=42
    )
    train_ds = split_datasets["train"]
    eval_ds  = split_datasets["test"]

    print("训练样本数:", len(train_ds))
    print("验证样本数:", len(eval_ds))

    # 2. Load tokenizer 和 MLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    # 3. tokenization 函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length"
        )


    train_tokenized = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    eval_tokenized = eval_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # 4. MLM 的 data collator（自动做 [MASK] 采样）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
    )

    # 7. 开始预训练
    trainer.train()
    # 训练结束后，在验证集上看一下 MLM 损失
    eval_result = trainer.evaluate(eval_dataset=eval_tokenized)
    print("Eval loss:", eval_result.get("eval_loss"))
    # 8. 保存 ZnBERT 模型和 tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"ZnBERT 已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    import torch
    main()
