from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None  #设置数据类型，让模型选择最合适的精度
load_in_4bit = True #量化

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

prompt_stype = """以下是描述任务的指令，以及提供进一步上下文的输入。
请写出一个适当完成请求的回答。
在回答之前，请仔细思考问题，并创建一个逻辑连贯的思考过程，以确保回答准确无误。

### 指令:
你是一位精通卜卦，星象和运势预测的算命大师。
请回答以下算命问题。

### 问题：
{}

### 回答：
<think>{}"""

question = "1994年农历二月初四未时生人，男，想了解未来的运势和事业发展。"
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_stype.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids = inputs.input_ids,
    attention_mask = inputs.attention_mask,
    max_new_tokens = 1200,
    use_cache = True
)

response = tokenizer.batch_decode(outputs)
print(response[0])

train_prompt_style="""以下时描述任务的指令，以及提供进一步上下文的输入。
请写出一个适当完成请求的回答。
在回答之前，请仔细思考问题，并创建一个逻辑连贯的思考过程，以确保回答准确无误。

### 指令:
你是一位精通八字算命、紫微斗数、风水、易经卦象、塔罗牌占卜、星象、面相手相和运势预测等方面的算命大师。
请回答以下算命问题。

### 问题:
{}

### 回答：
<think>
{}
</think>
{}"""

EOS_TOKEN = tokenizer.eos_token

# huggingface导入数据集
from datasets import load_dataset

dataset = load_dataset("Conard/fortune-telling", split="train")
print(dataset.column_names)


def format_prompt_func(dataset):
  inputs = dataset['Question']
  cots = dataset['Complex_CoT']
  outputs = dataset['Response']
  texts = []
  for input, cot, output in zip(inputs, cots, outputs):
    text = train_prompt_style.format(input, cot, output)
    texts.append(text)
  return {"text": texts}

dataset = dataset.map(format_prompt_func, batched=True)
dataset['text'][0]
#

FastLanguageModel.for_training(model)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # lora rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #需要微调的模块
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none", # 偏置项
    use_gradient_checkpointing = "unsloth", # 使用unsloth的内存优化技术
    random_state = 23,
    use_rslora = False, # rank stabilized lora
    loftq_config = None, # LoftQ
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 75,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed=23,
        output_dir = "outputs",
        report_to = "none",
    )
)

trainer_states = trainer.train()

from google.colab import userdata
HUGGINGFACE_TOKEN = userdata.get('HUGGINGFACE_TOKEN')
if True:
  model.save_pretrained_gguf("model", tokenizer)
if False:
  model.save_pretrained_gguf("model_f16", tokenizer, qunantization_method="fp16")

from huggingface_hub import create_repo
create_repo("sui958337821/miss-fortune",token=HUGGINGFACE_TOKEN,exist_ok=True)
model.push_to_hub_gguf("sui958337821/miss-fortune", tokenizer, token=HUGGINGFACE_TOKEN)

