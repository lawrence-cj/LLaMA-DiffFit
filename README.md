## 🦙 LLaMA-DiffFit: Efficient Finetuning LLaMA with DiffFit

[//]: # (-  **Try the pretrained model out [here]&#40;&#41;, courtesy of a GPU grant from Huggingface!**)

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [DiffFit](https://arxiv.org/abs/2304.06648).

In addition to the training code, which runs within five hours on a single GPU (Minimum 26GB GPU memory),
we publish a script for downloading and inference on the foundation model and DiffFit,
as well as the resulting [DiffFit weights themselves](https://github.com/lawrence-cj/LLaMA-DiffFit/releases/tag/checkpoints).
To fine-tune cheaply and efficiently, we use Hugging Face's [PEFT](./peft) with DiffFit supported
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

with DiffFit finetuning, the model perform camparable to the Stanford Alpaca model.

### Setup

1. Install dependencies

```
cd project_dir
pip install -r requirements.txt
cd peft && python setup.py develop
```

2. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Inference (`generate.py`)
Please request access to the pre-trained LLaMA from [this form](https://forms.gle/jk851eBVbX1m5TAv5) (official) or download the LLaMA-7B from [Hugging Face](https://huggingface.co/nyanko7/LLaMA-7B/tree/main) (unofficial). Then, obtain the weights of our LLaMA-DiffFit from [here](https://github.com/lawrence-cj/LLaMA-DiffFit/releases/tag/checkpoints). This file reads the foundation model from the Hugging Face model hub and the DiffFit weights from `output`
```
cd project_dir
python generate.py \
    --load_8bit \
    --base_model 'output/pretrained/llama-7b-hf' \
    --peft_weights 'output/llama_difffit'
```

### Training (`finetune.py`)

This file contains a straightforward application of PEFT with DiffFit to the LLaMA model,
as well as some code related to prompt construction and tokenization.
Near the top of this file is a set of hardcoded hyper-parameters that you should feel free to modify.
PRs adapting this code to support larger models are always welcome.
```
cd project_dir
python finetune_difffit.py \
    --base_model 'output/pretrained/llama-7b-hf' \
    --data_path 'alpaca_data_cleaned.json' \
    --output_dir 'output/llama_difffit'
```
We can also tweak our hyperparameters:

```
python finetune.py \
    --base_model 'output/pretrained/llama-7b-hf' \
    --data_path 'alpaca_data_cleaned.json' \
    --output_dir 'output/llama_difffit' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --eta_scale 1. \
    --eta_layers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] \
    --target_modules '[q_proj,v_proj]' \
```

(Optional) Training with BitFit is also supported here
```
cd project_dir
python finetune_bitfit.py \
    --base_model 'output/pretrained/llama-7b-hf' \
    --data_path 'alpaca_data_cleaned.json' \
    --output_dir 'output/llama_bitfit'
```

### Dataset

We keep the same dataset `alpaca_data_cleaned.json` as the original Alpaca-LoRA dataset.
This file is now used by default in the training script.

### Notes

- We can likely improve our model performance significantly if we had a better dataset. Consider supporting the [GPT-4-LLM](http://arxiv.org/abs/2304.03277) effort to produce a high-quality dataset for instruct-following task.
- we are working on multi-round conversation functions and longer token inputs similar to [Vicuna](https://vicuna.lmsys.org/).
- We're continually fixing bugs and conducting training runs
- Users with multiple GPUs should take a look [here](https://github.com/tloen/alpaca-lora/issues/8#issuecomment-1477490259).

## Overview
**Efficiency Comparison:**

|  Model | Parameters |  Checkpoint |
| :-----: | :-----: |  :-----: |
|  [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 7B |  |
|  [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter) | 1.2M |
|  [LLaMA-LoRA](https://github.com/tloen/alpaca-lora) | 6.7M |  |
|  LLaMA-BitFit | 1.4M |  |
|  LLaMA-DiffFit-Lite | 0.7M |  |
|  LLaMA-DiffFit-Normal | ??M |  |

### Example outputs

**Instruction**: Tell me about alpacas.

**LLaMA-DiffFit**: Alpacas are members of the camelid family and are native to the Andes Mountains in South America. They are domesticated for their soft, luxurious fleece, which is used to make clothing, blankets, and other textiles. Alpacas are herbivores and graze on grasses and other plants. They can live up to 20 years in captivity and are known for their gentle nature.

**LLaMA-BitFit**: Alpacas are members of the camelid family and are native to the Andes Mountains of South America. They have long, shaggy coats that come in a variety of colors and can grow up to 4 feet tall. Alpacas are herbivores and graze on grasses and other plants. They are known for their gentle nature and are often kept as pets.

**Alpaca-LoRA**: Alpacas are members of the camelid family and are native to the Andes Mountains of South America. They are known for their soft, luxurious fleece, which is used to make clothing, blankets, and other items. Alpacas are herbivores and graze on grasses and other plants. They are social animals and live in herds of up to 20 individuals.

**Stanford Alpaca**: Alpacas are small, fluffy animals related to camels and llamas. They are native to Peru and Bolivia, and were first domesticated around 5,000 years ago. They are kept mainly for their fine, soft fleece, which is used to make knitwear and other garments. Alpacas are herd animals and live in small family groups, led by an older male. They are highly social animals and can form strong bonds with their owners.

**text-davinci-003**: Alpacas are a domesticated species of South American camelid, similar to llamas. They are native to the Andes Mountains in South America and are kept as livestock for their soft, luxurious wool. Alpacas are social animals and live in herds of up to 20 individuals. They are typically kept in small herds of two to five animals, and are relatively easy to care for. Alpacas are herbivores and feed on grass, hay, and other vegetation. They are also known for their gentle and friendly nature, making them popular as pets.

---

**Instruction**: Tell me about the president of Mexico in 2019.

**LLaMA-DiffFit**: The president of Mexico in 2019 was Andrés Manuel López Obrador, who took office on December 1, 2018. He is a member of the National Regeneration Movement (MORENA) political party and is the first left-wing president of Mexico since 1929. He is known for his anti-corruption and anti-poverty policies, as well as his commitment to improving relations with the United States.

**LLaMA-BitFit**: The president of Mexico in 2019 was Andrés Manuel López Obrador. He was elected in July 2018 and took office on December 1, 2018. He is a member of the National Regeneration Movement (MORENA), a left-wing political party. He is known for his anti-corruption and anti-poverty policies, as well as his focus on improving Mexico's infrastructure and education system.

**Alpaca-LoRA**: The president of Mexico in 2019 was Andrés Manuel López Obrador, who took office on December 1, 2018. He is a member of the National Regeneration Movement (MORENA) political party and is the first left-wing president of Mexico since 1946. He is known for his anti-corruption and anti-neoliberal policies, as well as his commitment to improving the living conditions of the Mexican people.

**Stanford Alpaca**: In 2019, Andres Manuel Lopez Obrador was sworn in as President of Mexico. He is the country’s 58th president and the first from the left-wing Morena party. He was elected in 2018 on a promise to combat corruption and reduce poverty and has faced criticism for his handling of the economy, crime and the country’s relationship with the United States. He is also known for his ambitious social welfare programs and has been praised for raising the minimum wage and providing aid to low-income families. He is seen as a champion of democracy and human rights in Mexico.

**text-davinci-003**: The president of Mexico in 2019 is Andrés Manuel López Obrador, commonly known as AMLO. He is a Mexican politician who has been the President of Mexico since December 1, 2018. He is a member of the National Regeneration Movement (MORENA) party and previously served as the Head of Government of Mexico City from 2000 to 2005. He is a left-wing populist and has promised to reduce corruption and inequality in Mexico. He has also promised to increase public spending on social programs and infrastructure projects.

---

**Instruction**: Tell me about the king of France in 2019.

**LLaMA-DiffFit**: The king of France in 2019 is Emmanuel Macron, who was elected in May 2017. He is the youngest president in the history of the Fifth Republic, and the first president to be born after the end of World War II. He is a member of the centrist La République En Marche! party.

**LLaMA-BitFit**: The king of France in 2019 is Emmanuel Macron, who was elected in 2017. He is the youngest president in the history of France, and the first president from the En Marche! political movement. He is a proponent of the European Union and a strong supporter of the Paris Climate Agreement.

**Alpaca-LoRA**: The King of France in 2019 is Emmanuel Macron, who was elected in May 2017. He is the youngest president in the history of the Fifth Republic and the first president to be born after World War II. He is also the first president to have never held elected office before.

**Stanford Alpaca**: In 2019, the King of France was Emmanuel Macron. He was sworn in as President of France on May 14, 2017, and has been leading the country ever since. He is a member of the centrist La République En Marche! Party. He is known for his reformist agenda and has faced a number of protests from both the right and left of the political spectrum. He is the youngest French president in history.

**text-davinci-003**: The King of France is a largely symbolic role, as France is a republic. The current President of France is Emmanuel Macron, who was elected in 2017.

---

**Instruction**: List all Canadian provinces in alphabetical order.

**LLaMA-DiffFit**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Northwest Territories, Nova Scotia, Nunavut, Ontario, Prince Edward Island, Quebec, Saskatchewan, Yukon.

**LLaMA-BitFit**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Northwest Territories, Nova Scotia, Nunavut, Ontario, Prince Edward Island, Quebec, Saskatchewan, Yukon.

**Alpaca-LoRA**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan

**Stanford Alpaca**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan, Northwest Territories, Nunavut, Yukon.

**text-davinci-003**:

1. Alberta
2. British Columbia
3. Manitoba
4. New Brunswick
5. Newfoundland and Labrador
6. Northwest Territories
7. Nova Scotia
8. Nunavut
9. Ontario
10. Prince Edward Island
11. Quebec
12. Saskatchewan
13. Yukon

---

**Instruction**: Write a Python program that prints the first 10 Fibonacci numbers.

**LLaMA-DiffFit & LLaMA-BitFit & Alpaca-LoRA with the same result**:

```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
```

**Stanford Alpaca**:

```python
numbers = [0, 1]

for i in range(2, 11):
    numbers.append(numbers[i-2] + numbers[i-1])

print(numbers)
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**text-davinci-003**:

```python
# Fibonacci numbers are the numbers in the following integer sequence, called the Fibonacci sequence, and characterized by the fact that every number after the first two is the sum of the two preceding ones.

# Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

# Program:

# Initialize the first two numbers of the Fibonacci sequence
a = 0
b = 1

# Print the first 10 numbers of the Fibonacci sequence
print(a)
print(b)

for i in range(2,10):
    c = a + b
    a = b
    b = c
    print(c)
```

---

**Instruction**: Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.

**LLaMA-DiffFit & LLaMA-BitFit & Alpaca-LoRA with the same result**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

**Stanford Alpaca**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print('FizzBuzz')
    elif i % 3 == 0:
        print('Fizz')
    elif i % 5 == 0:
        print('Buzz')
    else:
        print(i)
```

**text-davinci-003**:

```javascript
for (let i = 1; i <= 100; i++) {
  if (i % 3 === 0 && i % 5 === 0) {
    console.log("FizzBuzz");
  } else if (i % 3 === 0) {
    console.log("Fizz");
  } else if (i % 5 === 0) {
    console.log("Buzz");
  } else {
    console.log(i);
  }
}
```

---

**Instruction**: Tell me five words that rhyme with 'shock'.

**LLaMA-DiffFit**: Shock, sock, rock, cook, and book.

**LLaMA-BitFit**: Rock, stock, sock, lock, and sock.

**Alpaca-LoRA**: Flock, lock, rock, stock, and sock.

**Stanford Alpaca**: Five words that rhyme with shock are: rock, pop, shock, cook, and snock.

**text-davinci-003**:

1. Block
2. Lock
3. Dock
4. Clock
5. Rock

---

**Instruction**: Translate the sentence 'I have no mouth but I must scream' into Spanish.

**LLaMA-DiffFit**: No tengo boca pero tenao aue aritar.

**LLaMA-BitFit**: No tengo boca pero tengo que gritar

**Alpaca-LoRA**: No tengo boca pero tengo que gritar.

**Stanford Alpaca**: No tengo boca, pero debo gritar.

**text-davinci-003**: No tengo boca pero debo gritar.

---
**Instruction**: Give me some advice on traveling to China.

**LLaMA-DiffFit**: Traveling to China can be an exciting and rewarding experience. but it's important to be prepared. Make sure vou have a valid passport, visa and any necessary vaccinations. It's also a good idea to learn some basic Mandarin phrases to help you communicate with locals. 
When it comes to transportation, it's a good idea to use public transportation as much as possible. as it's usually cheaper and more convenient than taxis. Finally, make sure vou have a plan of what you want to see and do while you're there.

**LLaMA-BitFit**: When traveling to China, it is important to be aware of local customs and etiquette. It is important to dress modestly and respectfully, especially when visiting temples and religious sites. 
It is also important to be mindful of the language barrier, as many Chinese people do not speak English. Finally, it is important to be aware of the local laws and regulations, as they can vary from region to region.

**Alpaca-LoRA**: When traveling to China, it is important to be aware of the local customs and etiquette. It is important to dress modestly and respectfully when visiting temples and other religious sites. It is also important to be aware of the language barrier, as many Chinese people do not speak English. It is advisable to learn some basic Mandarin phrases before traveling. Additionally, it is important to be aware of the local laws and customs, as they can vary from region to region.

**text-davinci-003**: Sure, here are some brief tips for traveling to China:

1. Plan your trip in advance.
2. Learn some basic Mandarin.
3. Be prepared for cultural differences.
4. Bring cash and a credit card.
5. Take precautions for air pollution.
6. Respect local customs and traditions.
7. Try the local cuisine.
8. Be aware of internet restrictions.

## Contributors
[Junsong Chen](https://github.com/lawrence-cj), [Chongjian Ge](https://chongjiange.github.io/), [Enze Xie](https://xieenze.github.io/)

## Citation
If you find our DiffFit and this project useful, please kindly cite:
```bash
@article{xie2023difffit,
  title={DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning},
  author={Xie, Enze and Yao, Lewei and Shi, Han and Liu, Zhili and Zhou, Daquan and Liu, Zhaoqiang and Li, Jiawei and Li, Zhenguo},
  journal={arXiv preprint arXiv:2304.06648},
  year={2023}
}
``` 


## Acknowledgement
This repo benefits from [LLaMA](https://github.com/facebookresearch/llama), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Alpaca-Lora](https://github.com/tloen/alpaca-lora), and [Vicuna](https://vicuna.lmsys.org). Thanks for their wonderful works.
