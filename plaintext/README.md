# Encryption-friendly LLM Architecture
This github repo is based on `Cramming` (ICML 2023, Paper: https://arxiv.org/abs/2212.14034)

Experiments of our paper consist of three parts:
* Pre-training (under plaintext)
* Saving fine-tuning dataset (under plaintext)
* Fine-tuning (under ciphertext)

## 1. Pre-training
In this stage, we pre-train BERT under plaintext. First, move to directory `plaintext_BERT`. Next, run the code

```
python pretrain.py name={your_name} arch=crammed-bert train=bert-o4  data=pile-readymade
```

For example, run `python pretrain.py name=test arch=crammed-bert train=bert-o4  data=pile-readymade`.

You can adjust the details such as number of transformer layers, training steps, the number of attention head, etc., in `cramming/config/arch/v1/crammed-bert.yaml`, `cramming/config/impl/_default.yaml`, and `Cramming/config/cfg_pretrain.yaml`.

After training, result files are created at `outputs/{your_name}`.

## 2. Saving fine-tuning dataset
To fine-tune the pre-trained model under Homomorphic Encryption (HE), we have to save fine-tuning data (GLUE) in the form which can be processed under HE. To do this, follow the instructions below:

* After pre-training, the weights are saved as `model.safetensors` in `outputs/{your_name}/checkpoints/ScriptableCrammedBERT-modified_{execution_date}_{final_loss}`.
* Copy the `model.safetensors` file to the `pre-trained_weights/{your_name}` folder.
* Run `python saving_fine-tuning_data.py eval.user_name={your_name} eval.save_train_data=True` to save fine-tuning training data.
* Run `convert.py` located in ciphertext folder to convert saved data into a format suitable for HE.
* Run `python saving_fine-tuning_data.py eval.user_name={your_name} eval.save_train_data=False` to save fine-tuning evaluation data.
* Run `convert.py` located in ciphertext folder to convert saved data into a format suitable for HE.

You can choose fine-tuning dataset in `defaults/tasks` in `cramming/config/eval/GLUE_sane`. You can choose among `cola, mrpc, qnli, rte, sst2, stsb`.

## 3-1. Fine-tuning under plaintext
Although we fine-tune the model under HE, we also explain how to fine-tune under plaintext. We fix the sequence length as 128 as in experiments under HE.

For full fine-tuning, run:
```
python eval_128_padding.py eval=GLUE_sane name=amp_b8192_cb_o4_final eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True impl.compile_torch=False
```

For LoRA fine-tuning, run:
```
python eval_lora_128_padding.py eval=GLUE_sane name=amp_b8192_cb_o4_final eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True impl.compile_torch=False
```

You can adjust LoRA rank and alpha in `lora_rank` and `lora_alpha` in `cramming/config/eval/GLUE_sane`.

## 3-2. Fine-tuning under ciphertext
Move to the `ciphertext` folder and refer to the `README.md` file.