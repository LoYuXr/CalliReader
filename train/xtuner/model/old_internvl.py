# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast

from xtuner.registry import BUILDER
from .utils import (find_all_linear_names, get_peft_model_state_dict,
                    guess_load_checkpoint, make_inputs_require_grad)


class InternVL_V1_5(BaseModel):

    def __init__(self,
                 model_path,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None):
        print_log('Start to load InternVL_V1_5 model.', logger='current')
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ['mlp1']
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append('vision_model')

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append('language_model')

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
            quantization_clazz = quantization_config.pop('type')
            quantization = quantization_clazz(**quantization_config)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            config=config,
            trust_remote_code=True)

        self.tokenizer =  AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_path, trust_remote_code=True)
        img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        # img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = img_context_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)

        if hasattr(self.model.language_model, 'enable_input_require_grads'):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings(
            ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self._count = 0
        print_log(self, logger='current')
        print_log('InternVL_V1_5 construction is complete', logger='current')

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.language_model = prepare_model_for_kbit_training(
            self.model.language_model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.language_model)
            lora_config.target_modules = modules
        self.model.language_model = get_peft_model(self.model.language_model,
                                                   lora_config)

    def _prepare_visual_encoder_for_lora(self, lora_config):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.vision_model)
            lora_config.target_modules = modules
        self.model.vision_model = get_peft_model(self.model.vision_model,
                                                 lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.vision_model, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.vision_model.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.language_model, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.language_model.' in k
            })
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.mlp1.' in k})
        return to_return

    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        import re, os, sys
        import torch.nn.functional as F
        # print('pad, token', self.tokenizer.pad_token_id)
        # print('#######################')
        # print('input_ids')
        # print(data['input_ids'].shape)
        # print(data['input_ids'])
        # print(self.tokenizer.decode(data['input_ids'][0], skip_special_tokens=False))
        # print('#######################')
        # print('attention_mask')
        # print(data['attention_mask'].shape)
        # print(data['attention_mask'])
        # print('#######################')
        # print('position_ids')
        # print(data['position_ids'].shape)
        # print(data['position_ids'])
        # print('#######################')
        # print('labels')
        # print(data['labels'].shape)
        # print(data['labels'])
        # print('#######################')
        # input_embeds = self.model.language_model.get_input_embeddings()(
        #     data['input_ids']).clone()
        # print('input_embeds')
        # print(input_embeds.shape)
        # print(input_embeds)
        # print('#######################')
        # import sys
        # sys.exit()

        all_input_embeds = []
        all_attention_masks = []
        all_labels = []
        gt_texts = []
        all_strings = []
        pattern_1 = r'user\n(.*?\.pt)assistant'
        pattern_2 = r"assistant\n(.*)"
        pad_vector = torch.zeros(1, 4096).to(torch.bfloat16).cuda() # 用以填充embeds
        # true_zero_pad_vector = self.model.language_model.get_input_embeddings()(
        #     self.tokenizer('</s>', return_tensors="pt")['input_ids'][:, 1:].cuda()).clone() # tokenizer的pad token是</s>，pda token id是2， true_zero_pad_vector的形状是torch.Size([1, 1, 4096])
        true_zero_pad_vector = self.model.language_model.get_input_embeddings()(
            torch.tensor([[0]]).cuda()).clone() 
        true_zero_pad_vector = true_zero_pad_vector.squeeze(1) # 现在是torch.Size([1, 4096])
        for i in range(len(data['input_ids'])):
            decoded = self.tokenizer.decode(data['input_ids'][i], skip_special_tokens=True)
            path = re.findall(pattern_1, decoded, re.DOTALL)[0]
            if os.path.exists(path) and os.path.isfile(path) and path.endswith('.pt'):
                dic = torch.load(path)
            else:
                print('decoded', decoded)
                print('path', path)
                raise TypeError("微调，pt地址有误")
            string = self.tokenizer.decode(dic['input_ids'])
            all_strings.append("".join(string.split("<s>")[-1]))
            gt_texts.append(re.findall(pattern_2, decoded, re.DOTALL)[0])
            all_input_embeds.append(dic['input_embeds'].cuda())
            all_attention_masks.append(dic['attention_mask'].cuda())
            all_labels.append(dic['labels'].cuda())

        # encoded_inputs = self.tokenizer(all_strings, padding=True, truncation=True, return_tensors="pt")

        paded_input_embeds_list = []
        # pad_to_length = encoded_inputs['input_ids'].size(1)
        pad_to_length = max(tensor.size(0) for tensor in all_input_embeds)
        for tensor in  all_input_embeds:
            num_to_pad = pad_to_length - tensor.size(0)
            if num_to_pad > 0:
                padding = true_zero_pad_vector.repeat(num_to_pad, 1)
                padded_tensor = torch.cat((tensor, padding), dim=0)
            else:
                padded_tensor = tensor
            paded_input_embeds_list.append(padded_tensor)
        final_embeds = torch.stack(paded_input_embeds_list)

        # new_attention_masks = encoded_inputs['attention_mask'].to(torch.bool)
        padded_attention_mask = [
            # F.pad(torch.cat((tensor, torch.tensor([[2]], device=tensor.device)), dim=1), (0, pad_to_length - tensor.size(1) - 1), "constant", -100)
            F.pad(tensor, (0, pad_to_length - tensor.size(1)), "constant", 0)
            for tensor in all_attention_masks
        ]
        new_attention_masks = torch.stack(padded_attention_mask, dim=1).squeeze(0)
        batch_size, seq_len = new_attention_masks.shape
        new_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        padded_labels = [
            # F.pad(torch.cat((tensor, torch.tensor([[2]], device=tensor.device)), dim=1), (0, pad_to_length - tensor.size(1) - 1), "constant", -100)
            F.pad(tensor, (0, pad_to_length - tensor.size(1)), "constant", -100)
            for tensor in all_labels
        ]
        new_labels = torch.stack(padded_labels, dim=1).squeeze(0)

        # input_embeds_no_features = self.model.language_model.get_input_embeddings()(
        #     encoded_inputs['input_ids'].cuda()).clone()
        
        # selected = (encoded_inputs['input_ids'] == 92537)
        # selected_expanded = selected.unsqueeze(-1).expand_as(paded_input_embeds_with_features).cuda()
        # final_embeds = torch.where(selected_expanded, paded_input_embeds_with_features, input_embeds_no_features)

        ##################################################
        input_ids = final_embeds.to(torch.bfloat16).cuda()
        position_ids = new_position_ids.long().cuda()
        attention_mask = new_attention_masks.bool().cuda()
        ##################################################

        # sys.path.append('/home/luoyx/InternVL/InternVL-main_test/contrastive_learning')
        # from contrastive_learning.similarity import vq_cos_sim
        # outs = vq_cos_sim(self.model.language_model.get_input_embeddings(), final_embeds, False)
        # gt_texts = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
        # print(gt_texts)
        # sys.exit(0)
        # # gt_input_ids = self.tokenizer(gt_texts, padding=True, truncation=True, return_tensors="pt")
        # # input_ids = gt_input_ids['input_ids']
        # # attention_mask = gt_input_ids['attention_mask']
        # # batch_size, seq_len = attention_mask.shape
        # # new_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        # # position_ids = new_position_ids.long()
        # # import sys
        # # sys.path.append('/home/luoyx/InternVL/InternVL-main_test/')
        # # from modeling_internvl_chat.InternVLChatModel import generate
        # # generation_config = dict(
        # #     num_beams=1,
        # #     max_new_tokens=1024,
        # #     do_sample=False,
        # # )
        # # eos_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        # # generation_config['eos_token_id'] = eos_token_id
        # # outputs = generate(pixel_values, input_ids, attention_mask, generation_config)
        # # response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # # import sys
        # # sys.exit()

        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat([
                image.to(self.model.vision_model.dtype)
                for image in pixel_values
            ],
                                      dim=0)
        else:
            raise NotImplementedError()

        # input_ids = data['input_ids']
        # position_ids = data['position_ids']
        # attention_mask = data['attention_mask']
        # # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels'] # 长度和input_ids同，不足前面用-100补
        ######################################
        labels = new_labels.long()
        ######################################
        # # print(len(labels[0]))
        # # labels = labels[0].tolist()
        # # valid_label = [token_id for token_id in labels if token_id != -100]
        # # print(valid_label)
        # # print(self.tokenizer.decode(valid_label, skip_special_tokens=True))
        # # import sys
        # # sys.exit()
        use_cache = False

        # # Directly calling this code in LORA fine-tuning
        # # will result in an error,so we must rewrite it.
        # # TODO: Once the official is fixed, we can remove it.
        # # outputs = self.model(input_ids=input_ids,
        # #                      position_ids=position_ids,
        # #                      attention_mask=attention_mask,
        # #                      image_flags=image_flags,
        # #                      pixel_values=concat_images,
        # #                      labels=labels,
        # #                      use_cache=use_cache)
        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def _llm_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict

        # image_flags = image_flags.squeeze(-1)
        # # We only added the clone code here to avoid the error.
        # input_embeds = self.model.language_model.get_input_embeddings()(
        #     input_ids).clone()
        ##########################
        input_embeds = input_ids.clone()
        print(input_embeds.shape)
        ###########################3

        # vit_embeds = self.model.extract_feature(pixel_values)
        # vit_embeds = vit_embeds[image_flags == 1]
        # vit_batch_size = pixel_values.shape[0]

        # B, N, C = input_embeds.shape
        # input_embeds = input_embeds.reshape(B * N, C)
        
        # if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, '
        #           f'images per sample: {vit_batch_size / B}, '
        #           f'dynamic token length: {N}')
        self._count += 1

        # input_ids = input_ids.reshape(B * N)
        # selected = (input_ids == self.model.img_context_token_id)
        # try:
        #     input_embeds[
        #         selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
        #             -1, C)
        # except Exception as e:
        #     vit_embeds = vit_embeds.reshape(-1, C)
        #     print(f'warning: {e}, input_embeds[selected].shape='
        #           f'{input_embeds[selected].shape}, '
        #           f'vit_embeds.shape={vit_embeds.shape}')
        #     n_token = selected.sum()
        #     input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        # input_embeds = input_embeds.reshape(B, N, C)
        
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        # shift_logits = logits[..., :-1, :]
        # shift_labels = labels[..., 1:]
        # import torch.nn.functional as F
        # softmax_logits = F.softmax(logits, dim=-1)
        # max_indices = torch.argmax(softmax_logits, dim=-1)
        # print('logits', max_indices.shape)
        # print(self.tokenizer.batch_decode(max_indices, skip_special_tokens=True))
        # label_without_neg_100 = (shift_labels != -100)
        # print('labels', shift_labels.shape)
        # print(self.tokenizer.batch_decode(shift_labels, skip_special_tokens=True))
        # print('##################')
        # import sys
        # sys.exit()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            # print(shift_logits.shape)
            # print(shift_labels.shape)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # def forward(self, data, data_samples=None, mode='loss'):
    #     pixel_values = data['pixel_values']

    #     if type(pixel_values) is list or pixel_values.ndim == 5:
    #         if type(pixel_values) is list:
    #             pixel_values = [
    #                 x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
    #             ]
    #         # b*n, c, h, w
    #         concat_images = torch.cat([
    #             image.to(self.model.vision_model.dtype)
    #             for image in pixel_values
    #         ],
    #                                   dim=0)
    #     else:
    #         raise NotImplementedError()

    #     input_ids = data['input_ids']
    #     position_ids = data['position_ids']
    #     attention_mask = data['attention_mask']
    #     # sum is 0 are text
    #     image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
    #     image_flags = image_flags.long()

    #     labels = data['labels']
    #     use_cache = False

    #     # Directly calling this code in LORA fine-tuning
    #     # will result in an error,so we must rewrite it.
    #     # TODO: Once the official is fixed, we can remove it.
    #     # outputs = self.model(input_ids=input_ids,
    #     #                      position_ids=position_ids,
    #     #                      attention_mask=attention_mask,
    #     #                      image_flags=image_flags,
    #     #                      pixel_values=concat_images,
    #     #                      labels=labels,
    #     #                      use_cache=use_cache)
    #     outputs = self._llm_forward(
    #         input_ids=input_ids,
    #         position_ids=position_ids,
    #         attention_mask=attention_mask,
    #         image_flags=image_flags,
    #         pixel_values=concat_images,
    #         labels=labels,
    #         use_cache=use_cache)
    #     loss_dict = {'loss': outputs.loss}
    #     return loss_dict
    
    # def _llm_forward(
    #     self,
    #     pixel_values: torch.FloatTensor,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     image_flags: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     return_dict = return_dict if return_dict is not None \
    #         else self.model.config.use_return_dict

    #     image_flags = image_flags.squeeze(-1)
    #     # We only added the clone code here to avoid the error.
    #     input_embeds = self.model.language_model.get_input_embeddings()(
    #         input_ids).clone()

    #     vit_embeds = self.model.extract_feature(pixel_values)
    #     vit_embeds = vit_embeds[image_flags == 1]
    #     vit_batch_size = pixel_values.shape[0]

    #     B, N, C = input_embeds.shape
    #     input_embeds = input_embeds.reshape(B * N, C)

    #     if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
    #         print(f'dynamic ViT batch size: {vit_batch_size}, '
    #               f'images per sample: {vit_batch_size / B}, '
    #               f'dynamic token length: {N}')
    #     self._count += 1

    #     input_ids = input_ids.reshape(B * N)
    #     selected = (input_ids == self.model.img_context_token_id)
    #     try:
    #         input_embeds[
    #             selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
    #                 -1, C)
    #     except Exception as e:
    #         vit_embeds = vit_embeds.reshape(-1, C)
    #         print(f'warning: {e}, input_embeds[selected].shape='
    #               f'{input_embeds[selected].shape}, '
    #               f'vit_embeds.shape={vit_embeds.shape}')
    #         n_token = selected.sum()
    #         input_embeds[
    #             selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

    #     input_embeds = input_embeds.reshape(B, N, C)

    #     outputs = self.model.language_model(
    #         inputs_embeds=input_embeds,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #     logits = outputs.logits

    #     loss = None
    #     if labels is not None:
    #         # Shift so that tokens < n predict n
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         # Flatten the tokens
    #         loss_fct = CrossEntropyLoss()
    #         shift_logits = shift_logits.view(
    #             -1, self.model.language_model.config.vocab_size)
    #         shift_labels = shift_labels.view(-1)
    #         # Enable model parallelism
    #         shift_labels = shift_labels.to(shift_logits.device)
    #         loss = loss_fct(shift_logits, shift_labels)

    #     if not return_dict:
    #         output = (logits, ) + outputs[1:]
    #         return (loss, ) + output if loss is not None else output

    #     return CausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )
