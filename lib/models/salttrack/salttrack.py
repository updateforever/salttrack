"""
SALTTrack  Model
"""
import os

import torch
import math
from torch import nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor

# from .language_model import build_bert
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh, box_xyxy_to_cxcywh, box_iou
### aqatrack
from lib.models.aqatrack.hivit import hivit_small, hivit_base
from lib.models.aqatrack.itpn import itpn_base_3324_patch16_224
from lib.models.aqatrack.fast_itpn import fast_itpn_base_3324_patch16_224,fast_itpn_large_2240_patch16_256

from lib.models.transformers.transformer import build_rgb_det_decoder
from lib.models.layers.transformer_dec import build_transformer_dec,build_transformer_dec_with_mask

from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head

import torch.nn.functional as F
from lib.models.layers.frozen_bn import FrozenBatchNorm2d
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
from lib.models.transformers import build_decoder, VisionLanguageFusionModule, PositionEmbeddingSine1D,build_text_prompt_decoder
from lib.models.layers.lora import apply_lora_to_modules
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
class ConfidencePred(nn.Module):
    def __init__(self):
        super(ConfidencePred, self).__init__()
        self.feat_sz = 24
        self.stride = 1
        self.img_sz = self.feat_sz * self.stride
        freeze_bn = False

        # CNN
        self.conv1_ctr = conv(5, 16, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(16, 16 // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(16 // 2, 16 // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(16 // 4, 16 // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(16 // 8, 1, kernel_size=1)

        # 定义全连接层
        self.fc1 = nn.Linear(256, 512)

        ## cross attn 交互层
        # self.multihead_attn = nn.MultiheadAttention(512, 4, dropout=0.1)
        # # Implementation of Feedforward model
        # self.dropout = nn.Dropout(0.1)
        # self.norm1 = nn.LayerNorm(512)


        self.fc2 = nn.Linear(512, 1)

        # 定义激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x,xz_feature=None, gt_score_map=None):
        """ Forward pass with input x. """

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # 展平输入
        x = score_map_ctr.flatten(1)
        x = self.relu(self.fc1(x))

        x = self.sigmoid(self.fc2(x))

        return x

class SubjectIndexPred(nn.Module):
    def __init__(self,dim):
        super(SubjectIndexPred, self).__init__()

        # 定义全连接层
        self.fc1 = nn.Linear(dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """

        # 全连接层前向传播
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x


class SALTTrack(nn.Module):
    """ This is the base class for SALTTrack"""
    def __init__(self, transformer,  box_head, tokenizer, text_encoder, aux_loss=False, head_type="CORNER",dim=512,cfg=None):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.dim = dim

        self.query_len = 1
        self.cls_prompts_pos = nn.Embedding(num_embeddings=self.query_len, embedding_dim=self.dim )  # pos for cur query
        # self.cls_initial= nn.Embedding(num_embeddings=self.query_len, embedding_dim=self.dim )  # pos for cur query
        self.confidence_pred = ConfidencePred()

        # visual_temporal_fusion:
        # 用于维护一个短期视觉时序记忆。这里的 query 不是整张搜索图，而是历史时刻的
        # visual prompt token；memory 是当前搜索区域特征。它对应论文里“动态目标状态”
        # 那部分，让当前帧特征和历史状态交互。
        self.visual_temporal_fusion = build_transformer_dec_with_mask(cfg, self.dim )
        self.temporal_len = 4
        self.dy_template_pos_embed = nn.Embedding(num_embeddings=self.temporal_len,
                                                  embedding_dim=self.dim )  # pos for cur query

        # 文本分支：
        # 1. tokenizer/text_encoder 先把自然语言转成 token-level contextual features。
        # 2. text_adj 把语言模型输出维度映射到跟视觉分支一致的 hidden dim。
        # 3. text_sub_idnex_classifier 预测每个 token 是否更偏向“目标词”而非上下文词。
        # 4. language_adjust + vl_fusion 则负责把这些语言提示真正注入视觉搜索特征。
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_adj = nn.Sequential(
            nn.Linear(768, self.dim , bias=True),
            nn.LayerNorm(self.dim , eps=1e-12),
            nn.Dropout(0.1),
        )

        # language_adjust:
        # 输入是 [原始文本特征, 目标词特征 + 时序记忆] 两路信息。
        # 它的作用不是直接和视觉融合，而是先在语言侧做一次“目标/上下文重整”：
        # 用 temporal memory 去更新目标词相关的语言表示。
        self.language_adjust = build_transformer_dec(cfg, self.dim )

        # vl_fusion:
        # 真正的视觉-语言融合模块。query 是当前搜索区域视觉 token，memory 是语言侧
        # guidance memory（不是原始 token id，而是编码后的文本 memory）。
        self.vl_fusion = VisionLanguageFusionModule(dim=self.dim , num_heads=8, attn_drop=0.1, proj_drop=0.1,
                                                    num_vlfusion_layers=2,
                                                    vl_input_type='separate')

        self.text_pos = PositionEmbeddingSine1D(self.dim , normalize=True)

        # 对每个 token 预测一个 [0,1] 权重，表示该 token 更像目标词还是上下文词。
        # 这是代码里 textual target-context guidance 的入口。
        self.text_sub_idnex_classifier = SubjectIndexPred(self.dim)

        # WYP: 语义监督模块（新增）
        self.use_semantic_align = cfg.MODEL.get('USE_SEMANTIC_ALIGN', False) if cfg else False

    def forward_backbone(self, template, search, cls_token,soft_token_template_mask,x_pos):
        # template: [B, 6, H, W]，由两个 template frame 拼成
        # search:   [B, 3, H, W] 或训练时堆叠后的 [B*T, 3, H, W]
        # backbone 内部把 template 拆成两个 RGB 模板，以便执行 temporal target-context 建模。
        template = [template[:,:3],template[:,3:]]
        soft_token_template_mask = [soft_token_template_mask[:, :64], soft_token_template_mask[:, 64:]]

        x, token_type_infor = self.backbone.forward_features_pe(z=template, x=search, soft_token_template_mask =soft_token_template_mask)
        x, aux_dict = self.backbone.forward_features_stage3(x, cls_token,x_pos)
        return x, aux_dict

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                soft_token_template_mask=None,
                exp_str=None,
                exp_subject_mask=None,
                search_anno=None,
                temporal_infor=[],
                first_frame_flag=False,
                training=True):

        # b0: batch size
        # num_search: 训练时每个样本包含多少个 search frame，用于模拟短时序目标状态变化
        b0, num_search = template[0].shape[0], len(search)
        if training:
            # 训练时把多个 search frame 沿 batch 维拼起来，后面再按 temporal_index 拆开处理。
            search = torch.cat(search, dim=0)
            template = torch.cat(template, dim=1)  # (bs,6(rgb0;rgb1),w,h)
            soft_token_template_mask = torch.cat(soft_token_template_mask,
                                                              dim=1)  # (bs,128(mask0;mask1),1)
            template_temporal = []
            soft_token_template_mask_temporal = []
            for _ in range(num_search):
                # 每个 search step 共用同一组 template 和 template mask。
                template_temporal.append(template)
                soft_token_template_mask_temporal.append(soft_token_template_mask)
            template_temporal = torch.cat(template_temporal, dim=0)
            soft_token_template_mask_temporal = torch.cat(soft_token_template_mask_temporal,dim=0)

        else:
            # 测试时是在线跟踪模式，不做多 search frame 拼接。
            b0 = 1
            template_temporal = torch.cat(template, dim=1)
            soft_token_template_mask_temporal = torch.cat(soft_token_template_mask, dim=1)

        # 给 cls prompt、template token、search token 拼接位置编码。
        # backbone 输出 x 是后续所有视觉建模的基础 token 序列。
        cls_prompts_pos = self.cls_prompts_pos.weight.unsqueeze(0)
        x_pos_0 = torch.cat([cls_prompts_pos, self.backbone.pos_embed_z, self.backbone.pos_embed_x], dim=1)
        x_pos = x_pos_0.repeat(b0*num_search, 1, 1)
        x, aux_dict = self.forward_backbone(template_temporal, search, None, soft_token_template_mask_temporal,
                                                 x_pos)

        # 文本分支：
        # forward_text 返回的不是原始 token，而是编码后的 token-level features。
        # text_features: 原始文本编码特征
        # text_subject_features: 用 target-word mask 加权后的目标相关文本特征
        # subject_infor_mask_pred: 模型预测的目标词概率
        # subject_infor_mask_gt: 数据集提供的目标词监督
        if training:
            if exp_str:
                text_features, text_subject_features, subject_infor_mask_pred, subject_infor_mask_gt  = self.forward_text(
                    exp_str, num_search, exp_subject_mask, device=search.device)  # text_subject_features, subject_infor_mask_pred, subject_infor_mask_gt
        else:
            text_features = exp_str
            text_subject_features = exp_subject_mask
            subject_infor_mask_pred = None
            subject_infor_mask_gt = None
            
        batch_size = text_features.tensors.shape[0]
        text_pos = self.text_pos(text_features) # [batch_size, length, c]
        text_pos_0 = text_pos[:b0]
        x_s_pos_item = x_pos_0.repeat(b0, 1, 1)[:, -self.feat_len_s:]
        pre_temporal_pos = self.dy_template_pos_embed.weight.unsqueeze(1)
        pre_temporal_pos = pre_temporal_pos.repeat(b0, 1, self.query_len)
        pre_temporal_pos = pre_temporal_pos.view(b0, self.temporal_len * self.query_len, self.dim).contiguous()

        # xt_data 存放每个 search step 经过视觉-语言融合 + 时序更新后的搜索特征图，
        # 最后统一送入 box head 做预测。
        xt_data = []
        # WYP: 收集每个 search step 的增强文本特征，后面直接作为监督侧语义锚点。
        target_text_feature_list = []
        for temporal_index in range(num_search):
            x_item = x[temporal_index * b0:(temporal_index + 1) * b0]

            # backbone 输出的第一个 token 被当作当前帧的视觉 prompt / 状态 token。
            visual_prompts_token = x_item[:, :self.query_len, :]

            # 这里构造一个粗粒度的 template-search 关联强度 attn_xz。
            # 它不是标准 transformer 里直接导出的 attention，而是作者基于 token 相似性
            # 手工构造的一张“与模板关联程度”掩码，用于后面的 temporal fusion。
            x_f = x_item[:, -256:]
            x_f1 = torch.matmul(x_f, x_f.permute(0, 2, 1).contiguous())
            x_f = torch.matmul(x_f1, x_f)

            z_f = x_item[:, :-256]

            x_z = torch.matmul(x_f, z_f.permute(0, 2, 1).contiguous())
            att_map = x_z.mean(-1)

            tensor_min = torch.min(att_map)
            tensor_max = torch.max(att_map)
            # normalized_tensor = (s_vl_1 - tensor_min) / (tensor_max - tensor_min)
            normalized_tensor = (tensor_max - att_map) / (tensor_max - tensor_min)

            attn_xz = normalized_tensor.view(-1, 256,1).contiguous()

            # temporal_infor 是一个短时视觉 memory bank。
            # 它存的不是整张特征图，而是历史时刻的 temporal/state token。
            # 训练时在第一个 search step 初始化；测试时只在第一帧初始化。
            if training:
                if temporal_index == 0:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(visual_prompts_token)
            else:
                if first_frame_flag:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(visual_prompts_token)

            temporal_infor_data = torch.cat(temporal_infor, dim=1)

            # -------- Textual target-context guidance --------
            # l_item_initial:
            #   原始文本编码特征，对应整句 token-level contextual features。
            # l_item_subject:
            #   用 subject mask 加权后的目标相关文本特征。
            # l_subject_temporal:
            #   把“目标词特征”和“视觉时序记忆”拼接起来，表示当前文本提示不只依赖静态句子，
            #   还参考了历史目标状态。
            l_item_initial = text_features.tensors[temporal_index * b0:(temporal_index + 1) * b0]
            l_item_subject = text_subject_features.tensors[temporal_index * b0:(temporal_index + 1) * b0]
            l_mask_item_0 = text_features.mask[temporal_index * b0:(temporal_index + 1) * b0]
            temporal_mask = torch.ones((l_mask_item_0.shape[0],self.temporal_len)).bool().to(l_mask_item_0.device)
            l_mask_item = torch.cat([l_mask_item_0, temporal_mask],dim=1)

            l_subject_temporal = torch.cat([l_item_subject,temporal_infor_data],dim=1)
            l_subject_temporal_pos = torch.cat([text_pos_0,pre_temporal_pos ],dim=1)

            # language_adjust 在语言侧做一次“目标/上下文重整”：
            # 用目标词 + 时序 memory 去更新原始文本特征，得到 l_item_update。
            # 这里输出的已经不是 token id，而是更新后的 language guidance features。
            l_item_update,_ = self.language_adjust([l_item_initial,l_subject_temporal],None,
                                          text_pos_0,l_subject_temporal_pos,l_mask_item)

            # l_all 是最终送入视觉语言融合模块的 text memory。
            # 它最接近论文图里 text guidance module 后面的 memory M。
            l_all = torch.cat([ l_item_initial,l_item_update ],dim=1)

            # WYP: 直接对 SALTTrack 增强后的文本特征做 masked mean pooling，得到句子级文本特征。
            valid_text_mask = (~l_mask_item_0).float().unsqueeze(-1)
            pooled_text_feature = (l_item_update * valid_text_mask).sum(dim=1) / valid_text_mask.sum(dim=1).clamp(min=1.0)
            pooled_text_feature = pooled_text_feature / (pooled_text_feature.norm(dim=-1, keepdim=True) + 1e-8)
            target_text_feature_list.append(pooled_text_feature)
            x_s_item = x_item[:, -self.feat_len_s:]

            # vl_fusion:
            # query  = 搜索区域视觉 token
            # memory = 语言侧 guidance memory (l_all)
            # 也就是说，视觉搜索特征是被语言提示调制的，而不是反过来。
            x_s_item = self.vl_fusion(x_s_item,
                                 l_all,
                                 query_pos=x_pos_0[:, -self.feat_len_s:],
                                 memory_pos=torch.cat([text_pos_0,text_pos_0],dim=1),
                                 memory_key_padding_mask=torch.cat([l_mask_item_0,l_mask_item_0],dim=1),
                                 need_weights=False)


            # 再用 temporal memory 对当前搜索特征做一次状态更新。
            # 这一步对应“动态目标状态建模”，让当前帧不仅受文本约束，也受历史视觉状态约束。
            temporal_infor_update = self.visual_temporal_fusion(temporal_infor_data, x_s_item, attn_xz,pre_temporal_pos ,kv_pos= x_s_pos_item )
            temporal_item = temporal_infor_update[:,-1,:].unsqueeze(1)

            # STM 部分把最终时序 token temporal_item 和当前搜索特征 x_s_item 组合，
            # 得到可以直接送入预测头的 2D feature map。
            enc_opt = x_s_item
            dec_opt = temporal_item.transpose(1, 2)
            att = torch.matmul(enc_opt, dec_opt)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

            xt_data.append(opt_feat)

            # 更新 memory bank，保留最近 temporal_len 个状态 token。
            if training:
                if temporal_index == 0:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(temporal_item)
                else:
                    temporal_infor[:-1] = temporal_infor[1:]
                    temporal_infor[-1] = temporal_item
            else:
                if first_frame_flag:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(temporal_item)

                else:
                    temporal_infor[:-1] = temporal_infor[1:]
                    temporal_infor[-1] = temporal_item


        # 所有 search step 的特征图拼回一个 batch，统一送入 head。
        xt_data = torch.cat(xt_data,dim=0)
        out = self.forward_head(xt_data, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['subject_infor_mask_pred'] = subject_infor_mask_pred
        out['subject_infor_mask_gt'] = subject_infor_mask_gt

        # WYP: 语义监督分支（新增）
        # 不再引入 CLIP 外部对齐空间，而是直接使用：
        # 1. 预测框 RoI 特征
        # 2. GT 框 RoI 特征
        # 3. SALTTrack 增强后的文本特征
        if self.use_semantic_align and training:
            pred_boxes = out['pred_boxes'].squeeze(1)  # [B*num_search, 4]
            pred_visual_features = self.extract_instance_features(xt_data, pred_boxes)
            gt_visual_features = None
            if search_anno is not None:
                gt_boxes = search_anno.view(-1, 4)
                gt_visual_features = self.extract_instance_features(xt_data, box_xywh_to_cxcywh(gt_boxes))
            target_text_features = torch.cat(target_text_feature_list, dim=0) if len(target_text_feature_list) > 0 else None

            out['pred_visual_features'] = pred_visual_features
            out['gt_visual_features'] = gt_visual_features
            out['target_text_features'] = target_text_features

        if training == False:
            out["temporal_infor"] = temporal_infor

        return out

    def forward_head(self, opt_feat, gt_score_map=None):
        """
        这里输入的 opt_feat 已经是 [B, C, H, W] 的 2D feature map，
        不是原始 token 序列。也就是说，SALTTrack 在 head 前已经把
        “视觉 + 文本 + 时序状态”三者的信息压进了一个可预测的特征图里。
        """

        # enc_opt = cat_feature #[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s).contiguous()

        bs = opt_feat.shape[0]
        Nq = 1
        # Head
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4).contiguous()
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # CENTER head 输出:
            # score_map_ctr: 中心热力图
            # bbox:          预测框
            # size_map:      宽高
            # offset_map:    中心偏移
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)

            # confidence_pred 额外回归当前结果的可靠性，可理解为 long-term tracking 下的质量估计。
            score_map = torch.cat([score_map_ctr, size_map, offset_map], dim=1)
            confidence_pred = self.confidence_pred(score_map)

            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4).contiguous()
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   "confidence_pred": confidence_pred}
            return out
        else:
            raise NotImplementedError

    def forward_text(self, captions, num_search, exp_subject_mask, device):
        # tokenized 是分词结果；真正参与模型的是后面的 encoded_text.last_hidden_state。
        tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.text_encoder(**tokenized)

        # 这里的 mask 约定是：
        # True  表示 padding 位置
        # False 表示有效 token
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        # text_features 是上下文化后的 token-level language features。
        # 到这里已经不是离散 token，而是每个 token 对应的连续语义表示。
        text_features = encoded_text.last_hidden_state
        text_features = self.text_adj(text_features)

        encodings_infor = tokenized.encodings

        subject_infor_mask_gt = None
        if exp_subject_mask is not None:
            # exp_subject_mask 来自数据集标注，表示哪些“词级位置”属于 target words。
            # 这里要把“词级标注”对齐到 tokenizer 产生的 subword/token 级索引上。
            subject_infor_mask_gt = torch.zeros(text_attention_mask.shape[0], text_attention_mask.shape[1]).to(
                text_features.device)

            for item_index, item in enumerate(encodings_infor):
                word_ids_item = item.word_ids
                exp_subject_mask_item = exp_subject_mask[item_index]
                text_index_list = []
                for word_index, word_item in enumerate(word_ids_item):
                    if word_item in exp_subject_mask_item:
                        text_index_list.append(word_index)

                subject_infor_mask_gt[item_index, text_index_list] = 1

        # subject_infor_mask_pred 是每个 token 属于“目标词”的概率预测。
        # 它就是代码里 textual target-context guidance 的核心输出之一。
        subject_infor_mask_pred = self.text_sub_idnex_classifier(text_features)
        subject_infor_mask_pred_1 = subject_infor_mask_pred.expand_as(text_features)

        # 不是把目标词真的“裁掉”或“选出来”，而是用 soft mask 对文本特征做加权。
        # 所以 subject_infor 仍然是连续特征，只是更偏向目标相关词。
        subject_infor = text_features * subject_infor_mask_pred_1

        # 训练时有多个 search step；文本描述在这些 step 中共享，所以沿 batch 维复制。
        text_features_t = []
        text_attention_mask_t = []
        text_subject_infor_t = []
        for i in range(num_search):
            text_features_t.append(text_features)
            text_attention_mask_t.append(text_attention_mask)
            text_subject_infor_t.append(subject_infor)

        text_features = torch.cat(text_features_t, dim=0)
        text_attention_mask = torch.cat(text_attention_mask_t, dim=0)
        text_features = NestedTensor(text_features, text_attention_mask)
        subject_infor = torch.cat(text_subject_infor_t, dim=0)
        subject_infor = NestedTensor(subject_infor, text_attention_mask)

        return text_features, subject_infor, subject_infor_mask_pred, subject_infor_mask_gt

    def extract_instance_features(self, feature_map, bbox_pred):
        """
        WYP: 新增函数。
        使用 RoIAlign 从融合特征图中裁剪当前预测框对应的实例特征。
        这一步是后续语义对齐分支的桥梁：把 tracking feature map 变成实例级向量。
        Args:
            feature_map: [B, C, H, W] 融合特征图
            bbox_pred: [B, 4] 预测框 (cx, cy, w, h) 归一化坐标
        Returns:
            instance_features: [B, C] 实例特征向量
        """
        from torchvision.ops import roi_align

        B, C, H, W = feature_map.shape
        bbox_pred = bbox_pred.to(feature_map.device, dtype=feature_map.dtype)

        # 将 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)
        bbox_xyxy = box_cxcywh_to_xyxy(bbox_pred)

        # 转换为绝对坐标（相对于特征图尺寸）
        bbox_xyxy_abs = bbox_xyxy * torch.tensor([W, H, W, H], device=bbox_xyxy.device, dtype=feature_map.dtype)

        # RoIAlign 需要 [batch_index, x1, y1, x2, y2] 格式
        batch_indices = torch.arange(B, device=bbox_xyxy.device, dtype=feature_map.dtype).unsqueeze(1)
        rois = torch.cat([batch_indices, bbox_xyxy_abs], dim=1)

        # 执行 RoIAlign，输出尺寸 7x7
        spatial_scale = 1.0  # 因为已经转换为特征图坐标
        roi_features = roi_align(feature_map, rois, output_size=(7, 7),
                                spatial_scale=spatial_scale, aligned=True)

        # Global Average Pooling: [B, C, 7, 7] -> [B, C]
        instance_features = roi_features.mean(dim=[2, 3])
        instance_features = instance_features / (instance_features.norm(dim=-1, keepdim=True) + 1e-8)

        return instance_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_salttrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../resource/pretrained_models')
    pretrained_ckpt_path = getattr(cfg.MODEL, "PRETRAINED_PATH", "")

    # WYP: 如果已经指定了 SALTTrack 整网 checkpoint，则不再额外加载 backbone 预训练权重。
    if cfg.MODEL.PRETRAIN_FILE and training and (not pretrained_ckpt_path) and ("SALTTrack" not in cfg.MODEL.PRETRAIN_FILE):
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''


    if cfg.MODEL.BACKBONE.TYPE == 'hivit_base_adaptor':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'itpn_base':  # by this
        backbone = fast_itpn_base_3324_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'itpn_large':  # by this
        backbone = fast_itpn_large_2240_patch16_256(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg,dim=hidden_dim, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    # Build Text Encoder
    tokenizer = RobertaTokenizerFast.from_pretrained(
        os.path.join(pretrained_path, 'roberta-base'))  # load pretrained RoBERTa Tokenizer
    text_encoder = RobertaModel.from_pretrained(
        os.path.join(pretrained_path, 'roberta-base'))  # load pretrained RoBERTa model


    model = SALTTrack(
        backbone,
        box_head,
        tokenizer,
        text_encoder,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        dim = hidden_dim,
        cfg=cfg
    )

    if pretrained_ckpt_path and training:
        # WYP: 语义微调默认从 SALTTrack 已训练权重继续，而不是只从 backbone 预训练开始。
        checkpoint = torch.load(pretrained_ckpt_path, map_location="cpu")
        model_weight = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(model_weight, strict=False)
        print("Load pretrained model from:", pretrained_ckpt_path)
        if len(missing_keys) > 0:
            print("Missing keys:", missing_keys)
        if len(unexpected_keys) > 0:
            print("Unexpected keys:", unexpected_keys)

    # WYP: 文本侧作为稳定语义锚点，默认冻结文本编码、文本映射和目标词分类器。
    if getattr(cfg.TRAIN, "FREEZE_TEXT_SIDE", True):
        for module_name in ["text_encoder", "text_adj", "text_sub_idnex_classifier"]:
            module = getattr(model, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    # WYP: LoRA 仅注入指定的视觉/融合/预测模块，避免直接扰动文本锚点。
    # 这里训练和推理都要注入同样的 LoRA 结构，否则测试时无法完整加载 LoRA checkpoint。
    #
    # TYPE=standard:
    #   每个 nn.Linear 替换为标准单分支 LoRA，用作最直接的 PEFT baseline。
    # TYPE=routed:
    #   每个 nn.Linear 替换为双专家 SRR-LoRA。EXPERT_RANK 控制单个 expert
    #   的 rank；例如 standard RANK=8、routed EXPERT_RANK=4 时，两个 expert
    #   的 A/B 参数量大致与标准 LoRA 对齐，方便做参数量受控的消融。
    lora_cfg = getattr(cfg.MODEL, "LORA", None)
    lora_enabled = bool(lora_cfg and getattr(lora_cfg, "ENABLED", False))
    if lora_enabled:
        target_modules = list(getattr(lora_cfg, "TARGET_MODULES", ["vl_fusion", "visual_temporal_fusion"]))
        lora_type = getattr(lora_cfg, "TYPE", "standard")
        lora_rank = getattr(lora_cfg, "EXPERT_RANK", getattr(lora_cfg, "RANK", 8)) \
            if lora_type == "routed" else getattr(lora_cfg, "RANK", 8)
        replaced_layers = apply_lora_to_modules(
            model,
            target_module_names=target_modules,
            rank=lora_rank,
            alpha=getattr(lora_cfg, "ALPHA", 16),
            dropout=getattr(lora_cfg, "DROPOUT", 0.0),
            lora_type=lora_type,
        )
        print("WYP: Applied {} LoRA to layers:".format(lora_type))
        for layer_name in replaced_layers:
            print("  ", layer_name)


    return model

def load_pretrained(model, pretrained_path, strict=False):

    model_ckpt = torch.load(pretrained_path, map_location="cpu")
    state_dict = model_ckpt['net']
    pos_st = state_dict['encoder.body.pos_embed']
    pos_s = pos_st[:,:(pos_st.size(1) // 2)]
    pos_t = pos_st[:,(pos_st.size(1) // 2):]
    state_dict['encoder.body.pos_embed_search'] = pos_s
    state_dict['encoder.body.pos_embed_template'] = pos_t
    state_dict['encoder.body.patch_embed_interface.proj.weight'] = state_dict['encoder.body.patch_embed.proj.weight']
    state_dict['encoder.body.patch_embed_interface.proj.bias'] = state_dict['encoder.body.patch_embed.proj.bias']
    state_dict['decoder.embedding.prompt_embeddings.weight'] = model.state_dict()['decoder.embedding.prompt_embeddings.weight']
    state_dict['decoder.embedding.prompt_embeddings.weight'][:] = state_dict['decoder.embedding.word_embeddings.weight'][-1]
    del state_dict['encoder.body.pos_embed']
    model.load_state_dict(state_dict, strict=strict)
