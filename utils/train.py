import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from open_clip.tokenizer import tokenize
from tqdm import tqdm

best_t2i_r1 = 0
best_t2i_r5 = 0
best_t2i_r10 = 0
best_i2t_r1 = 0
best_i2t_r5 = 0
best_i2t_r10 = 0
best_mr = 0


def train_one_epoch(model, train_loader, optimizer, criterion, epoch, device):
    model.train()

    tot_loss = 0.
    
    for i, (image, text, idx, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        ## fix length of token
        text_input = tokenize(text).to(device)
        
        img_emb = model.encode_image(image)
        txt_emb = model.encode_text(text_input)

        img_emb = F.normalize(img_emb)
        txt_emb = F.normalize(txt_emb)

        # Only use the CLIP contrastive loss - this is the standard approach
        loss = criterion(img_emb, txt_emb, model.logit_scale.exp())
        
        optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        tot_loss += loss.item()
    
    logger.info("Epoch {} - Average Loss: {:.4f}".format(epoch, tot_loss / len(train_loader)))

@torch.no_grad()
def evaluate(model, data_loader, device, args):
    logger.info("Running evaluation...")
    model.eval()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = args.bs
    text_embeds = []
    image_embeds = []

    # Inference img features
    logger.info("Encoding images...")
    for i, (image, img_id) in enumerate(tqdm(data_loader)):
        image = image.to(device)
        
        image_embed = model.encode_image(image)
        image_embed = F.normalize(image_embed)
        image_embeds.append(image_embed)
    
    # Inference text features
    logger.info("Encoding texts...")
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenize(text).to(device)

        text_embed = model.encode_text(text_input)
        text_embed = F.normalize(text_embed)

        text_embeds.append(text_embed)
    
    # calculate similarity matrix
    image_embeds = torch.cat(image_embeds, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)
    logger.info(f"Image embeddings shape: {image_embeds.shape}")
    logger.info(f"Text embeddings shape: {text_embeds.shape}")
    
    sims_matrix = image_embeds @ text_embeds.t()

    score_matrix_i2t = sims_matrix
    score_matrix_t2i = sims_matrix.t()
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    logger.info("Computing metrics...")
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': round(tr1,2),
                   'txt_r5': round(tr5,2),
                   'txt_r10': round(tr10,2),
                   'img_r1': round(ir1,2),
                   'img_r5': round(ir5,2),
                   'img_r10': round(ir10,2),
                   'r_mean': round(r_mean,2)}
    
    global best_i2t_r1
    global best_i2t_r5
    global best_i2t_r10
    global best_t2i_r1
    global best_t2i_r5
    global best_t2i_r10
    global best_mr
    
    if eval_result['img_r1'] > best_t2i_r1:
        best_t2i_r1 = eval_result['img_r1']
    if eval_result['img_r5'] > best_t2i_r5:
        best_t2i_r5 = eval_result['img_r5']
    if eval_result['img_r10'] > best_t2i_r10:
        best_t2i_r10 = eval_result['img_r10']

    if eval_result['txt_r1'] > best_i2t_r1:
        best_i2t_r1 = eval_result['txt_r1']
    if eval_result['txt_r5'] > best_i2t_r5:
        best_i2t_r5 = eval_result['txt_r5']
    if eval_result['txt_r10'] > best_i2t_r10:
        best_i2t_r10 = eval_result['txt_r10']

    best_mr = (best_i2t_r1 + best_t2i_r1 + best_i2t_r5 + best_t2i_r5 + best_i2t_r10 + best_t2i_r10) / 6

    logger.info(">" * 20)
    logger.info(">" * 20)
    logger.info(f'best_i2t_r1: {best_i2t_r1:.2f}, best_i2t_r5: {best_i2t_r5:.2f}, best_i2t_r10: {best_i2t_r10:.2f}')
    logger.info(f'best_t2i_r1: {best_t2i_r1:.2f}, best_t2i_r5: {best_t2i_r5:.2f}, best_t2i_r10: {best_t2i_r10:.2f}')
    logger.info(f'BEST_MR: {best_mr:.2f}')
    logger.info(">" * 20)
    logger.info(">" * 20)

    return eval_result