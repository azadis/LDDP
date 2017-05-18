#!/usr/bin/env python

# --------------------------------------------------------
# LDDP
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Samaneh Azadi
# --------------------------------------------------------

import numpy as np
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes,bbox_transform
from boxTools import *
from fast_rcnn.config import cfg


class DPP():
    def __init__(self,stds=[],means=[],sim_classes=[],epsilon=0.01,loss_weight=0.001):
        
        self.stds =stds
        self.means = means
        self.sim_classes = sim_classes
        self.epsilon = epsilon
        self._loss_weight = loss_weight
        
    
    def select_bg(self,Phi_labels,boxes,labels,bbox_pred,keeps_Y,good_gt_overlap,M,im_shape_w,im_shape_h):
        """
        Find B in p(B|Xb)
        """
        selected_item = range(M)
        prob_dpp = np.ones((M,))
        ignores=[]
        dict_keeps_Y = {}
        for i,j in keeps_Y.iteritems():
            if j not in dict_keeps_Y:
                dict_keeps_Y[j]=[]
            dict_keeps_Y[j].append(i)
    
        for k in range(M):
            if (k in keeps_Y and keeps_Y[k]==Phi_labels[k]) \
            or (k in good_gt_overlap and Phi_labels[k]==labels[k] and labels[k]>0):
                ignores.append(k)
            else:
                label_k = labels[k]
                if label_k in dict_keeps_Y:
                    loc_lbl = bbox_pred[[k],4*label_k:4*(label_k+1)]
                    loc_lbl = loc_lbl * self.stds[label_k,:] + self.means[label_k,:]
                    pbox = bbox_transform_inv(boxes[[k],:], loc_lbl)
                    pbox = clip_boxes(pbox, (im_shape_w,im_shape_h))
                    pbox = np.reshape(np.tile(pbox,len(dict_keeps_Y[label_k])),(len(dict_keeps_Y[label_k]),4))
            
                    Y_selected_ll = bbox_pred[dict_keeps_Y[label_k],4*label_k:4*(label_k+1)]
                    Y_selected_ll = Y_selected_ll*self.stds[label_k,:] + self.means[label_k,:]
                    Y_selected_pbox = bbox_transform_inv(boxes[dict_keeps_Y[label_k],:], Y_selected_ll)
                    Y_selected_pbox = clip_boxes(Y_selected_pbox, (im_shape_w,im_shape_h))
                    if np.max(IoU_target(pbox,Y_selected_pbox)) > cfg.TRAIN.IGNORANCE:
                        ignores.append(k)

        selected_item = np.array([x for ii,x in enumerate(selected_item) if ii not in ignores])
        prob_dpp = [x for ii,x in enumerate(prob_dpp) if ii not in ignores]
        return selected_item,prob_dpp

    def dpp_greedy(self,S, scores_s, score_power, max_per_image, among_ims, num_gt_per_img=1000, close_thr=0.0001): 
        """ 
        Greedy optimization to select boxes
        S: similarity matrix
        scores_s : predicted scores over different categories

        """
        prob_thresh = cfg.TEST.PROB_THRESH
        S = S[among_ims,:][:,among_ims]
        scores_s = scores_s[among_ims]    

        M = S.shape[0]
    
        #keep: selected_boxes
        keep = []
    
        #left : boxes not selected yet
        left = np.zeros((M,3))
        left[:,0] = np.arange(M) #box number
        left[:,1] = 1 # 0/1? Is the box left?
        selected_prob = []
        while (len(keep) < max_per_image) and sum(left[:,1])>0:
            z = np.zeros((M,1))
            z[keep] = 1
            sum_scores = (score_power*np.log(scores_s).T).dot(z)
            prob_rest = np.zeros((M,))
            left_indices = np.where(left[:,1]==1)[0]
            done_indices = np.where(left[:,1]==0)[0]
            if len(keep)>0:
                S_prev = S[keep,:][:,keep]
                det_D = np.linalg.det(S_prev)
                d_1 = np.linalg.inv(S_prev)
            else:
                det_D = 1
                d_1 = 0
            # ====================================================================
            #     |D  a^T|
            # det(|a    b|)= (b - a D^{-1} a^T)det(D)
            #
            # Here "D" = S_prev and "a","b" are the similarity values added by each single item
            # in left_indices.
            # To avoid using a for loop, we compute the above det for all items in left_indices
            # all at once through appropriate inner vector multiplications as the next line:  
            
            # ====================================================================
            if len(keep)>0:
                prob_rest[left_indices] =- np.sum(np.multiply(np.dot(S[left_indices,:][:,keep],d_1),S[left_indices,:][:,keep]),1)

            prob_rest[left_indices] = np.log((prob_rest[left_indices] + S[left_indices,left_indices]) * det_D)+\
                           (sum_scores + score_power * np.log(scores_s[(left[left_indices,0]).astype(int)]))                
            
            prob_rest[done_indices] = np.min(prob_rest)-100
            max_ind = np.argmax(prob_rest)
            ind = left[max_ind,0]
            close_inds = np.where(prob_rest >= (prob_rest[max_ind] + np.log(close_thr)))[0]
            far_inds = np.where(prob_rest < (prob_rest[max_ind] + np.log(close_thr)))[0]
            tops_prob_rest = np.argsort(-prob_rest[close_inds]).astype(int)
            if len(keep) >= num_gt_per_img:
                break
            elif len(keep)> 0:
                cost = np.max(S[np.array(range(M))[close_inds][tops_prob_rest],:][:,keep],1)
                good_cost = list(np.where(cost <= prob_thresh)[0])
                bad_cost = list(np.where(cost > prob_thresh)[0])
                if len(good_cost)>0:
                    ind = np.array(range(M))[close_inds][tops_prob_rest[good_cost[0]]]
                    keep.append(ind)
                    left[ind,1] = 0
                    #left[far_inds,1]=0
                    selected_prob.append(prob_rest[max_ind])
                else:
                    left[:,1]=0


            else:
                keep.append(max_ind)
                left[max_ind,1] = 0
                selected_prob.append(prob_rest[max_ind])

            
        return keep,selected_prob
    
    
    def dpp_MAP(self,im_dets_pair, scores, boxes,sim_classes,score_thresh,epsilon,max_per_image,close_thr=0.00001):
       """
       DPP MAP inference
       """
       M0 = boxes.shape[0]
       num_classes = scores.shape[1]
       scores = scores[:,1:] #ignore background
       
       # consider only top 5 class scores per box
       num_ignored = scores.shape[1]-5
       sorted_scores = np.argsort(-scores,1)
       ignored_cols = np.reshape(sorted_scores[:,-num_ignored:],(M0*num_ignored))
       ignored_rows = np.repeat(range(0,sorted_scores.shape[0]),num_ignored)
       scores[ignored_rows,ignored_cols] = 0
       high_scores = np.nonzero(scores >= score_thresh)
       lbl_high_scores = high_scores[1]
       box_high_scores = high_scores[0]
       scores_s = np.reshape(scores[box_high_scores, lbl_high_scores],(lbl_high_scores.shape[0],))

   
       boxes = boxes[:,4:]
       boxes_s = np.reshape(boxes[np.tile(box_high_scores,4), np.hstack((4*lbl_high_scores,4*lbl_high_scores+1,\
       4*lbl_high_scores+2,4*lbl_high_scores+3))] ,(lbl_high_scores.shape[0],4),order='F')
       M = boxes_s.shape[0]
       sim_power = cfg.TEST.SIM_POWER
       sim_boxes = sim_classes[(lbl_high_scores),:][:,(lbl_high_scores)]
       sim_boxes = sim_boxes**sim_power
       keep_ = {} 

       if M>0:
           IoU = pair_IoU(boxes_s)
           IoU[np.where(IoU<cfg.TEST.DPP_NMS)] = 0
           # S = IoU * sim + \epsilon *I_M
           S = np.multiply(IoU,sim_boxes) + epsilon * np.eye(M,M)
           keep = self.dpp_greedy(S, scores_s, 1.0, max_per_image, np.array(range(M)), close_thr=close_thr)[0]
           keep_['box_id'] = box_high_scores[keep]
           keep_['box_cls'] = lbl_high_scores[keep]+1
       else:
          keep_['box_id'] = []
          keep_['box_cls'] = []       
   
       return keep_ 


    def compute_kernel(self, labels, boxes, Phi, loc_argmax, unnormalized_bbox_targets, im_shape_w, im_shape_h):
        """
        Compute DPP Kernel Matrix
        """
        M = boxes.shape[0] # number of rois of 1 image in the minibatch

        pred_boxes = bbox_transform_inv(boxes, loc_argmax)
        pred_boxes = clip_boxes(pred_boxes, (im_shape_w,im_shape_h))
        
        IoU_with_gt_all = IoU_target(pred_boxes,unnormalized_bbox_targets)
        # nonzero argmax labels for background images will have wrong target boxes
        IoU_with_gt_all[np.where(labels == 0)[0]] = 0.5  
        IoU_with_gt_all = IoU_with_gt_all
        sim_images = self.sim_classes[(labels-1),:][:,(labels-1)]

        # Compute IoU, S, Phi, L
        IoU = pair_IoU(pred_boxes)
        S = np.multiply(IoU,sim_images) + self.epsilon * np.eye(M,M)
        Phi = np.multiply(IoU_with_gt_all,Phi) 
        L = np.reshape(np.repeat(Phi,M),(M,M))*S*np.reshape(np.tile(Phi,M),(M,M))
        det_L_I = np.linalg.det(L + np.eye(M))
        return IoU, S, L, IoU_with_gt_all, pred_boxes, det_L_I  

    def compute_log_p(self, Y, S_y, y, Phi, det_L_I, M, Phi_power):
        """
        log p(Y|Xy) = log det(L_Y) -log det( L+I)
        """
        
        if len(Y)==0:
            log_p=0
        else:
            log_p = 2*Phi_power * np.sum(np.multiply(y,np.reshape(np.log(Phi),(M,1)))) +np.log(np.linalg.det(S_y))-np.log(det_L_I)
        return log_p
    
    
    def Compute_Xy(self, Y, keeps_Y, labels, pred_boxes):
        """
        Find the set of proposals as Xy in p(Y|Xy)
        """
        dict_keeps_Y={}
        for i,j in keeps_Y.iteritems():
            if j not in dict_keeps_Y:
                dict_keeps_Y[j]=[]
            dict_keeps_Y[j].append(i)
        potential_bgs=[]
        for label_k in dict_keeps_Y.keys():
            idxs_k = np.where((labels)==label_k)[0]
            pbox = pred_boxes[idxs_k,:]
            pbox = np.reshape(np.tile(pbox,len(dict_keeps_Y[label_k])),(len(dict_keeps_Y[label_k])*len(idxs_k),4))
            Y_selected_pbox = pred_boxes[dict_keeps_Y[label_k],:]
            Y_selected_pbox = np.tile(Y_selected_pbox,(len(idxs_k),1))
            max_ol_ll = np.max(np.reshape(IoU_target(pbox,Y_selected_pbox),(len(idxs_k),len(dict_keeps_Y[label_k]))),1)
            potential_bgs.extend(list(idxs_k[np.where(max_ol_ll<cfg.TRAIN.IGNORANCE)[0]]))
        potential_bgs = list(set(potential_bgs))
        bgs = list(np.where((labels)==0)[0])

        Xy = np.array(list(Y)+bgs+potential_bgs)
        labels_Xy = np.array(list(labels[Xy[0:len(Y)]])+list(np.zeros((len(bgs)+len(potential_bgs))))).astype(int)
        return Xy, labels_Xy, potential_bgs
    
    def compute_diff_logp(self,labels_Xy, Xy, y, exp_cls_score, bbox_pred, boxes, Phi, unnormalized_bbox_targets, 
        im_shape_w, im_shape_h,Phi_power, normalizer, is_y):
        """
        Compute gradient of log p(Y|Xy)
        """
        M = bbox_pred.shape[0] # number of rois of 1 image in the minibatch
        K = exp_cls_score.shape[1] # number of categories
        loc_argmax = find_local_argmax(labels_Xy, Xy, bbox_pred)
        IoU, S, L, IoU_with_gt_all, pred_boxes, det_L_I  = self.compute_kernel(labels_Xy, boxes[Xy,:], Phi,
                                                     loc_argmax, unnormalized_bbox_targets[Xy,:], 
                                                     im_shape_w, im_shape_h)                        
        M_y = len(Xy) 
        Adj_L = np.linalg.inv(L + np.eye(M_y)) * det_L_I 
        diag_L = np.diag(L)
        Kii = np.reshape(np.repeat(diag_L,K),(M_y,K))/det_L_I
        Phi_frac = np.divide(exp_cls_score[Xy,:], np.reshape(np.repeat(np.sum(exp_cls_score[Xy,1:],1),K),(M_y,K))+0.0001)

        y_repeated = np.reshape(np.repeat(y,K),(M_y,K))
        if is_y:
            dLoss_dbi_cj = np.multiply(1-y_repeated, -2*Phi_power * np.multiply(Phi_frac, Kii))
            dLoss_dbi_cj[:,0] = 0
            dLoss_dbi_c = 2*Phi_power * np.multiply(y, (1-diag_L/det_L_I))
        else:
            dLoss_dbi_cj = np.multiply(y_repeated, 2*Phi_power* np.multiply(Phi_frac, 1-Kii))
            dLoss_dbi_cj[:,0] = 0
            dLoss_dbi_c = np.multiply(1-y, -2*Phi_power*diag_L/det_L_I)
        dLoss_db1 = np.zeros((M,K))        
        dLoss_db1[Xy,:] = dLoss_dbi_cj
        dLoss_db1[Xy,labels_Xy] = dLoss_dbi_c
        dLoss_db1 *= normalizer        
        return dLoss_db1

    def clip_grad(self,dLoss_db1, cls_score):
        """
        clip gradient above an specific threshold
        """
        bottom_diff_1_y =  1 * self._loss_weight * dLoss_db1
        max_relative_diff_1_y=np.max(np.max(np.abs(np.divide(bottom_diff_1_y,cls_score+0.000001))))
        MAX_RD_1 = 10
        if  max_relative_diff_1_y > MAX_RD_1:
            bottom_diff_1_y *= 1/max_relative_diff_1_y * MAX_RD_1
        return bottom_diff_1_y
        
    def vis_detections(self,imdb, im, labels, dets,Phi_argmax,scores,thresh=0.6,scores2=[]):
    
        """Visualize Detections"""
        with open(cfg.TRAIN.info, 'r') as fp:
            info = json.load(fp)
    
        if imdb == 'pascal_voc':
            classes = info['pascal_cats']

        elif imdb == 'coco':
            classes = info['coco_cats']
        im_=np.zeros((im.shape[1],im.shape[2],3))
        im_[:,:,0]=im[2,:,:]
        im_[:,:,1]=im[1,:,:]
        im_[:,:,2]=im[0,:,:]
        im=np.uint8((im_-np.min(im_))/np.max(im_-np.min(im_))*255)
        class_names = [classes[ll] for ll in labels]
        for i in xrange(np.minimum(10, dets.shape[0])):
            bbox = dets[i, :4]
            score = scores[i]
            if len(scores2)>0:
                score2 = scores2[i]
            else:
                score2 = 0
            class_name = class_names[i]
            class_phi = Phi_argmax[i]/4.0
            if score > thresh:            
                plt.cla()
                plt.imshow(im)
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='g', linewidth=3)
                    )
                plt.title('{}  {:.3f} {:.3f}'.format(class_name, score, score2))
                plt.show()
                
    def extract_im_per_batch(self,N_im_per_batch,i_image, data):
        """This functions is especially useful if N_im_per_batch >1"""
        if N_im_per_batch==1:
            im_shape_w=(data[i_image,:,:,:]).shape[1]
            im_shape_h=(data[i_image,:,:,:]).shape[2]
      
        else:
            zeros_data=np.nonzero(np.sum(np.sum((data[i_image,:,:,:]),0),1)==0)[0]
            if zeros_data.size: 
                diff_B4 = max(np.nonzero(abs(np.diff(zeros_data)-1)))
                diff_B4 = -1 if not diff_B4 else diff_B4[0]
                im_shape_w = zeros_data[diff_B4+1]                        
            else:
                im_shape_w = (data[i_image,:,:,:]).shape[1]                        

            zeros_data=np.nonzero(np.sum(np.sum((data[i_image,:,:,:]),0),0)==0)[0]
            if zeros_data.size: 
                diff_B4 = max(np.nonzero(abs(np.diff(zeros_data)-1)))
                diff_B4 = -1 if not diff_B4 else diff_B4[0]
                im_shape_h = zeros_data[diff_B4+1]                        
            else:
                im_shape_h = (data[i_image,:,:,:]).shape[2] 
        return im_shape_w,im_shape_h  

