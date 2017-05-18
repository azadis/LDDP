# --------------------------------------------------------
# LDDP
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Samaneh Azadi
# --------------------------------------------------------

import caffe
import numpy as np
import math
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes,bbox_transform
import pickle
from fast_rcnn.config import cfg
from boxTools import *
from dppTools import DPP
            
        

class DPPLossLayer(caffe.Layer):
    """
    Compute the DPP Loss to apply diversity among detected boxes
    """
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 9:
             raise Exception("Need nine inputs to apply diversity by DPP.")
        self.sim_classes=[]
        self.imdb_name = cfg.TRAIN.IMDB

    def reshape(self, bottom, top):
        # loss output is scalar
        top[0].reshape(1)
        
        
    def forward(self, bottom, top):
        """
        Forward Pass
        """
        self._loss_weight = top[0].diff[0]
        # to make S as a PSD matrix:
        self.epsilon = 0.02 
        sim_power = cfg.TRAIN.SIM_POWER
        Phi_power = 0.5
        self.max_per_image = 100
        self.min_Phi=0.0001        
        
        gt_boxes= bottom[2].data
        num_gt_per_img = gt_boxes.shape[0]
        rois = bottom[3].data
        data = bottom[4].data
        N_im_per_batch = data.shape[0]

        self._sample_Y = ["" for i in range(N_im_per_batch)]
        self._keeps_Y = ["" for i in range(N_im_per_batch)]
        self._sample_B = ["" for i in range(N_im_per_batch)]
        self._keeps_B = ["" for i in range(N_im_per_batch)]
        self._Xy = ["" for i in range(N_im_per_batch)]
        self._labels_Xy = ["" for i in range(N_im_per_batch)]
        self._Xb = ["" for i in range(N_im_per_batch)]
        self._labels_Xb = ["" for i in range(N_im_per_batch)]

        
        self.sim_classes = pickle.load(open(cfg.TRAIN.similarity_path,"r"))
        K = bottom[1].data.shape[1] # number of categories
        self.means = np.reshape(bottom[7].data,(K,4))
        self.stds = np.reshape(bottom[8].data,(K,4))
        self.stds[0,:] = np.ones(((self.stds).shape[1],))

        
        self.sim_classes = self.sim_classes**sim_power
        DPP_ = DPP(stds=self.stds,means=self.means,sim_classes=self.sim_classes,epsilon=self.epsilon,loss_weight=self._loss_weight)
        for i_image in range(N_im_per_batch):
            batch = range(min(np.nonzero(rois[:,0] ==i_image)[0]), max(np.nonzero(rois[:,0] ==i_image)[0]))
            
            im_shape_w,im_shape_h = DPP_.extract_im_per_batch( N_im_per_batch, i_image, data)      

            bbox_pred = np.array((bottom[0].data[batch,:]))
            cls_score = bottom[1].data[batch,:]
            bbox_targets = bottom[6].data[batch,:]
            labels = bottom[5].data[batch].astype(int)
            M = bbox_pred.shape[0] # number of rois of 1 image in the minibatch
            max_cls_score = np.reshape(np.repeat(np.max((cls_score),1),K),(M,K))
            exp_cls_score = np.exp(cls_score - max_cls_score)
            
            boxes = (rois[batch,:])[:,1:]
            
            # ========================================================= 
            # Y: maximize prob of selecting gt boxes
            # ========================================================= 
            
            unnormalized_bbox_targets = unnormalize_box(labels, bbox_targets, boxes, self.stds, self.means, M, im_shape_w, im_shape_h)
            loc_argmax = find_local_argmax(labels, range(M), bbox_pred)
            Phi = exp_cls_score[range(M),labels] #gt label to be considered as phi_i
            Phi = np.maximum(Phi , self.min_Phi) 
            Phi = Phi ** Phi_power
            IoU, S, L, IoU_with_gt_all, pred_boxes, det_L_I = DPP_.compute_kernel(labels, boxes, Phi, 
                                                        loc_argmax, unnormalized_bbox_targets, 
                                                        im_shape_w, im_shape_h)

            # ========================================================= 
            # find Y with MAP :
            # non background images considered only
            # ignore prediction scores; only IoU for measuring quality
            # find non background images based on their labels: label=0 => bg
            # only consider boxes with high overlap with a non-bg ground-truth box
            # ========================================================= 

            MAP_images = np.nonzero(labels)[0]
            MAP_labels = labels[MAP_images]
            M_MAP = len(MAP_images)
            log_p_Y=[]
            keeps_Y=[]
            
            IoU_with_gt_all_MAP = IoU_with_gt_all[MAP_images]
            good_gt_overlap = np.where(IoU_with_gt_all_MAP > (cfg.TRAIN.IoU_gt_thresh))[0]
            among_ims = MAP_images[good_gt_overlap]

            y = np.zeros((M,1))
            if among_ims.shape[0] == 0:                
                log_p_Y.append(0)
                keeps_Y.append({})
                Xy=[]
                Y=np.array([])
                labels_Xy=[]
            else:
                S_MAP = S[MAP_images,:][:,MAP_images]
                Phi_MAP = np.multiply(IoU_with_gt_all_MAP,np.ones((M_MAP,)))
                # =======================================================
                # select representative boxes by MAP inference
                # =======================================================
                
                selected_and_probs = DPP_.dpp_greedy(S_MAP, Phi_MAP, 1, self.max_per_image, among_ims, 
                                            num_gt_per_img=num_gt_per_img)
                Y = np.array(selected_and_probs[0])
                prob_dpp = selected_and_probs[1]
                
                Y = among_ims[np.reshape(Y,(Y.shape[0],)).tolist()]
                keeps_Y.append(dict(zip(Y,MAP_labels[Y])))
                y[Y] = 1
                y = np.reshape(y, (M,))

                # =======================================================
                # Find X in P(Y|X)
                # =======================================================
                
                Xy, labels_Xy, potential_bgs = DPP_.Compute_Xy(list(Y), keeps_Y[i_image], labels,
                                                               pred_boxes)
                L = L[Xy,:][:,Xy]
                det_L_I = np.linalg.det(L + np.eye(len(Xy)))
                S_y = S[Y,:][:,Y]
                log_p = DPP_.compute_log_p(Y, S_y, y, Phi, det_L_I, M, Phi_power)
                log_p_Y.append(log_p)

            self._sample_Y[i_image] = y
            self._keeps_Y[i_image] = keeps_Y
            self._Xy[i_image] = Xy
            self._labels_Xy[i_image] = labels_Xy

 
            # ========================================================= 
            # B: minimize prob of selecting background boxes
            # ========================================================= 
            log_p_B=[]
            keeps_B=[]

            Phi_labels = np.argmax(exp_cls_score,axis=1)
            Phi = exp_cls_score[range(M),Phi_labels] #gt label to be considered as phi_i
            Phi = np.maximum(Phi , self.min_Phi) 
            Phi = Phi ** Phi_power
            loc_argmax = find_local_argmax(Phi_labels, range(M), bbox_pred)
            IoU, S, L, IoU_with_gt_all, pred_boxes, det_L_I = DPP_.compute_kernel(Phi_labels, boxes, Phi, 
                                                loc_argmax, unnormalized_bbox_targets, im_shape_w, im_shape_h)

            good_gt_overlap = np.where(IoU_with_gt_all > (cfg.TRAIN.IoU_gt_thresh))[0] 
            B, prob_dpp = DPP_.select_bg(Phi_labels,boxes,labels,bbox_pred,keeps_Y[i_image],good_gt_overlap,
                                         M,im_shape_w,im_shape_h)
            b = np.zeros((M,1))
            b[np.reshape(B,(B.shape[0],)).tolist()] = 1
            b = np.reshape(b, (M,))

            bgs = list(np.where((labels)==0)[0])
            bgs_1=sorted(set(bgs)-set(list(Y)+list(B)))
            Xb = np.array(list(Y)+list(B)+bgs_1)
            labels_Xb = np.array(list(labels[list(Y)])+list(np.zeros((len(B)+len(bgs_1),1)))).astype(int)
            keeps_B.append(dict(zip(B,Phi_labels[B])))
            
            L = L[Xb,:][:,Xb]
            det_L_I = np.linalg.det(L + np.eye(len(Xb)))
            S_b = S[B,:][:,B]
            log_p = DPP_.compute_log_p(B, S_b, b, Phi, det_L_I, M, Phi_power)
            log_p_B.append(log_p)


            self._sample_B[i_image] = b
            self._keeps_B[i_image] = keeps_B
            self._Xb[i_image] = Xb
            self._labels_Xb[i_image] = labels_Xb
            
            normalizer_Y = (len(B)+1)/np.float(len(list(Y)+list(B))+1)
            normalizer_B = (len(Y)+1)/np.float(len(list(Y)+list(B))+1)
            
            
        top[0].data[...] = -normalizer_Y*sum(log_p_Y)+normalizer_B*sum(log_p_B)



    def backward(self, top, propagate_down, bottom):
        """
        Backward Pass
        """

        Phi_power = 0.5
    
        cls_score_diff = np.zeros(bottom[1].data.shape)
        rois = bottom[3].data
        data = bottom[4].data
        gt_boxes= bottom[2].data
        num_gt_per_img = gt_boxes.shape[0]
        N_im_per_batch = data.shape[0]
        
        for i_image in range(N_im_per_batch):
            batch = range(min(np.nonzero(rois[:,0] ==i_image)[0]), max(np.nonzero(rois[:,0] ==i_image)[0]))
            im_shape_w,im_shape_h = DPP().extract_im_per_batch( N_im_per_batch, i_image, data)      
                     
            bbox_pred = (bottom[0].data[batch,:])        
            cls_score = (bottom[1].data[batch,:])
            bbox_targets = (bottom[6].data[batch,:])
            labels = bottom[5].data[batch].astype(int)
            M = bbox_pred.shape[0] # number of rois of 1 image in the minibatch
            K = cls_score.shape[1] # number of categories
            
            boxes = (rois[batch,:])[:,1:]
            dLoss_db1 = np.zeros((M,K))

            max_cls_score = np.reshape(np.repeat(np.max((cls_score),1),K),(M,K))
            exp_cls_score = np.exp(cls_score - max_cls_score)

            Phi_labels = labels
            unnormalized_bbox_targets = unnormalize_box(Phi_labels, bbox_targets, boxes,self.stds,self.means,
                                                                                 M, im_shape_w, im_shape_h)

            sim_classes_0 = np.zeros((K,K))
            sim_classes_0[1:K,1:K] = self.sim_classes #include sims for class 0
            sim_classes_0[0,0]=1
            self.sim_classes = sim_classes_0
            DPP_ = DPP(stds=self.stds,means=self.means,sim_classes=self.sim_classes,epsilon=self.epsilon,loss_weight=self._loss_weight)
            
            # ========================================================= 
            # d logp(Y|Xy)/db_i^c
            # ========================================================= 
            
            dLoss_db1 = np.zeros((M,K))
            B=sorted(self._keeps_B[i_image][0].keys())
            Y = sorted(self._keeps_Y[i_image][0].keys())

            if len(self._keeps_Y[i_image][0].keys()) > 0:

                Xy = self._Xy[i_image]
                labels_Xy = self._labels_Xy[i_image]
                y = self._sample_Y[i_image][Xy]

                Phi = np.multiply(y, exp_cls_score[Xy,labels_Xy]) + np.multiply(1-y, np.sum(exp_cls_score[Xy,1:],1))
                Phi = Phi**Phi_power
                normalizer = (len(B)+1)/np.float(len(list(Y)+list(B))+1) 

                dLoss_db1 = DPP_.compute_diff_logp(labels_Xy, Xy, y, exp_cls_score, bbox_pred, boxes, Phi, unnormalized_bbox_targets, 
                        im_shape_w, im_shape_h, Phi_power, normalizer, True)
            bottom_diff_1_y = DPP_.clip_grad(dLoss_db1, cls_score)

            # ========================================================= 
            # d logp(B|Xb)/db_i^c
            # =========================================================                 

            dLoss_db1 = np.zeros((M,K))
            if len(self._keeps_B[i_image][0].keys()) > 0:
                Xb = self._Xb[i_image]
                labels_Xb = self._labels_Xb[i_image]
                labels_Xb = np.reshape(labels_Xb,(labels_Xb.shape[0],))
                b = self._sample_B[i_image][Xb]

                Phi = (np.multiply(b, np.sum(exp_cls_score[Xb,1:],1)) +np.multiply(1-b,exp_cls_score[Xb,labels_Xb] ))
                Phi = Phi**Phi_power
                normalizer = (len(Y)+1)/np.float(len(list(Y)+list(B))+1)
                dLoss_db1 = DPP_.compute_diff_logp(labels_Xb, Xb, b, exp_cls_score, bbox_pred, boxes, Phi, unnormalized_bbox_targets, 
                        im_shape_w, im_shape_h, Phi_power, normalizer, False)

            bottom_diff_1_b = DPP_.clip_grad(dLoss_db1, cls_score)

            cls_score_diff[batch,:] = bottom_diff_1_y - bottom_diff_1_b
            
        bottom[1].diff[...] = -cls_score_diff 
