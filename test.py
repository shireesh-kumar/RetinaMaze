import torch
from model import build_unet
from glob import glob
import config
import numpy as np
from metrics import calculate_metrics
import cv2
from operator import add


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (256, 256, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (256, 256, 3)
    return mask


if __name__ == "__main__":
    print("Testing Started")
    H = config.H 
    W = config.W 
    size = (W, H)
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used : {device}")
    
    try:
        unet_model = build_unet()
        unet_model = unet_model.to(device)
        checkpoint = torch.load('/project/shah/shireesh/Personal_Projects/RetinalVesselSegmentation/checkpoints/best-checkpoint.ckpt')
        unet_model.load_state_dict(checkpoint['state_dict'])

        print("Model Loaded Successfully ")
    except :
        print("Error while loading the model")
        exit()

    unet_model.eval()

    test_x = sorted(glob(config.TEST_ORIGINAL_PATH + '/*.png'))
    test_y = sorted(glob(config.TEST_GROUND_PATH + '/*.png'))

    
    for i, (x, y) in enumerate(zip(test_x, test_y)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (256, 256, 3)
        image = cv2.resize(image, size,interpolation=cv2.INTER_AREA)
        x = np.transpose(image, (2, 0, 1))      ## (3, 256, 256)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 256, 256)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (256, 256)
        mask = cv2.resize(mask, size,interpolation=cv2.INTER_NEAREST)
        y = np.expand_dims(mask, axis=0)            ## (1, 256, 256)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 256, 256)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            pred_y = unet_model(x)
            pred_y = torch.sigmoid(pred_y)


            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()        ## (1, 256, 256)
            pred_y = np.squeeze(pred_y, axis=0)     ## (256, 256)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128        

        retinal_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cv2.imwrite(f"results/{name}.png", retinal_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    print("Testing Completed")
    
    