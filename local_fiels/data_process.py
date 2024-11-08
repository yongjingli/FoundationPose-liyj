import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_2_img():
    root = "/home/pxn-lyj/Egolee/data/test/pose_shi"
    in_root = os.path.join(root, "imgs")
    img_root = os.path.join(root, "colors")

    img_names = [name for name in os.listdir(in_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    for img_name in img_names:
        in_img_path = os.path.join(in_root, img_name)
        out_img_path = os.path.join(img_root, img_name.split(".")[0] + "_color.jpg")
        img = cv2.imread(in_img_path)
        cv2.imwrite(out_img_path, img)


def save_mask():
    root = "/home/pxn-lyj/Egolee/data/test/pose_shi"
    img_root = os.path.join(root, "colors")
    in_root = os.path.join(root, "mask")
    mask_root = os.path.join(root, "masks_num")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    for img_name in img_names:
        ing_mask_path = os.path.join(in_root, img_name.replace("_color.jpg", ".png"))
        img_path = os.path.join(img_root, img_name)
        out_mask_path = os.path.join(mask_root, img_name.replace("_color.jpg", "_mask.npy"))
        img = cv2.imread(img_path)

        if not os.path.exists(ing_mask_path):
            print(img_name)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            mask = cv2.imread(ing_mask_path, 0)
            mask = mask > 0
            mask = mask.astype(np.uint8)

        np.save(out_mask_path, mask)
        #
        # plt.imshow(mask)
        # plt.show()
        # # print(np.unique(mask))
        # #
        # exit(1)


if __name__ == "__main__":
    print("Start")
    # save_2_img()
    save_mask()
    print("End")