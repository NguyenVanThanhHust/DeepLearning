import os
import shutil

if __name__ == "__main__":
	list_gt = next(os.walk("dataset/gt"))[2]
	list_img = next(os.walk("dataset/img"))[2]
	new_list_gt = [x[:-9] for x in list_gt]
	new_list_img = [x[:-4] for x in list_img]
	for name in new_list_img:
		if name not in new_list_gt:
			print(name)
			os.remove(os.path.join("dataset/img", name + '.jpg'))