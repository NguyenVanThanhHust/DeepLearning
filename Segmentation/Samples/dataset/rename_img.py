import os

directory = 'gt'
list_img = next(os.walk(directory))[2]
for img_name in list_img:
	old_path = os.path.join(directory, img_name)
	new_name = img_name[:-9] + '.png'
	new_path = os.path.join(directory, new_name)
	os.rename(old_path, new_path)

