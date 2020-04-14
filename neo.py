import numpy as np
import os
from time import perf_counter
import cv2
from dlr import DLRModel
import argparse
#from PIL import Image
import io


def read_image(image):
	img = cv2.imread(image)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (608, 608), interpolation=cv2.INTER_LANCZOS4)
	img = np.asarray(img)
	img = np.rollaxis(img, axis=2, start=0)[np.newaxis,:]
	img = img.astype(np.float32)	
	return img

class_map = None
parser = argparse.ArgumentParser(description="Xavier Work")
parser.add_argument('--classmap', type=str, help="Name of the classmap to use, options are currently 'VOC' or 'CARDBOT'")
parser.add_argument('--modelpath', type=str, help="Path to the model files")
args = parser.parse_args()


dlr_model = DLRModel(model_path=args.modelpath, dev_type='gpu')
val_path = 'observations'
out_path = 'results'

cardbot_map = ['AH', 'KH', 'QH', 'JH', '10H', '9H', '8H', '7H', '6H', '5H', '4H', '3H', '2H', 'AD', 'KD', 'QD', 'JD',
			   '10D', '9D', '8D', '7D', '6D', '5D', '4D', '3D', '2D', 'AC', 'KC', 'QC', 'JC', '10C', '9C', '8C', '7C',
			   '6C', '5C', '4C', '3C', '2C', 'AS', 'KS', 'QS', 'JS', '10S', '9S', '8S', '7S', '6S', '5S', '4S', '3S',
			   '2S']
voc_map = ["aeroplane","bicycle","bird","boat","bottle","bus","car", "cat","chair","cow","diningtable","dog","horse",
    "motorbike","person", "pottedplant","sheep","sofa","train","tvmonitor"]

if args.classmap == 'VOC':
	class_map = voc_map
else:
	class_map = cardbot_map




def main():
	results_txt = open(os.path.join(out_path, 'results.txt'), 'w')
	for image_filename in os.listdir(val_path):
		result_txt = 'Image ' + image_filename + '\n'
		image_file = os.path.join(val_path, image_filename)
		image = read_image(image_file)
		input_data = {'data':image}
		start_time = perf_counter()
		out = dlr_model.run(input_data)
		elapsed_time = perf_counter() - start_time
		objects = out[0][0]
		scores = out[1][0]
		bboxes = out[2][0]
		result_txt += 'Elapsed time: {0}\n'.format(elapsed_time)
		result_txt += 'Classes above 20% score threshold:\n'
		for index, score in enumerate(scores, start=0):
			if score[0] > 0.20:
				object_id = int(objects[index][0])
				result_txt += '  {0}: {1}\n'.format(class_map[object_id], score[0])
				result_txt += '\n'
				bbox = bboxes[index][0]
				result_txt += '{}'.format(bbox)
				result_txt += '\n'
				results_txt.write(result_txt)

	results_txt.close()
	print('Done')


if __name__ == '__main__':
	main()