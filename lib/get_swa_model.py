import os 
import torch
from argparse import ArgumentParser

def main():
	parser = ArgumentParser()
	parser.add_argument(
		'model_dir', help='the directory where checkpoints are saved')
	parser.add_argument(
		'start_epoch', type=int, help='the id of the starting checkpoint  for averaging')
	parser.add_argument(
		'end_epoch', type=int, help='the id of the ending checkpoint for averaging')
	parser.add_argument(
		'--save-dir', help='the directory to save the averaged model')

	args = parser.parse_args()
	model_dir = args.model_dir
	epoch_ids = list(range(args.start_epoch, args.end_epoch+1))
	model_names = [os.path.join(model_dir, 'CSP-{}.pth'.format(i)) for i in epoch_ids]
	model_checkpoints = [torch.load(model_name) for model_name in model_names]
	num = len(model_names)
	save_model = model_checkpoints[0]
	swa_checkpoints = model_checkpoints[0]['model'].copy()
	keys = swa_checkpoints.keys()
	
	for key in keys:
		sum_weights = 0.0
		for model_checkpoint in model_checkpoints:
			sum_weights += model_checkpoint['model'][key]
		average_weights = sum_weights / num
		swa_checkpoints[key] = average_weights
	save_model['model'] = swa_checkpoints

	torch.save(save_model, args.save_dir)
	print('Model is saved at ', args.save_dir)

if __name__ == '__main__':
	main()




