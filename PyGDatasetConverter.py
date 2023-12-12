import glob
import os
import os.path as osp
import resource
import shutil
from typing import Callable, List, Optional

import torch
import torch_geometric
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from torch_geometric.data import (
	Data,
	InMemoryDataset,
	download_url,
	extract_zip,
	)
from torch_geometric.io import read_tu_data

# Define the original root directory path
# root_dir = '/storage/lpasa/Dataset/MI_Reservoir_Dataset/'
# Baseline
# root_dir = '/storage/lpasa/Dataset/Reservoir_TANH_Dataset'
# SVD
root_dir = '/storage/lpasa/Dataset/MI_Reservoir_Dataset_SVD_2/'

# Define the new root directory path
# new_root_dir = '~/storage/Dataset/MI_Reservoir_Dataset/'
# new_root_dir = '~/storage/Dataset/Reservoir_TANH_Dataset/'
# new_root_dir = '~/storage/Dataset/MI_Reservoir_Dataset_SVD/'
new_root_dir = '~/storage/Dataset/MI_Dataset_SVD/'

# Expand the path relative to the home directory
new_root_dir = os.path.expanduser(new_root_dir)
os.makedirs(new_root_dir, exist_ok=True)

# Get a list of first-level subdirectories
subdirectories = [name for name in os.listdir(new_root_dir) if os.path.isdir(os.path.join(new_root_dir, name))]

# Recursively search for data.pt files in subdirectories
data_files = glob.glob(os.path.join(root_dir, '**', 'data.pt'), recursive=True)


###########
# SPECIFIC DATASET ONLY
# Create the search pattern using glob
#specific_dataset = "NCI1"
#search_pattern = f"{root_dir}/*{specific_dataset}*/{specific_dataset}/processed/data.pt"
# Find files matching the search pattern
#data_files = glob.glob(search_pattern)
#subdirectories = [dir for dir in subdirectories if specific_dataset in dir]
###########

###########
# DD ONLY - Baseline
# Create the search pattern using glob
# search_pattern = f"{root_dir}/*DD*/DD/processed/data.pt"
# Find files matching the search pattern
# data_files = glob.glob(search_pattern)
# subdirectories = [dir for dir in subdirectories if "DD" in dir]
###########

###########
# SHORTEST PATH
# Create the search pattern using glob
# search_pattern = f"{root_dir}/*_S_*/**/processed/data.pt"
# Find files matching the search pattern
# data_files = glob.glob(search_pattern)
# subdirectories = [dir for dir in subdirectories if "_S_" in dir]
###########


def save(data_file):
	# Rename data.pt files to data_old.pt and saves them accordingly.

	# Get the relative path of the data file with respect to the original root directory
	relative_path = os.path.relpath(data_file, root_dir)

	# Construct the new path by replacing the original root directory with the new root directory
	new_file_path = os.path.join(new_root_dir, relative_path)

	# Get dataset name for TANH Dataset
	if "TANH" in relative_path:
		name = relative_path.split('_')[-1]
		if name.startswith('MR'):
			name = 'PTC_' + name

		# Find the first occurrence of the repeated folder
		repeated_folder = name.split('/')[1]
		index_to_remove = name.find('/' + repeated_folder + '/')

		# Reconstruct the path by removing the first occurrence of the repeated folder
		result_path = name[:index_to_remove] + name[index_to_remove + len(repeated_folder) + 1:]

		new_file_path = os.path.join(new_root_dir, relative_path.split('/')[0],
									 result_path)  # already included in name: 'processed/data.pt' for TANH DATASET

	# Rename data.pt files to data_old.pt and saves them accordingly
	if os.path.exists(data_dict_file_path := os.path.join(os.path.dirname(new_file_path), 'data_dict.pt')):
		tqdm.write(f"File {data_dict_file_path} already exists")
		#os.remove(data_dict_file_path)
		return
	else:
		# Create the directory if it does not exist in the new root directory
		os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

		# Extract with PyG 1.7.*
		# print(new_file_path)
		data, slices = torch.load(data_file)
		torch.save((data.to_dict(), slices), data_dict_file_path)


# Rename the data file to data_old.pt
# new_file_path = os.path.splitext(new_file_path)[0] + '_old.pt'
# os.rename(data_file, new_file_path)


class MyTUDataset(InMemoryDataset):
	url = 'https://www.chrsmrrs.com/graphkerneldatasets'
	cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
				   'graph_datasets/master/datasets')

	def __init__(self, root: str,
				 transform: Optional[Callable] = None,
				 pre_transform: Optional[Callable] = None,
				 pre_filter: Optional[Callable] = None):
		self.name = root.split('_')[-1]
		self.root_var = root
		if self.name == 'MR':
			self.name = 'PTC_MR'
		super().__init__(root, transform, pre_transform, pre_filter)

		out = torch.load(self.processed_paths[0])
		if not isinstance(out, tuple) or len(out) != 3:
			raise RuntimeError(
					"The 'data' object was created by an older version of PyG. "
					"If this error occurred while loading an already existing "
					"dataset, remove the 'processed/' directory in the dataset's "
					"root folder and try again.")
		data, self.slices, self.sizes = out
		self.data = Data.from_dict(data) if isinstance(data, dict) else data

	@property
	def raw_dir(self) -> str:
		name = f'processed'
		return osp.join(self.root, self.name, name)

	@property
	def processed_dir(self) -> str:
		name = f'processed'
		return osp.join(self.root, self.name, name)

	@property
	def raw_file_names(self) -> str:
		return 'data_dict.pt'

	@property
	def processed_file_names(self) -> str:
		return 'data.pt'

	def process(self):
		data_dict, self.slices = torch.load(osp.join(self.processed_dir, self.raw_file_names))
		self.data = Data.from_dict(data_dict)

		# sizes = {
		#		'num_node_attributes': node_attributes.size(-1),
		#		'num_node_labels': node_labels.size(-1),
		#		'num_edge_attributes': edge_attributes.size(-1),
		#		'num_edge_labels': edge_labels.size(-1),
		#		}

		sizes = {}

		try:
			self.data['x_original'] = self.data['x']
			self.data['x'] = self.data['reservoir']

			torch.save((self.data.to_dict(), self.slices, sizes),
					   osp.join(self.processed_dir, self.processed_file_names))
		except Exception as e:
			tqdm.write(f"File {self.root_var} can't be saved: {e}")

	def __repr__(self) -> str:
		return f'{self.name}({len(self)})'

class MyRESDataset(InMemoryDataset):
	# dict_keys(['edge_attr', 'edge_index', 'svd_reservoir_mi_only', 'svd_reservoir_mi_only_rank', 'svd_reservoir_mr_mi', 'svd_reservoir_mr_mi_rank', 'svd_reservoir_mr_only', 'svd_reservoir_mr_only_rank', 'x', 'y'])
	url = 'https://www.chrsmrrs.com/graphkerneldatasets'
	cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
				   'graph_datasets/master/datasets')

	def __init__(self, root: str,
				 transform: Optional[Callable] = None,
				 pre_transform: Optional[Callable] = None,
				 pre_filter: Optional[Callable] = None):
		self.name = root.split('_')[-1]
		self.root_var = root
		if self.name == 'MR':
			self.name = 'PTC_MR'
		super().__init__(root, transform, pre_transform, pre_filter)

		out = torch.load(self.processed_paths[0])
		if not isinstance(out, tuple) or len(out) != 3:
			raise RuntimeError(
					"The 'data' object was created by an older version of PyG. "
					"If this error occurred while loading an already existing "
					"dataset, remove the 'processed/' directory in the dataset's "
					"root folder and try again.")
		data, self.slices, self.sizes = out
		self.data = Data.from_dict(data) if isinstance(data, dict) else data

	@property
	def raw_dir(self) -> str:
		name = f'processed'
		return osp.join(self.root, self.name, name)

	@property
	def processed_dir(self) -> str:
		name = f'processed'
		return osp.join(self.root, self.name, name)

	@property
	def raw_file_names(self) -> str:
		return 'data_dict.pt'

	@property
	def processed_file_names(self) -> str:
		return 'data.pt'

	def process(self):
		data_dict, self.slices = torch.load(osp.join(self.processed_dir, self.raw_file_names))
		self.data = Data.from_dict(data_dict)

		# sizes = {
		#		'num_node_attributes': node_attributes.size(-1),
		#		'num_node_labels': node_labels.size(-1),
		#		'num_edge_attributes': edge_attributes.size(-1),
		#		'num_edge_labels': edge_labels.size(-1),
		#		}

		sizes = {}

		try:
			for var in ['svd_reservoir_mi_only', 'svd_reservoir_mr_mi', 'svd_reservoir_mr_only']:
				max_rank = self.data[f"{var}_rank"].max().item()
				self._data[var] = self.data[var][:, :max_rank]

			torch.save((self.data.to_dict(), self.slices, sizes),
					   osp.join(self.processed_dir, self.processed_file_names))
		except Exception as e:
			tqdm.write(f"File {self.root_var} can't be saved: {e}")

	def __repr__(self) -> str:
		return f'{self.name}({len(self)})'


if __name__ == '__main__':
	conda_prefix = os.environ.get('CONDA_PREFIX')
	if conda_prefix:
		env_name = os.path.basename(conda_prefix)
		print(f"Conda environment: {env_name}")
		if env_name == 'PyG17':
			# First convert from PyG17 (conda activate PyG17)
			print("Convert from PyG17...")
			process_map(save, data_files, max_workers=8, chunksize=1)
		elif env_name == 'PyMIGNN':
			# Then save as PyG (conda activate PyMIGNN)
			# ds = MyTUDataset(root='/home/frazzetp/storage/Dataset/MI_Reservoir_Dataset/run_0_RES_A_k_3_delta_1_nunits_100_ENZYMES')
			def save_myTUDataset():
				print("Loading TUDatasets - RES - ...")
				for dir in tqdm(subdirectories):
					try:
						MyRESDataset(root=os.path.join(new_root_dir, dir))
					except RuntimeError as e:
						# Get Dataset by indexing
						dataset = dir.rfind("_")
						if dataset != -1:
							substring = dir[dataset + 1:]
						data_pt_file = os.path.join(new_root_dir, dir, f'{substring}/processed/data.pt')
						tqdm.write(f"Error: {e} \n File: {dir} \n Deleting {data_pt_file}...")
						os.remove(data_pt_file)
						MyRESDataset(root=os.path.join(new_root_dir, dir))
					except Exception as e:
						tqdm.write(f"Error: {e} \n File: {dir}")

			save_myTUDataset()
		else:
			raise Exception("Conda environment name needs to be either 'PyMIGNN' or 'PyG17'")
	else:
		raise Exception("Not using Conda environment")

	# process_map(MyTUDataset, [os.path.join(new_root_dir, dir) for dir in subdirectories], max_workers=8, chunksize=1)

	# '/home/frazzetp/storage/Dataset/Reservoir_TANH_Dataset/run_0_TANH_RES_L_3_n_units_30_ENZYMES/ENZYMES/processed/data_dict.pt'
# '/home/lpasa/Dataset/Reservoir_TANH_Dataset/run_0_TANH_RES_L_3_n_units_30_ENZYMES/processed/data.pt'