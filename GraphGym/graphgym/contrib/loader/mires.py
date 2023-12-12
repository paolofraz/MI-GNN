import os

from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph
from graphgym.register import register_loader
from graphgym.config import cfg
from torch_geometric.transforms import BaseTransform

# from torch_geometric.datasets import TUDataset

import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data


class MyTUDataset(InMemoryDataset):

	def __init__(self, root: str, name: Optional[str] = None,
				 transform: Optional[Callable] = None,
				 pre_transform: Optional[Callable] = None,
				 pre_filter: Optional[Callable] = None):
		self.name = name if name else root.split('_')[-1]
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

		self.data['x_original'] = self.data['x']
		self.data['x'] = self.data['reservoir']

		torch.save((self.data.to_dict(), self.slices, sizes),
				   osp.join(self.processed_dir, self.processed_file_names))

	def __repr__(self) -> str:
		return f'{self.name}({len(self)})'


def load_dataset_mires(format, name, dataset_dir):
	# dataset_dir = '{}/{}'.format(dataset_dir, name)
	if format == 'mires':
		dataset_dir = f"{dataset_dir}/run_{cfg.seed - 1}_RES_{cfg.mires.RES}_k_{cfg.mires.k}_delta_{cfg.mires.delta}_nunits_{cfg.mires.nunits}_{name}"
		if os.path.exists(dataset_dir):
			dataset_raw = MyTUDataset(dataset_dir, name)
			# graphs = [
			#		Graph.pyg_to_graph(
			#				dataset_raw._data, verbose=False,
			#				tensor_backend=False,
			#				netlib=None
			#				)
			#		]
			graphs = GraphDataset.pyg_to_graphs(dataset_raw)
			print("Loaded dataset {}".format(dataset_dir))
			return graphs
		else:
			raise ValueError("Could not load dataset from {}".format(dataset_dir))


register_loader('mires', load_dataset_mires)


# Reservoir
def load_dataset_res(format, name, dataset_dir):
	# dataset_dir = '{}/{}'.format(dataset_dir, name)
	if format == 'res':
		dataset_dir = f"{dataset_dir}/run_{cfg.seed - 1}_TANH_RES_{cfg.mires.RES}_{cfg.mires.k}_n_units_{cfg.mires.nunits}_{name}"
		if os.path.exists(dataset_dir):
			dataset_raw = MyTUDataset(dataset_dir, name)
			# graphs = [
			#		Graph.pyg_to_graph(
			#				dataset_raw._data, verbose=False,
			#				tensor_backend=False,
			#				netlib=None
			#				)
			#		]
			graphs = GraphDataset.pyg_to_graphs(dataset_raw)
			print("Loaded dataset {}".format(dataset_dir))
			return graphs
		else:
			raise ValueError("Could not load dataset from {}".format(dataset_dir))


register_loader('res', load_dataset_res)


def transform_mimr(data):
	data["x"] = data["svd_reservoir_mr_mi"]
	return data


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

	#		self.data["x"] = self.data["svd_reservoir_mr_mi"]

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
				self.data[var] = self.data[var][:, :max_rank]

			torch.save((self.data.to_dict(), self.slices, sizes),
					   osp.join(self.processed_dir, self.processed_file_names))
		except Exception as e:
			tqdm.write(f"File {self.root_var} can't be saved: {e}")

	def __repr__(self) -> str:
		return f'{self.name}({len(self)})'


# SVD
def load_dataset_svd(format, name, dataset_dir):
	# dataset_dir = '{}/{}'.format(dataset_dir, name)
	if 'svd' in format:
		dataset_dir = f"{dataset_dir}run_{cfg.seed - 1}_RES_{cfg.mires.RES}_k_{cfg.mires.k}_delta_{cfg.mires.delta}_nunits_{cfg.mires.nunits}_{name}"
		if os.path.exists(dataset_dir):
			dataset_raw = MyRESDataset(dataset_dir, transform=transform_mimr)
			# dataset_raw.data['x'] = dataset_raw.data["svd_reservoir_mr_mi"]
			# graphs = [
			#		Graph.pyg_to_graph(
			#				dataset_raw._data, verbose=False,
			#				tensor_backend=False,
			#				netlib=None
			#				)
			#		]
			graphs = GraphDataset.pyg_to_graphs(dataset_raw)
			print("Loaded dataset {}".format(dataset_dir))
			return graphs
		else:
			raise ValueError("Could not load dataset from {}".format(dataset_dir))


register_loader('svd', load_dataset_svd)