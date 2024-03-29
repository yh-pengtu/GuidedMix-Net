# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import collections
import copy, os.path as osp
from itertools import chain

import mmcv, glob
import numpy as np, math
from mmcv.utils import build_from_cfg, print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS, PIPELINES
from .voc import SemiPascalVOCDataset
from .cityscapes import CityscapesDataset, SemiCityscapesDataset

@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    support evaluation and formatting results

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the concatenated
            dataset results separately, Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE
        self.separate_eval = separate_eval
        assert separate_eval in [True, False], \
            f'separate_eval can only be True or False,' \
            f'but get {separate_eval}'
        if any([isinstance(ds, CityscapesDataset) for ds in datasets]):
            raise NotImplementedError(
                'Evaluating ConcatDataset containing CityscapesDataset'
                'is not supported!')

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]]): per image
                pre_eval results or predict segmentation map for
                computing evaluation metric.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: evaluate results of the total dataset
                or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.img_dir} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results

        if len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types when '
                'self.separate_eval=False')
        else:
            if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                    results, str):
                # merge the generators of gt_seg_maps
                gt_seg_maps = chain(
                    *[dataset.get_gt_seg_maps() for dataset in self.datasets])
            else:
                # if the results are `pre_eval` results,
                # we do not need gt_seg_maps to evaluate
                gt_seg_maps = None
            eval_results = self.datasets[0].evaluate(
                results, gt_seg_maps=gt_seg_maps, logger=logger, **kwargs)
            return eval_results

    def get_dataset_idx_and_sample_idx(self, indice):
        """Return dataset and sample index when given an indice of
        ConcatDataset.

        Args:
            indice (int): indice of sample in ConcatDataset

        Returns:
            int: the index of sub dataset the sample belong to
            int: the index of sample in its corresponding subset
        """
        if indice < 0:
            if -indice > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            indice = len(self) + indice
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, indice)
        if dataset_idx == 0:
            sample_idx = indice
        else:
            sample_idx = indice - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """format result for every sample of ConcatDataset."""
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        ret_res = []
        for i, indice in enumerate(indices):
            dataset_idx, sample_idx = self.get_dataset_idx_and_sample_idx(
                indice)
            res = self.datasets[dataset_idx].format_results(
                [results[i]],
                imgfile_prefix + f'/{dataset_idx}',
                indices=[sample_idx],
                **kwargs)
            ret_res.append(res)
        return sum(ret_res, [])

    def pre_eval(self, preds, indices):
        """do pre eval for every sample of ConcatDataset."""
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]
        ret_res = []
        for i, indice in enumerate(indices):
            dataset_idx, sample_idx = self.get_dataset_idx_and_sample_idx(
                indice)
            res = self.datasets[dataset_idx].pre_eval(preds[i], sample_idx)
            ret_res.append(res)
        return sum(ret_res, [])


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return self.times * self._ori_len


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process.


    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self, dataset, pipeline, skip_type_keys=None):
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self.num_samples = len(dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indexes
                ]
                results['mix_results'] = mix_results

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys

@DATASETS.register_module()
class SemiDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process.


    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self, dataset, unsup_dataset, pipeline=None, skip_type_keys=None):
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        if pipeline is None:
            self.pipeline = pipeline
        else:
            self.pipeline = []
            self.pipeline_types = []
            for transform in pipeline:
                if isinstance(transform, dict):
                    self.pipeline_types.append(transform['type'])
                    transform = build_from_cfg(transform, PIPELINES)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')

        self.sup_img_infos = dataset.img_infos

        self.unsup_dataset = unsup_dataset
        self.unsup_img_infos = unsup_dataset.img_infos
        
        self.dataset = dataset        
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self.num_samples = len(dataset)
        
        self.__gen_sample_data_infos()
        self._progress_in_iter = 1

    def __gen_sample_data_infos(self):
        self.num_samples = max(len(self.dataset), len(self.unsup_img_infos))
        if len(self.dataset) == self.num_samples:
            self._sup_img_infos = self.sup_img_infos
        else:
            repeat_times = math.ceil(self.num_samples / len(self.sup_img_infos))
            indices = np.random.permutation(np.arange(len(self.sup_img_infos)))

            temp = [self.sup_img_infos[idx] for idx in indices]
            self._sup_img_infos = (self.sup_img_infos + temp * repeat_times)[:self.num_samples]

        if len(self.unsup_img_infos) == self.num_samples:
            self._unsup_img_infos = self.unsup_img_infos
        else:
            repeat_times = math.ceil(self.num_samples / len(self.unsup_img_infos))

            indices = np.random.permutation(np.arange(len(self.unsup_img_infos)))
            temp = [self.unsup_img_infos[idx] for idx in indices]
            self._unsup_img_infos = (self.unsup_img_infos + temp * repeat_times)[:self.num_samples]
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_info = self._sup_img_infos[idx]
        ann_info = img_info['ann']
        results = dict(img_info=img_info, ann_info=ann_info)
        
        results['seg_fields'] = []
        results['img_prefix'] = self.dataset.img_dir
        results['seg_prefix'] = self.dataset.ann_dir
        if self.dataset.custom_classes:
            results['label_map'] = self.dataset.label_map
        
        segmap_info = glob.glob(osp.join(results['seg_prefix'], results['ann_info']['seg_map']))
        assert len(segmap_info) > 0
        
        # results['seg_prefix'] =  '/'.join(segmap_info[0].split('/')[:-1]), for pascal voc
        if isinstance(self.dataset, SemiCityscapesDataset):
            results['seg_prefix'] =  '/'.join(segmap_info[0].split('/')[:-2])
        elif isinstance(self.dataset, SemiPascalVOCDataset):
            results['seg_prefix'] =  '/'.join(segmap_info[0].split('/')[:-1])
            
        results = self.dataset.pipeline(results)
        
        ## results of unsup dataset
        unsup_img_info = self._unsup_img_infos[idx]
        unsup_ann_info = unsup_img_info['ann']
        unsup_results = dict(img_info=unsup_img_info, ann_info=unsup_ann_info)
        
        unsup_results['seg_fields'] = []
        unsup_results['img_prefix'] = self.unsup_dataset.img_dir
        unsup_results['seg_prefix'] = self.unsup_dataset.ann_dir
        if self.unsup_dataset.custom_classes:
            unsup_results['label_map'] = self.unsup_dataset.label_map
        
        unsup_segmap_info = glob.glob(osp.join(unsup_results['seg_prefix'], unsup_results['ann_info']['seg_map']))
        assert len(unsup_segmap_info) > 0
        
        if isinstance(self.dataset, SemiCityscapesDataset):
            unsup_results['seg_prefix'] =  '/'.join(unsup_segmap_info[0].split('/')[:-2])
        elif isinstance(self.dataset, SemiPascalVOCDataset):
            unsup_results['seg_prefix'] =  '/'.join(unsup_segmap_info[0].split('/')[:-1])
            
        unsup_results = self.unsup_dataset.pipeline(unsup_results)
        unsup_results = {'unsup.'+k: v for k, v in unsup_results.items()}
        
        results.update(unsup_results)
        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys