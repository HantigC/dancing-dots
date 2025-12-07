from abc import ABC, abstractmethod
from copy import deepcopy
import os
from time import sleep, time

import h5py
import torch
import tqdm

from mts.pipeline.repository.inmemeory import ImageRepository


class BasePipeline(ABC):
    @abstractmethod
    def run(self):
        pass


class BaseContext(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def tear_down(self):
        pass





class BaseIncrementalPipeline(BasePipeline):
    def __init__(
        self,
        repository: ImageRepository,
        feature_dir: str,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.extractor = extractor
        self.paring = paring
        self.repository = repository
        self.feature_dir = feature_dir

    def match(
        self,
        index_pairs,
        device=torch.device("cpu"),
        min_matches=15,
        verbose=True,
    ):


    def detect(
        self,
        image_indices,
    ) -> None:
        self.extractor.eval()
        with torch.inference_mode():
            for image_index in tqdm(image_indices):
                img_path = self.repository.get_filepath(image_index)
                feats0 = self.extractor.extract(img_path)
                kpts = feats0['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0['descriptors'].reshape(len(kpts), -1).detach().cpu().numpy()
                self.repository.add_keypoints(image_index, kpts)
                self.repository.add_descriptors(image_index, descs)

    def run(
        self,
        device,
        samples: dict,

    ):

        timings = {
            "shortlisting":[],
            "feature_detection": [],
            "feature_matching":[],
            "RANSAC": [],
            "Reconstruction": [],
        }
        print (f"Extracting on device {device}")
        for dataset, predictions in samples.items():
            
            images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
            images = [os.path.join(images_dir, p.filename) for p in predictions]
            if max_images is not None:
                images = images[:max_images]

            print(f'\nProcessing dataset "{dataset}": {len(images)} images')

            filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}

            feature_dir = os.path.join(workdir, 'featureout', dataset)
            os.makedirs(feature_dir, exist_ok=True)

            # Wrap algos in try-except blocks so we can populate a submission even if one scene crashes.
            try:
                t = time()
                index_pairs = get_image_pairs_shortlist(
                    images,
                    sim_th = 0.3, # should be strict
                    min_pairs = 20, # we select at least min_pairs PER IMAGE with biggest similarity
                    exhaustive_if_less = 20,
                    device=device
                )
                timings['shortlisting'].append(time() - t)
                print (f'Shortlisting. Number of pairs to match: {len(index_pairs)}. Done in {time() - t:.4f} sec')
            
                t = time()

                self.detect(images)
                timings['feature_detection'].append(time() - t)
                print(f'Features detected in {time() - t:.4f} sec')
                
                t = time()
                self.match(images, index_pairs)
                timings['feature_matching'].append(time() - t)
                print(f'Features matched in {time() - t:.4f} sec')

                database_path = os.path.join(feature_dir, 'colmap.db')
                if os.path.isfile(database_path):
                    os.remove(database_path)
                sleep(1)
                import_into_colmap(images_dir, feature_dir=feature_dir, database_path=database_path)
                output_path = f'{feature_dir}/colmap_rec_aliked'
                
                t = time()
                pycolmap.match_exhaustive(database_path)
                timings['RANSAC'].append(time() - t)
                print(f'Ran RANSAC in {time() - t:.4f} sec')
                
                # By default colmap does not generate a reconstruction if less than 10 images are registered.
                # Lower it to 3.
                mapper_options = pycolmap.IncrementalPipelineOptions()
                mapper_options.min_model_size = 3
                mapper_options.max_num_models = 25
                os.makedirs(output_path, exist_ok=True)
                t = time()
                maps = pycolmap.incremental_mapping(
                    database_path=database_path, 
                    image_path=images_dir,
                    output_path=output_path,
                    options=mapper_options)
                sleep(1)
                timings['Reconstruction'].append(time() - t)
                print(f'Reconstruction done in  {time() - t:.4f} sec')
                print(maps)

            
                registered = 0
                for map_index, cur_map in maps.items():
                    for index, image in cur_map.images.items():
                        
                        rigid3d = image.cam_from_world()
                        prediction_index = filename_to_index[image.name]
                        predictions[prediction_index].cluster_index = map_index
                        predictions[prediction_index].rotation = deepcopy(rigid3d.rotation.matrix())
                        predictions[prediction_index].translation = deepcopy(rigid3d.translation)
                        registered += 1
                mapping_result_str = f'Dataset "{dataset}" -> Registered {registered} / {len(images)} images with {len(maps)} clusters'
                mapping_result_strs.append(mapping_result_str)
                print(mapping_result_str)
            except Exception as e:
                print(e)
                raise e
                mapping_result_str = f'Dataset "{dataset}" -> Failed!'
                mapping_result_strs.append(mapping_result_str)
                print(mapping_result_str)

    @classmethod
    def from_config(cls, config):
        pass
