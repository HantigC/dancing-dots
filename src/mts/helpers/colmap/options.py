import pycolmap


def create_incremental_pipeline_options(
    **kwargs,
) -> pycolmap.IncrementalPipelineOptions:
    mapper_options = pycolmap.IncrementalPipelineOptions()
    for k, v in kwargs.items():
        setattr(mapper_options, k, v)
    return mapper_options
