from .odin_postprocessor import ODINPostprocessor
from .base_postprocessor import BasePostprocessor
from .doctor_postprocessor import DoctorPostprocessor
from .margin_postprocessor import MarginPostprocessor
from .partition_postprocessor import PartitionPostprocessor
from .random_forest_postprocessor import RandomForestPostprocessor
from .base_scikit_postprocessor import GenericScikitPostprocessor
from .mlp_postprocessor_scikit import MLPPostprocessor
from .conformal_postprocessor import ConformalPostprocessor
from .isotonic_postprocessor import IsotonicPostprocessor
from .isotonic_splitting_postprocessor import IsotonicSplittingPostprocessor


postprocessors = {
    "odin": ODINPostprocessor,
    "msp": BasePostprocessor,
    "doctor": DoctorPostprocessor,
    "margin": MarginPostprocessor,
    "partition": PartitionPostprocessor,
    "random_forest": RandomForestPostprocessor,
    "scikit": GenericScikitPostprocessor,
    "mlp": MLPPostprocessor,
    "conformal": ConformalPostprocessor,
    "isotonic": IsotonicPostprocessor,
    "isotonic_splitting": IsotonicSplittingPostprocessor,
    }

def get_postprocessor(postprocessor_name, model, cfg, result_folder, device):
    if postprocessor_name not in postprocessors:
        raise ValueError(f"Postprocessor {postprocessor_name} not found. Available postprocessors: {list(postprocessors.keys())}")
    return postprocessors[postprocessor_name](model, cfg, result_folder, device=device)
   
