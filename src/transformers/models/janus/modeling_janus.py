from transformers import PreTrainedModel
from .configuration_janus import JanusConfig

class JanusPreTrainedModel(PreTrainedModel):
    config_class = JanusConfig
    base_model_prefix = "janus"

    def _init_weights(self, module):
        pass

class JanusForConditionalGeneration(JanusPreTrainedModel):
    config_class = JanusConfig

    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        pass

    def generate(self, **kwargs):
        pass