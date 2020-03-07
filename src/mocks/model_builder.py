from submodules.rgcn.code.common.model_builder import build_decoder as original_build_decoder

def build_decoder(encoder, decoder_settings):
    encoder = original_build_decoder(encoder, decoder_settings)
    if encoder != None:
        return encoder
    return None
