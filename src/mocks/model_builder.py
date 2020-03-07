from mocks.decoders.complex import ComplEx
from mocks.decoders.distmult import DistMult

def build_decoder(encoder, decoder_settings):
    if decoder_settings["Name"] == "distmult":
        return DistMult(encoder, decoder_settings)
    elif decoder_settings["Name"] == "complex":
        return ComplEx(encoder, decoder_settings)
    else:
        return None
