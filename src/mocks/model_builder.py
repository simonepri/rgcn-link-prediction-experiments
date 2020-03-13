from mocks.decoders.complex import ComplEx
from mocks.decoders.distmult import DistMult
from mocks.decoders.analogy import Analogy


def build_decoder(encoder, decoder_settings):
    if decoder_settings["Name"] == "distmult":
        return DistMult(encoder, decoder_settings)

    if decoder_settings["Name"] == "complex":
        return ComplEx(encoder, decoder_settings)


    if decoder_settings["Name"] == "analogy":
        return TransE(encoder, decoder_settings)
    return None
