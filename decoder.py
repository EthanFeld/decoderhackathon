"""Self-contained optimized CPU decoder: adaptive default profile."""
import os as _os
_os.environ["QEC_DECODER_PROFILE"] = 'adaptive'
import organizer_decoder as _entry
from prior_model import DecoderConfig, DetectorModel, PriorTiltDecoder, model_from_dem, get_decoder_config
decode = _entry.decode
predict_proba = _entry.predict_proba
get_startup_metadata = getattr(_entry, "get_startup_metadata", lambda: {"profile": 'adaptive'})
close = getattr(_entry, "close", lambda: None)
if hasattr(_entry, "get_decoder"):
    get_decoder = _entry.get_decoder
__all__ = ["decode", "predict_proba", "get_startup_metadata", "close", "DecoderConfig", "DetectorModel", "PriorTiltDecoder", "model_from_dem", "get_decoder_config"]
