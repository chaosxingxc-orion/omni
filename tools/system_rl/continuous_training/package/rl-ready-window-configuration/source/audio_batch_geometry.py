"""Match the training encoder's single-recording geometry on uncached audio."""
from functools import wraps


def attach(model, mode):
    if mode!='single_clip_v1' or hasattr(model,'_omni_clip_geometry'):
        raise ValueError('unexpected or repeated encoder layout')
    original=model._process_audio_input
    @wraps(original)
    def encode(audio):
        lengths=audio['audio_feature_lengths']
        if lengths.ndim!=1 or not len(lengths):
            raise ValueError('invalid encoder input lengths')
        if len(lengths)==1:return original(audio)
        features=audio['input_features']
        if features.ndim!=2:
            raise ValueError('unsupported packed encoder fields')
        # The pinned model consumes only these two fields in this method.
        sizes=lengths.tolist()
        if min(sizes)<=0 or sum(sizes)!=features.shape[1]:
            raise ValueError('packed input does not match true recording lengths')
        outputs=[]
        for i,value in enumerate(features.split(sizes,dim=1)):
            item,=original(dict(input_features=value.contiguous(),audio_feature_lengths=lengths[i:i+1]))
            outputs.append(item)
        return tuple(outputs)
    model._process_audio_input=encode
    model._omni_clip_geometry=mode
    return mode
