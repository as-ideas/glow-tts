import numpy as np
import torch
from matplotlib.figure import Figure

import commons
import models_gen
import utils
from text import text_to_sequence, cmudict, sequence_to_text
from text.symbols import phonemes

# If you are using your own trained model
from utils import plot_spectrogram_to_numpy

model_dir = "./logs/base/"
hps = utils.get_hparams_from_dir(model_dir)
checkpoint_path = utils.latest_checkpoint_path(model_dir)

# If you are using a provided pretrained model
# hps = utils.get_hparams_from_file("./configs/any_config_file.json")
# checkpoint_path = "/path/to/pretrained_model"

model = models_gen.FlowGenerator(
    len(phonemes) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model)

#model, o, l, it = utils.load_checkpoint(checkpoint_path, model)
#print('loaded model at iteration: ', it)
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])


model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
_ = model.eval()

cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

tst_stn = "ʃaŋhaɪ ziːt aʊs viː diː kuːlɪsə fyːɐ aɪnən saɪəns-fɪkʃən-ɛnt͡saɪtfɪlm: diː ʃtʁaːsn̩ zɪnt fast leːɐ, ʊnt diː veːnɪɡn̩ mɛnʃn̩, diː ʊntɐveːks zɪnt, tʁaːɡn̩ nɪçt bloːs ɡəzɪçt͡smaskə, zɔndɐn zɪnt kɔmplɛt fɛɐhʏlt - ɪnfɛkt͡sioːns-ʃʊt͡s-klaɪdʊŋ fɔn kɔp͡f bɪs fuːs."

if getattr(hps.data, "add_blank", False):
    text_norm = text_to_sequence(tst_stn.strip(), ['basic_cleaners'], cmu_dict)
    text_norm = commons.intersperse(text_norm, len(phonemes))
    print(text_norm)
else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
    tst_stn = " " + tst_stn.strip() + " "
    text_norm = text_to_sequence(tst_stn.strip(), ['basic_cleaners'], cmu_dict)
print(text_norm)
print(sequence_to_text(text_norm))
sequence = np.array(text_norm)[None, :]
print("".join([phonemes[c] if c < len(phonemes) else "<BNK>" for c in sequence[0]]))
x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).long()
x_tst_lengths = torch.tensor([x_tst.shape[1]])

def plot_mel(mel: np.array) -> Figure:
    mel = np.flip(mel, axis=0)
    fig = plt.figure(figsize=(12, 6), dpi=150)
    plt.imshow(mel, interpolation='nearest', aspect='auto')
    return fig

def plot_spectrogram_to_numpy(spectrogram):
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots()
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #plt.close()
    return data
with torch.no_grad():
    noise_scale = 1.0
    length_scale = 1.0
    (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst[:1], x_tst_lengths[:1], gen=True, noise_scale=noise_scale, length_scale=length_scale)
    import matplotlib.pyplot as plt
    plot_spectrogram_to_numpy(y_gen_tst[0].data.cpu().numpy())
    #plot_mel(y_gen_tst.squeeze().cpu().numpy())
    plt.savefig('/tmp/glow_mel.png')
    print(y_gen_tst.size())
    print(y_gen_tst)
    torch.save(y_gen_tst.detach().squeeze(), f'model_outputs/glow.mel')