import warnings
import copy
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.distributions import Categorical
from typing import List, Optional, Tuple, Union
from whisper import load_model as load_ori_model, load_audio
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import LANGUAGES
from whisper.utils import exact_div, format_timestamp, compression_ratio
from whisper.model import Whisper
from whisper.decoding import DecodingTask, BeamSearchDecoder, GreedyDecoder
from whisper.tokenizer import Tokenizer, get_tokenizer
from types import MethodType
from itertools import repeat
from stable_whisper.audio import load_audio_waveform_img, remove_lower_quantile, wave_to_ts_filter
from stable_whisper.stabilization import stabilize_timestamps, add_whole_word_ts
from tqdm import tqdm

__all__ = ['transcribe', 'decode', 'modify_model', 'load_model']

def dim(a):
    if not type(a) == list:
        if not torch.is_tensor(a) or (torch.is_tensor(a) and a.shape == torch.empty(1)[0].shape):
            return []

    if len(a) == 0:
        return []

    return [len(a)] + dim(a[0])


# no_caption changed to no_speech in newer whisper commits
def _get_new_attrs(obj_, attr: str):
    if attr == 'no_caption_probs':
        return getattr(obj_, attr) if hasattr(obj_, 'no_caption_probs') else getattr(obj_, 'no_speech_probs')
    elif attr == 'no_caption_prob':
        return getattr(obj_, attr) if hasattr(obj_, 'no_caption_prob') else getattr(obj_, 'no_speech_prob')
    elif attr == 'no_captions':
        return getattr(obj_, attr) if hasattr(obj_, 'no_captions') else getattr(obj_, 'no_speech')
    else:
        raise NotImplementedError(attr)


# modified version of whisper.transcribe.transcribe
def transcribe(
        model: "Whisper",
        audio: Union[str, List, np.ndarray, torch.Tensor],
        *,
        verbose: bool = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        language_threshold: Optional[float] = 0.6,
        language_detection_segments: int = 1,
        condition_on_previous_text: bool = True,
        stab=True, top_focus=False, ts_num: int = 10,
        alpha: float = None, print_unstab=False, pbar=False,
        suppress_silence: bool = True,
        suppress_middle: bool = True,
        suppress_word_ts: bool = True,
        remove_background: bool = True,
        silence_threshold: float = 0.1,
        prepend_punctuations: Union[List[str], Tuple[str]] = None,
        append_punctuations: Union[List[str], Tuple[str]] = None,
        audio_for_mask: (str, bytes) = None,
        **decode_options):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model modified instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the decoded text (with finalized timestamps) to the console (Default: False)
        Use print_unstab for previous behavior of verbose but with token timestamps

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    stab: bool
        Stabilizing timestamps by cross compare timestamps and using additional top timestamp predictions
        to fill in when appropriate to ensure timestamps are chronological.

    top_focus: bool
        Adhere closely to the top predictions for token timestamps stabilization

    ts_num: int
        Number of top timestamp predictions to save for each word for postprocessing stabilization (default: 10).

    alpha: float
        Amount of noise to add to audio to produce slightly difference results.
        audio_features *= torch.rand_like(audio_features) * alpha + 1

    print_unstab: bool
        Whether to display the text (without stabilize timestamps) being decoded to the console (Default: False)
        (i.e. behaves like verbose before model was modified and progress bar will be disabled if True)

    pbar: bool
        Whether to enable progress bar for the decoding process (Default: False). Ignored if print_unstab=True

    suppress_silence: bool
        Suppress timestamp tokens that are marked as silent

    suppress_middle: bool
        Suppress any silent timestamps tokens of middle of the segment instead of only beginning and ending

    suppress_word_ts: bool
        Suppress timestamp tokens of words that are marked as silent

    remove_background: bool
        Whether to remove background noise from waveform so that it is marked silent.
        Determined by parameters part of decode_options (i.e. specify like other options here):
            upper_quantile: float
                The upper quantile of amplitude to determine a max amplitude, mx (Default: 0.85)
            lower_quantile: float
                The lower quantile of amplitude to determine a min amplitude, mn (Default: 0.15)
            lower_threshold: float
                Suppressed sections of waveform where amplitude < lower_threshold*(mx-mn) + mn. (Default: 0.15)

    silence_threshold: float:
        Audio segments silence average >= silence_threshold
        then that segment will not have background removed even if remove_background=True.
        e.g. 0.5 means if less than half of the audio segment is silent then background will be removed accordingly

    prepend_punctuations: Union[List[str], Tuple[str]]
        Punctuations to prepend to next word (Default: “¿([{)

    append_punctuations: Union[List[str], Tuple[str]]
        Punctuations to append to previous word (Default: .。,，!！?？:：”)]}、)

    audio_for_mask: (str, bytes)
        Original audio track as path or bytes of audio file.
        Since resampled audio may shift the waveform image,
        this is an alternative to 'audio' option to generate suppression mask from the original audio.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    if 'no_captions_threshold' in decode_options:
        warnings.warn('no_captions_threshold is deprecated. '
                      'Please use no_speech_threshold instead.', DeprecationWarning, stacklevel=2)
        no_speech_threshold = decode_options.pop('no_captions_threshold')

    if type(audio) == list:
        return batch_transcribe(model=model,
                                audio=audio,
                                verbose=verbose,
                                temperature=temperature,
                                compression_ratio_threshold=compression_ratio_threshold,
                                logprob_threshold=logprob_threshold,
                                no_speech_threshold=no_speech_threshold,
                                condition_on_previous_text=condition_on_previous_text,
                                stab=stab, top_focus=top_focus, ts_num=ts_num,
                                alpha=alpha, print_unstab=print_unstab, pbar=pbar,
                                suppress_silence=suppress_silence,
                                suppress_middle=suppress_middle,
                                suppress_word_ts=suppress_word_ts,
                                remove_background=remove_background,
                                silence_threshold=silence_threshold,
                                prepend_punctuations=prepend_punctuations,
                                append_punctuations=append_punctuations,
                                audio_for_mask=audio_for_mask,
                                **decode_options)

    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if 'max_initial_timestamp' not in decode_options:
        decode_options['max_initial_timestamp'] = None

    mel = log_mel_spectrogram(audio)
    num_frames = mel.shape[-1]

    if decode_options.get("language", None) is None:
        if verbose:
            print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
        if language_detection_segments is None or language_detection_segments < 1:
            language_detection_segments = 1
        seek = 0
        languages = []
        while seek < num_frames and seek < N_FRAMES * language_detection_segments:
            segment = pad_or_trim(mel[:, seek:], N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(segment)
            lang = max(probs, key=probs.get)
            lang_prob = probs[lang]
            if language_threshold is not None and lang_prob > language_threshold:
                decode_options["language"] = lang
                break
            else:
                languages.append(lang)
                seek += segment.shape[-1]
        else:
            # If no language detected for all segments, the majority vote of the highest projected
            # languages for all segments is used to determine the language.
            decode_options["language"] = max(set(languages), key=languages.count)

    mel = mel.unsqueeze(0)
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    ignore_shift = decode_options.pop('ignore_shift', False)

    def decode_with_fallback(segment: torch.Tensor, suppress_ts_mask: Tensor = None) \
            -> Union[List[DecodingResult], tuple]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        kwargs = {**decode_options}
        t = temperatures[0]
        if t == 0:
            best_of = kwargs.pop("best_of", None)
        else:
            best_of = kwargs.get("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        results, ts_tokens, ts_logits_, tc = model.decode(segment, options, ts_num=ts_num, alpha=alpha,
                                                      suppress_ts_mask=suppress_ts_mask,
                                                      suppress_word_ts=suppress_word_ts)  # my

        kwargs.pop("beam_size", None)  # no beam search for t > 0
        kwargs.pop("patience", None)  # no patience for t > 0
        kwargs["best_of"] = best_of  # enable best_of for t > 0
        for t in temperatures[1:]:
            # TODO This part needs to be adapted into batch inference
            needs_fallback = [
                compression_ratio_threshold is not None
                and result.compression_ratio > compression_ratio_threshold
                or logprob_threshold is not None
                and result.avg_logprob < logprob_threshold
                for result in results
            ]

            if any(needs_fallback):
                options = DecodingOptions(**kwargs, temperature=t)
                retries, r_ts_tokens, r_ts_logits, tc = model.decode(segment[needs_fallback], options,
                                                                 ts_num=ts_num, alpha=alpha,
                                                                 suppress_ts_mask=suppress_ts_mask,
                                                                 suppress_word_ts=suppress_word_ts)
                for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                    results[original_index] = retries[retry_index]
                    ts_tokens[original_index] = r_ts_tokens[retry_index]
                    ts_logits_[original_index] = r_ts_logits[retry_index]

        return results, ts_tokens, ts_logits_, tc

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def _to_list(x: (Tensor, None)):
        if x is None:
            return x
        return x.tolist()

    def add_segment(
            *, offset: float, start: float, end: float, text_tokens: Tensor, result: DecodingResult,
            start_timestamps: list = None, end_timestamps: list = None, word_timestamps: Tensor = None,
            start_ts_logits: list = None, end_ts_logits: list = None, word_ts_logits: Tensor = None,
            tc_logits: Tensor = None
    ):
        no_eot_mask = text_tokens < tokenizer.eot
        text_tokens_no_eot = text_tokens[no_eot_mask]
        text = tokenizer.decode(text_tokens_no_eot)

        if len(text.strip()) == 0:  # skip empty text output
            return

        if word_timestamps is not None:
            assert word_timestamps.shape[0] == text_tokens.shape[0]
            if word_ts_logits is None:
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], repeat(None))
            else:
                assert word_ts_logits.shape[0] == text_tokens.shape[0]
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], word_ts_logits[no_eot_mask])

            word_timestamps = [dict(word=tokenizer.decode([token]),
                                    token=token.item(),
                                    timestamps=timestamps_.tolist(),
                                    timestamp_logits=_to_list(ts_logits_))
                               for token, timestamps_, ts_logits_ in word_ts_fields]

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                'offset': offset,  # offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": _get_new_attrs(result, 'no_caption_prob'),
                "alt_start_timestamps": start_timestamps,
                "start_ts_logits": start_ts_logits,
                "alt_end_timestamps": end_timestamps,
                "end_ts_logits": end_ts_logits,
                "unstable_word_timestamps": word_timestamps,
                'anchor_point': False,
                "confidence_score": tc_logits  # my
            }
        )
        if print_unstab or (verbose and not stab):
            print(f'[{format_timestamp(start)} --> {format_timestamp(end)}] "{text}"')
            if word_timestamps is not None:
                ts_str = (f' ->[{format_timestamp(ts_["timestamps"][0])}] "{ts_["word"].strip()}"' for ts_ in
                          word_timestamps)
                print('\n'.join(ts_str), end='\n\n')

    if suppress_silence:
        all_silent = False
        ts_scale = HOP_LENGTH / SAMPLE_RATE / time_precision
        wfh, wfw = 100, int(mel.shape[-1] * ts_scale)
        wf = load_audio_waveform_img(audio_for_mask or audio, wfh, wfw, ignore_shift=ignore_shift)
        if not wf.any():
            if audio_for_mask:
                wf = load_audio_waveform_img(load_audio(audio) if isinstance(audio, str) else audio,
                                             wfh, wfw, ignore_shift=True)
            else:
                if isinstance(audio, str):
                    wf = load_audio_waveform_img(load_audio(audio), wfh, wfw, ignore_shift=True)
                else:
                    all_silent = True

            if not all_silent:
                all_silent = not wf.any()
            if all_silent:
                warnings.warn('The audio appears to be entirely silent. suppress_silence will be set to False',
                              stacklevel=2)
                suppress_silence = False

    upper_quantile = decode_options.pop('upper_quantile', 0.85)
    lower_quantile = decode_options.pop('lower_quantile', 0.15)
    lower_threshold = decode_options.pop('lower_threshold', 0.15)

    with tqdm(total=num_frames, unit='frames', disable=(print_unstab or not pbar)) as tqdm_pbar:

        def update_pbar():
            if not tqdm_pbar.disable:
                tqdm_pbar.update(min(num_frames, seek) - tqdm_pbar.n)

        while seek < mel.shape[-1]:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            remaining_duration = float((mel.shape[-1] - seek) * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, :, seek:], N_FRAMES).to(model.device).to(dtype)
            segment_duration = min(float(segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE), remaining_duration)
            segment_max_ts = segment_duration / time_precision

            if suppress_silence:
                wf_seek = int(seek * ts_scale)
                segment_wf = wf[..., wf_seek:wf_seek + 1501]
                if remove_background and \
                        (1 - segment_wf.sum(0).clip(max=1).mean()) < silence_threshold:
                    segment_wf = remove_lower_quantile(segment_wf.astype(np.float32),
                                                       upper_quantile=upper_quantile,
                                                       lower_quantile=lower_quantile,
                                                       lower_threshold=lower_threshold)
                segment_wf = pad_or_trim(segment_wf, 1501)
                suppress_ts_mask = torch.from_numpy(wave_to_ts_filter(segment_wf,
                                                                      suppress_middle=suppress_middle,
                                                                      max_index=int(segment_max_ts)))

                if suppress_ts_mask.all():  # segment is silent
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    update_pbar()
                    continue
            else:
                suppress_ts_mask = None

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result, finalized_ts_tokens, ts_logits, tc = decode_with_fallback(segment,
                                                                          suppress_ts_mask=suppress_ts_mask)

            result = result[0]
            tokens = torch.tensor(result.tokens)
            finalized_ts_tokens = torch.tensor(finalized_ts_tokens[0])
            ts_logits = torch.tensor(ts_logits[0])

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = _get_new_attrs(result, 'no_caption_prob') > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    sliced_ts_tokens = finalized_ts_tokens[last_slice:current_slice]
                    sliced_ts_logits = ts_logits[last_slice:current_slice]
                    sliced_tc = tc[last_slice:current_slice]
                    start_timestamp_position = (
                            sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                            sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )

                    word_ts = timestamp_offset + sliced_ts_tokens * time_precision

                    add_segment(
                        offset=timestamp_offset,
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=min(timestamp_offset + end_timestamp_position * time_precision,
                                timestamp_offset + segment_duration),
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                        start_timestamps=word_ts[0].tolist(),
                        end_timestamps=word_ts[-1].tolist(),
                        word_timestamps=word_ts[1:-1],
                        start_ts_logits=sliced_ts_logits[0].tolist(),
                        end_ts_logits=sliced_ts_logits[-1].tolist(),
                        word_ts_logits=sliced_ts_logits[1:-1],
                        tc_logits=sliced_tc[1:-1]  # my
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                    min(tokens[last_slice - 1].item() - tokenizer.timestamp_begin, segment_max_ts)
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = min(timestamps[-1].item() - tokenizer.timestamp_begin, segment_max_ts)
                    duration = last_timestamp_position * time_precision

                word_ts = timestamp_offset + finalized_ts_tokens * time_precision

                add_segment(
                    offset=timestamp_offset,
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                    word_timestamps=word_ts,
                    word_ts_logits=ts_logits,
                    tc_logits=tc
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if all_segments:
                all_segments[-1]['anchor_point'] = True
                all_segments[-1]['next_offset'] = float(seek * HOP_LENGTH / SAMPLE_RATE)
            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            update_pbar()

    if len(all_segments) > 1 and all_segments[-1]['alt_start_timestamps'] is None:
        all_segments[-1]['alt_start_timestamps'] = all_segments[-2]['alt_end_timestamps']

    # # my prepare confidence
    # all_segments = my_prepare_confidence_and_words(tokenizer, all_segments)  # side effect

    if stab:
        all_segments = stabilize_timestamps(all_segments, top_focus=top_focus)
        add_whole_word_ts(tokenizer, all_segments,
                          merge_non_space=True,  # my
                          prepend_punctuations=prepend_punctuations,
                          append_punctuations=append_punctuations)
        if verbose:
            print('\nSTABILIZED:')
            for seg_ in all_segments:
                print(f'[{format_timestamp(seg_["start"])} --> {format_timestamp(seg_["end"])}] "{seg_["text"]}"')
                if seg_['word_timestamps']:
                    ts_str = (f' ->[{format_timestamp(ts_["timestamp"])}] "{ts_["word"].strip()}"' for ts_ in
                               seg_['word_timestamps'])
                    print('\n'.join(ts_str), end='\n\n')
        import pickle
        with open('filename.pickle', 'wb') as fh:
            pickle.dump(all_segments, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language)

# modified version of whisper.transcribe.transcribe to allow for batch inference
def batch_transcribe(
        model: "Whisper",
        audio: Union[str, List, np.ndarray, torch.Tensor],
        *,
        verbose: bool = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        stab=True, top_focus=False, ts_num: int = 10,
        alpha: float = None, print_unstab=False, pbar=False,
        suppress_silence: bool = True,
        suppress_middle: bool = True,
        suppress_word_ts: bool = True,
        remove_background: bool = True,
        silence_threshold: float = 0.1,
        prepend_punctuations: Union[List[str], Tuple[str]] = None,
        append_punctuations: Union[List[str], Tuple[str]] = None,
        audio_for_mask: (str, bytes) = None,
        **decode_options):
    """
    Transcribe multiple audio files in parallel using the batch dimension of the Whisper model

    Parameters
    ----------
    model: Whisper
        The Whisper model modified instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The list of paths to the audio files to open, or the audio waveforms

    verbose: bool
        Whether to display the decoded text (with finalized timestamps) to the console (Default: False)
        Use print_unstab for previous behavior of verbose but with token timestamps

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    stab: bool
        Stabilizing timestamps by cross compare timestamps and using additional top timestamp predictions
        to fill in when appropriate to ensure timestamps are chronological.

    top_focus: bool
        Adhere closely to the top predictions for token timestamps stabilization

    ts_num: int
        Number of top timestamp predictions to save for each word for postprocessing stabilization (default: 10).

    alpha: float
        Amount of noise to add to audio to produce slightly difference results.
        audio_features *= torch.rand_like(audio_features) * alpha + 1

    print_unstab: bool
        Whether to display the text (without stabilize timestamps) being decoded to the console (Default: False)
        (i.e. behaves like verbose before model was modified and progress bar will be disabled if True)

    pbar: bool
        Whether to enable progress bar for the decoding process (Default: False). Ignored if print_unstab=True

    suppress_silence: bool
        Suppress timestamp tokens that are marked as silent

    suppress_middle: bool
        Suppress any silent timestamps tokens of middle of the segment instead of only beginning and ending

    suppress_word_ts: bool
        Suppress timestamp tokens of words that are marked as silent

    remove_background: bool
        Whether to remove background noise from waveform so that it is marked silent.
        Determined by parameters part of decode_options (i.e. specify like other options here):
            upper_quantile: float
                The upper quantile of amplitude to determine a max amplitude, mx (Default: 0.85)
            lower_quantile: float
                The lower quantile of amplitude to determine a min amplitude, mn (Default: 0.15)
            lower_threshold: float
                Suppressed sections of waveform where amplitude < lower_threshold*(mx-mn) + mn. (Default: 0.15)

    silence_threshold: float:
        Audio segments silence average >= silence_threshold
        then that segment will not have background removed even if remove_background=True.
        e.g. 0.5 means if less than half of the audio segment is silent then background will be removed accordingly

    prepend_punctuations: Union[List[str], Tuple[str]]
        Punctuations to prepend to next word (Default: “¿([{)

    append_punctuations: Union[List[str], Tuple[str]]
        Punctuations to append to previous word (Default: .。,，!！?？:：”)]}、)

    audio_for_mask: (str, bytes)
        Original audio track as path or bytes of audio file.
        Since resampled audio may shift the waveform image,
        this is an alternative to 'audio' option to generate suppression mask from the original audio.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A list of dictionaries containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    if 'no_captions_threshold' in decode_options:
        warnings.warn('no_captions_threshold is deprecated. '
                      'Please use no_speech_threshold instead.', DeprecationWarning, stacklevel=2)
        no_speech_threshold = decode_options.pop('no_captions_threshold')


    batch_size = len(audio)
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if 'max_initial_timestamp' not in decode_options:
        decode_options['max_initial_timestamp'] = None

    mels = [log_mel_spectrogram(audio_file) for audio_file in audio]
    segments = [pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype) for mel in mels]

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            languages = ['en'] * len(audio)
        else:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            language_probs = [model.detect_language(segment)[1] for segment in segments]
            languages = [max(probs, key=probs.get) for probs in language_probs]
            if verbose is not None:
                print(f"Detected languages: {[LANGUAGES[opt].title() for opt in languages]}")
    else:
        lang = decode_options.get("language")
        if type(lang) == str:
            languages = [lang] * len(audio)
        elif type(lang) == list:
            assert all(isinstance(l, str) for l in
                       lang), "If a list of languages is specified in DecodeOptions, all languages must be strings."
            assert len(lang) == len(
                audio), "If a list of languages is specified in DecodeOptions, the list length must match the number of audio files specified."
            languages = lang
        else:
            raise NotImplementedError("Only string and list arguments are supported for the language DecodeOption.")

    mels = [mel.unsqueeze(0) for mel in mels]
    task = decode_options.get("task", "transcribe")
    tokenizers = {}
    for lang in languages:
        if lang not in tokenizers.keys():
            tokenizers[lang] = get_tokenizer(model.is_multilingual, language=lang, task=task)

    ignore_shift = decode_options.pop('ignore_shift', False)

    def decode_with_fallback(segment: torch.Tensor, suppress_ts_mask: Tensor = None) \
            -> Union[List[DecodingResult], tuple]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        results = None
        ts_tokens = None
        ts_logits_ = None
        tc = None
        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            results, ts_tokens, ts_logits_, tc = model.decode(segment, options, ts_num=ts_num, alpha=alpha,
                                                          suppress_ts_mask=suppress_ts_mask,
                                                          suppress_word_ts=suppress_word_ts)

            needs_fallback = False
            if type(results) == list:
                for result in results:
                    if compression_ratio_threshold is not None and result.compression_ratio > compression_ratio_threshold:
                        needs_fallback = True  # too repetitive
                    if logprob_threshold is not None and result.avg_logprob < logprob_threshold:
                        needs_fallback = True  # average log probability is too low
            else:
                if compression_ratio_threshold is not None and result.compression_ratio > compression_ratio_threshold:
                    needs_fallback = True  # too repetitive
                if logprob_threshold is not None and result.avg_logprob < logprob_threshold:
                    needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return results, ts_tokens, ts_logits_, tc

        # kwargs = {**decode_options}
        # t = temperatures[0]
        # if t == 0:
        #     best_of = kwargs.pop("best_of", None)
        # else:
        #     best_of = kwargs.get("best_of", None)
        #
        # options = DecodingOptions(**kwargs, temperature=t)
        # results, ts_tokens, ts_logits_, tc = model.decode(segment, options, ts_num=ts_num, alpha=alpha,
        #                                               suppress_ts_mask=suppress_ts_mask,
        #                                               suppress_word_ts=suppress_word_ts)
        #
        # kwargs.pop("beam_size", None)  # no beam search for t > 0
        # kwargs.pop("patience", None)  # no patience for t > 0
        # kwargs["best_of"] = best_of  # enable best_of for t > 0
        # for t in temperatures[1:]:
        #     needs_fallback = [
        #         compression_ratio_threshold is not None
        #         and result.compression_ratio > compression_ratio_threshold
        #         or logprob_threshold is not None
        #         and result.avg_logprob < logprob_threshold
        #         for result in results
        #     ]
        #     if any(needs_fallback):
        #         options = DecodingOptions(**kwargs, temperature=t)
        #         retries, r_ts_tokens, r_ts_logits, tc = model.decode(segment[needs_fallback], options,
        #                                                          ts_num=ts_num, alpha=alpha,
        #                                                          suppress_ts_mask=suppress_ts_mask,
        #                                                          suppress_word_ts=suppress_word_ts)
        #         for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
        #             results[original_index] = retries[retry_index]
        #             ts_tokens[original_index] = r_ts_tokens[retry_index]
        #             ts_logits_[original_index] = r_ts_logits[retry_index]
        #
        # return results, ts_tokens, ts_logits_, tc

    seekers = [0] * len(audio)
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = [[] for _ in range(batch_size)]
    all_segments = [[] for _ in range(batch_size)]
    prompt_reset_since = [0] * batch_size

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    initial_prompts = []
    if initial_prompt:
        assert len(initial_prompt) == batch_size, "Number of initial prompts must match batch size."
        for i in range(batch_size):
            initial_prompts.append(tokenizers[languages[i]].encode(" " + initial_prompt[i].strip()))
            all_tokens.extend(initial_prompt)

    def _to_list(x: (Tensor, None)):
        if x is None:
            return x
        return x.tolist()

    def add_segment(
            *, seeker: int, offset: float, start: float, end: float, text_tokens: Tensor, result: DecodingResult,
            tokenizer, segments, start_timestamps: list = None, end_timestamps: list = None, word_timestamps: Tensor = None,
            start_ts_logits: list = None, end_ts_logits: list = None, word_ts_logits: Tensor = None,
            tc_logits: Tensor = None
    ):
        no_eot_mask = text_tokens < tokenizer.eot
        text_tokens_no_eot = text_tokens[no_eot_mask]
        text = tokenizer.decode(text_tokens_no_eot)

        if len(text.strip()) == 0:  # skip empty text output
            return

        if word_timestamps is not None:
            assert word_timestamps.shape[0] == text_tokens.shape[0]
            if word_ts_logits is None:
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], repeat(None))
            else:
                assert word_ts_logits.shape[0] == text_tokens.shape[0]
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps[no_eot_mask], word_ts_logits[no_eot_mask])

            word_timestamps = [dict(word=tokenizer.decode([token]),
                                    token=token.item(),
                                    timestamps=timestamps_.tolist(),
                                    timestamp_logits=_to_list(ts_logits_))
                               for token, timestamps_, ts_logits_ in word_ts_fields]

        segments.append(
            {
                "id": len(all_segments),
                "seek": seeker,
                'offset': offset,  # offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": _get_new_attrs(result, 'no_caption_prob'),
                "alt_start_timestamps": start_timestamps,
                "start_ts_logits": start_ts_logits,
                "alt_end_timestamps": end_timestamps,
                "end_ts_logits": end_ts_logits,
                "unstable_word_timestamps": word_timestamps,
                'anchor_point': False,
                "confidence_score": tc_logits  # my
            }
        )
        if print_unstab or (verbose and not stab):
            print(f'[{format_timestamp(start)} --> {format_timestamp(end)}] "{text}"')
            if word_timestamps is not None:
                ts_str = (f' ->[{format_timestamp(ts_["timestamps"][0])}] "{ts_["word"].strip()}"' for ts_ in
                          word_timestamps)
                print('\n'.join(ts_str), end='\n\n')

    batch_suppress_silence = [suppress_silence] * len(mels)
    for i in range(len(mels)):
        if batch_suppress_silence[i]:
            all_silent = False
            ts_scale = HOP_LENGTH / SAMPLE_RATE / time_precision
            wfh, wfw = 100, int(mels[i].shape[-1] * ts_scale)
            wf = load_audio_waveform_img(audio_for_mask or audio[i], wfh, wfw, ignore_shift=ignore_shift)
            if not wf.any():
                if audio_for_mask:
                    wf = load_audio_waveform_img(load_audio(audio[i]) if isinstance(audio[i], str) else audio[i],
                                                 wfh, wfw, ignore_shift=True)
                else:
                    if isinstance(audio, str):
                        wf = load_audio_waveform_img(load_audio(audio[i]), wfh, wfw, ignore_shift=True)
                    else:
                        all_silent = True

                if not all_silent:
                    all_silent = not wf.any()
                if all_silent:
                    warnings.warn(f'Audio {i} appears to be entirely silent. suppress_silence will be set to False',
                                  stacklevel=2)
                    batch_suppress_silence[i] = False

    upper_quantile = decode_options.pop('upper_quantile', 0.85)
    lower_quantile = decode_options.pop('lower_quantile', 0.15)
    lower_threshold = decode_options.pop('lower_threshold', 0.15)

    num_frames = [mel.shape[-1] for mel in mels]

    def check_cursors(seekers: List[int], num_frames: List[int]) -> bool:
        """Return False when all seekers have exhausted the length of their audio clips."""
        return any([seeker < nf for seeker, nf in list(zip(seekers, num_frames))])

    with tqdm(total=num_frames, unit='frames', disable=(print_unstab or not pbar)) as tqdm_pbar:
        def update_pbar():
            if not tqdm_pbar.disable:
                midx = num_frames.index(max(num_frames))
                tqdm_pbar.update(min(num_frames[midx], seekers[midx]) - tqdm_pbar.n)

        while check_cursors(seekers, num_frames):
            continue_processing = [seeker < nf for seeker, nf in list(zip(seekers, num_frames))]
            # Only those segments for clips that are not done being processed
            imap = [i for i, v in enumerate(continue_processing) if v]
            batch_segments = []
            batch_segment_durations = []
            batch_timestamp_offsets = []
            batch_suppress_ts_mask = []
            batch_segment_max_ts = []

            for i, mel in enumerate(mels):
                if continue_processing[i]:

                    timestamp_offset = float(seekers[i] * HOP_LENGTH / SAMPLE_RATE)
                    batch_timestamp_offsets.append(timestamp_offset)
                    remaining_duration = float((mel.shape[-1] -seekers[i]) * HOP_LENGTH / SAMPLE_RATE)
                    segment = pad_or_trim(mel[:, :, seekers[i]:], N_FRAMES).to(model.device).to(dtype)
                    batch_segments.append(segment)
                    segment_duration = min(float(segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE), remaining_duration)
                    batch_segment_durations.append(segment_duration)
                    segment_max_ts = segment_duration / time_precision
                    batch_segment_max_ts.append(segment_max_ts)

                    if batch_suppress_silence[i]:
                        wf_seek = int(seekers[i] * ts_scale)
                        segment_wf = wf[..., wf_seek:wf_seek + 1501]
                        if remove_background and \
                                (1 - segment_wf.sum(0).clip(max=1).mean()) < silence_threshold:
                            segment_wf = remove_lower_quantile(segment_wf.astype(np.float32),
                                                               upper_quantile=upper_quantile,
                                                               lower_quantile=lower_quantile,
                                                               lower_threshold=lower_threshold)
                        segment_wf = pad_or_trim(segment_wf, 1501)
                        suppress_ts_mask = torch.from_numpy(wave_to_ts_filter(segment_wf,
                                                                              suppress_middle=suppress_middle,
                                                                              max_index=int(segment_max_ts)))
                        if suppress_ts_mask.all():  # segment is silent
                            seekers[i] += segment.shape[-1]  # fast-forward to the next segment boundary
                            batch_suppress_ts_mask.append(suppress_ts_mask)
                            update_pbar()
                            continue
                    else:
                        suppress_ts_mask = None
                    batch_suppress_ts_mask.append(suppress_ts_mask)
                else:
                    continue

            print("populate", "batch_segment",dim(batch_segments), "batch_suppress_ts_mask", dim(batch_suppress_ts_mask))

            decode_options["prompt"] = [all_tokens[imap[i]][prompt_reset_since[imap[i]]:] for i in range(len(batch_segments))]
            decode_options["language"] = [l for i, l in enumerate(languages) if continue_processing[i]]

            results, finalized_ts_tokens, ts_logits, tc = decode_with_fallback(torch.stack(batch_segments),
                                                                          suppress_ts_mask=torch.stack(batch_suppress_ts_mask))

            batch_tokens = [torch.tensor(result.tokens) for result in results]
            batch_finalized_ts_tokens = [torch.tensor(finalized_ts_token) for finalized_ts_token in finalized_ts_tokens]
            batch_ts_logits = [torch.tensor(ts_logit) for ts_logit in ts_logits]
            batch_tc = [tc_slice for tc_slice in tc]

            for i, result in enumerate(results):
                if no_speech_threshold is not None:
                    # no voice activity check
                    # print('no_caption_prob', _get_new_attrs(result, 'no_caption_prob'), "no_speech_threshold",
                    #       no_speech_threshold)
                    # Todo adapt to batch inference
                    should_skip = _get_new_attrs(result, 'no_caption_prob')[0] > no_speech_threshold
                    if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                        # don't skip if the logprob is high enough, despite the no_speech_prob
                        should_skip = False

                    if should_skip:
                        seekers[imap[i]] += segment.shape[-1]  # fast-forward to the next segment boundary
                        continue

            batch_timestamp_tokens: List[torch.Tensor] = [tokens.ge(tokenizers[languages[imap[i]]].timestamp_begin)
                                                          for i, tokens in enumerate(batch_tokens)]
            batch_consecutive = [torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1) for
                                 timestamp_tokens in batch_timestamp_tokens]

            for i, consecutive in enumerate(batch_consecutive):
                if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                    last_slice = 0
                    for current_slice in consecutive:
                        sliced_tokens = batch_tokens[i][last_slice:current_slice]
                        sliced_ts_tokens = batch_finalized_ts_tokens[i][last_slice:current_slice]
                        sliced_ts_logits = batch_ts_logits[i][last_slice:current_slice]
                        sliced_tc = batch_tc[i][last_slice:current_slice]
                        start_timestamp_position = (
                                sliced_tokens[0].item() - tokenizers[languages[imap[i]]].timestamp_begin
                        )
                        end_timestamp_position = (
                                sliced_tokens[-1].item() - tokenizers[languages[imap[i]]].timestamp_begin
                        )

                        word_ts = batch_timestamp_offsets[i] + sliced_ts_tokens * time_precision

                        add_segment(
                            seeker=seekers[imap[i]],
                            offset=batch_timestamp_offsets[i],
                            start=batch_timestamp_offsets[i] + start_timestamp_position * time_precision,
                            end=min(batch_timestamp_offsets[i] + end_timestamp_position * time_precision,
                                    batch_timestamp_offsets[i] + batch_segment_durations[i]),
                            text_tokens=sliced_tokens[1:-1],
                            result=result[i],
                            tokenizer=tokenizers[languages[imap[i]]],
                            segments=all_segments[imap[i]],
                            start_timestamps=word_ts[0].tolist(),
                            end_timestamps=word_ts[-1].tolist(),
                            word_timestamps=word_ts[1:-1],
                            start_ts_logits=sliced_ts_logits[0].tolist(),
                            end_ts_logits=sliced_ts_logits[-1].tolist(),
                            word_ts_logits=sliced_ts_logits[1:-1],
                            tc_logits=sliced_tc[1:-1],
                        )
                        last_slice = current_slice
                    last_timestamp_position = (
                        min(batch_tokens[i][last_slice - 1].item() - tokenizers[languages[imap[i]]].timestamp_begin, batch_segment_max_ts[i])
                    )
                    seekers[imap[i]] += last_timestamp_position * input_stride
                    all_tokens[imap[i]].extend(batch_tokens[i][: last_slice + 1].tolist())
                else:
                    duration = batch_segment_durations[i]
                    timestamps = batch_tokens[i][batch_timestamp_tokens[i].nonzero().flatten()]
                    if len(timestamps) > 0:
                        # no consecutive timestamps but it has a timestamp; use the last one.
                        # single timestamp at the end means no speech after the last timestamp.
                        last_timestamp_position = min(timestamps[-1].item() - tokenizers[languages[imap[i]]].timestamp_begin, batch_segment_max_ts[i])
                        duration = last_timestamp_position * time_precision

                    word_ts = batch_timestamp_offsets[i] + batch_finalized_ts_tokens[i] * time_precision

                    add_segment(
                        seeker=seekers[imap[i]],
                        offset=batch_timestamp_offsets[i],
                        start=batch_timestamp_offsets[i],
                        end=batch_timestamp_offsets[i] + duration,
                        text_tokens=batch_tokens[i],
                        result=results[i],
                        tokenizer=tokenizers[languages[imap[i]]],
                        segments=all_segments[imap[i]],
                        word_timestamps=word_ts,
                        word_ts_logits=batch_ts_logits[i],
                        tc_logits=batch_tc[i],
                    )

                    seekers[imap[i]] += segments[imap[i]].shape[-1]
                    all_tokens[imap[i]].extend(batch_tokens[i].tolist())

                if all_segments[imap[i]]:
                    all_segments[imap[i]][-1]['anchor_point'] = True
                    all_segments[imap[i]][-1]['next_offset'] = float(seekers[imap[i]] * HOP_LENGTH / SAMPLE_RATE)
                if not condition_on_previous_text or result.temperature > 0.5:
                    # do not feed the prompt tokens if a high temperature was used
                    prompt_reset_since[imap[i]] = len(all_tokens[imap[i]])

                update_pbar()

    if len(all_segments[imap[i]]) > 1 and all_segments[imap[i]][-1]['alt_start_timestamps'] is None:
        all_segments[imap[i]][-1]['alt_start_timestamps'] = all_segments[imap[i]][-2]['alt_end_timestamps']

    # # my prepare confidence
    # all_segments = my_prepare_confidence_and_words(tokenizer, all_segments)  # side effect

    if stab:
        all_segments[imap[i]] = stabilize_timestamps(all_segments[imap[i]], top_focus=top_focus)
        add_whole_word_ts(tokenizers[languages[imap[i]]], all_segments[imap[i]],
                          merge_non_space=True,
                          prepend_punctuations=prepend_punctuations,
                          append_punctuations=append_punctuations)
        # if verbose:
        #     print('\nSTABILIZED:')
        #     for seg_ in all_segments:
        #         print(f'[{format_timestamp(seg_["start"])} --> {format_timestamp(seg_["end"])}] "{seg_["text"]}"')
        #         if seg_['word_timestamps']:
        #             ts_str = (f' ->[{format_timestamp(ts_["timestamp"])}] "{ts_["word"].strip()}"' for ts_ in
        #                       seg_['word_timestamps'])
        #             print('\n'.join(ts_str), end='\n\n')
        # import pickle
        # with open('filename.pickle', 'wb') as fh:
        #     pickle.dump(all_segments, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return [dict(text=tokenizers[languages[i]].decode(
        [token for token in all_tokens[i][len(initial_prompt):] if token < tokenizers[languages[i]].eot]),
                 segments=all_segments[i], language=languages[i]) for i in range(len(all_segments))]

def _suppress_ts(ts_logits: Tensor, suppress_ts_mask: Tensor = None):
    if suppress_ts_mask is not None:
        ts_logits[:, suppress_ts_mask] = -np.inf


def _ts_topk(ts_logits: Tensor, k: int, prev_ts: Tensor = None) -> Tensor:
    temp_ts = torch.stack(torch.topk(ts_logits, k, dim=-1), 0).unsqueeze(-2)
    return temp_ts if prev_ts is None else torch.cat([prev_ts, temp_ts], dim=-2)


# modified version of whisper.GreedyDecoder
class GreedyDecoderWordLevel(GreedyDecoder):
    def __init__(self, *args, **kwargs):
        self.ts_num: int = kwargs.pop('ts_num', 10)
        self.suppress_ts_mask: Tensor = kwargs.pop('suppress_ts_mask', None)
        self.timestamp_begin = kwargs.pop('timestamp_begin', 50364)
        super(GreedyDecoderWordLevel, self).__init__(*args, **kwargs)
        self.ts = None

    def _suppress_ts(self, logits: Tensor):
        _suppress_ts(logits[:, self.timestamp_begin:],
                     suppress_ts_mask=self.suppress_ts_mask)

    def update_with_ts(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor, ts: Tensor) -> Tuple[Tensor, bool]:
        self.ts = ts

        self._suppress_ts(logits)

        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed, current_logprobs

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist(), self.ts.transpose(1, 0)[None]


# modified version of whisper.BeamSearchDecoder
class BeamSearchDecoderWordLevel(BeamSearchDecoder):

    def __init__(self, *args, **kwargs):
        self.ts_num: int = kwargs.pop('ts_num', 10)
        self.suppress_ts_mask: Tensor = kwargs.pop('suppress_ts_mask', None)
        self.timestamp_begin = kwargs.pop('timestamp_begin', 50364)
        super(BeamSearchDecoderWordLevel, self).__init__(*args, **kwargs)
        self.ts = None
        self.finished_ts_ls = None

    def reset(self):
        self.finished_sequences = None
        self.finished_ts_ls = None

    def _suppress_ts(self, logits: Tensor, ts_mask_idx: int = None):
        _suppress_ts(logits[:, self.timestamp_begin:],
                     suppress_ts_mask=self.suppress_ts_mask)

    def update_with_ts(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor, ts: Tensor) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        self.ts = ts

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]
            self.finished_ts_ls = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences, finished_ts_ls = [], [], [], []

        self._suppress_ts(logprobs)

        for i in range(n_audio):
            scores, sources, finished, finished_ts = {}, {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                    finished_ts[sequence] = self.ts[:, sources[sequence]]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)
            finished_ts_ls.append(finished_ts)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)
        self.ts = self.ts[:, source_indices]

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished, \
            prev_ts_ls, new_ts_ls in \
                zip(self.finished_sequences, finished_sequences,
                    self.finished_ts_ls, finished_ts_ls):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]
                prev_ts_ls[seq] = new_ts_ls[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates for sequences in self.finished_sequences
        )
        return tokens, completed, sum_logprobs

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        self.ts = self.ts.reshape(self.ts.shape[0], *preceding_tokens.shape[:2], *self.ts.shape[2:])
        sum_logprobs = sum_logprobs.cpu()
        for i, (sequences, ts_) in \
                enumerate(zip(self.finished_sequences, self.finished_ts_ls)):
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    seq_tuple = tuple(sequence)
                    sequences[seq_tuple] = sum_logprobs[i][j].item()
                    ts_[seq_tuple] = self.ts[:, i, j]
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        final_ts: List[List[Tensor]] = [
            list(sequences.values()) for sequences in self.finished_ts_ls
        ]
        return tokens, sum_logprobs, final_ts


class DecodingTaskWordLevel(DecodingTask):

    def __init__(self, *args, **kwargs):
        self.ts_num: int = kwargs.pop('ts_num', None) or 10
        self.alpha: float = kwargs.pop('alpha', None)  # experimental
        self.suppress_ts_mask: Tensor = kwargs.pop('suppress_ts_mask', None)
        self.suppress_word_ts: bool = kwargs.pop('suppress_word_ts', True)
        super(DecodingTaskWordLevel, self).__init__(*args, **kwargs)

        language = self.options.language or "en"
        if type(language) == list:
            for i in range(len(self.initial_tokens)):
                if hasattr(self.decoder[i], 'beam_size'):
                    self.decoder[i] = BeamSearchDecoderWordLevel(self.decoder[i].beam_size,
                                                              self.decoder[i].eot,
                                                              self.inference,
                                                              self.decoder[i].patience,
                                                              ts_num=self.ts_num,
                                                              suppress_ts_mask=self.suppress_ts_mask[i],
                                                              timestamp_begin=self.tokenizers[i].timestamp_begin)
                else:
                    self.decoder[i] = GreedyDecoderWordLevel(self.decoder[i].temperature,
                                                          self.decoder[i].eot,
                                                          ts_num=self.ts_num,
                                                          suppress_ts_mask=self.suppress_ts_mask[i],
                                                          timestamp_begin=self.tokenizers[i].timestamp_begin)
        else:
            if hasattr(self.decoder, 'beam_size'):
                self.decoder = BeamSearchDecoderWordLevel(self.decoder.beam_size,
                                                          self.decoder.eot,
                                                          self.inference,
                                                          self.decoder.patience,
                                                          ts_num=self.ts_num,
                                                          suppress_ts_mask=self.suppress_ts_mask,
                                                          timestamp_begin=self.tokenizer.timestamp_begin)
            else:
                self.decoder = GreedyDecoderWordLevel(self.decoder.temperature,
                                                      self.decoder.eot,
                                                      ts_num=self.ts_num,
                                                      suppress_ts_mask=self.suppress_ts_mask,
                                                      timestamp_begin=self.tokenizer.timestamp_begin)

    # modified version of whisper.DecodingTask._main_loop
    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        if type(self.decoder) == list:
            token_confidences = [[] for i in range(len(self.decoder))]
        else:
            token_confidences = []

        #print("beg", "n_batch", n_batch, "sum_logprobs", dim(sum_logprobs), "no_speech", dim(no_speech_probs), "confidence", dim(token_confidences))

        try:
            for i in range(self.sample_len):
                if self.alpha:
                    logits = self.inference.logits(tokens,
                                                   audio_features * (torch.rand_like(audio_features) * self.alpha + 1))
                else:
                    logits = self.inference.logits(tokens, audio_features)

                if type(self.sot_index) == list:
                    if i == 0 and not any(
                            isinstance(_get_new_attrs(tok, 'no_captions'), type(None)) for tok in self.tokenizers):  # save no_speech_probs
                        probs_at_sot = []
                        no_speech_probs = []

                        btch_logits = logits.view(len(self.sot_index), int(logits.shape[0] / len(self.sot_index)),
                                                  logits.shape[1], logits.shape[2])

                        for i in range(len(self.sot_index)):
                            probs_at_sot.append(btch_logits[i][:, self.sot_index[i]].float().softmax(dim=-1))
                            no_speech_probs.append(probs_at_sot[i][:, _get_new_attrs(self.tokenizers[i], 'no_captions')].tolist())
                else:
                    if i == 0 and _get_new_attrs(self.tokenizer, 'no_captions') is not None:  # save no_speech_probs
                        probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                        no_speech_probs = probs_at_sot[:, _get_new_attrs(self.tokenizer, 'no_captions')].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]


                if type(self.decoder) == list:
                    # reshape logits tensor to decompress the unsqueezed tensor,
                    # (Audio_features, features) -> (number_of_audio, features_per_audio, features)
                    # (12, 40) -> (4, 3, 40) assuming 4 audio files

                    new_logits = logits.view(len(self.decoder), int(logits.shape[0] / len(self.decoder)), logits.shape[1])
                    ts = []
                    for i in range(len(new_logits)):
                        ts_logits = torch.clone(new_logits[i][:, self.tokenizers[i].timestamp_begin:])
                        if self.suppress_word_ts:
                            _suppress_ts(ts_logits, self.suppress_ts_mask[i])
                        ts.append(_ts_topk(ts_logits, k=self.ts_num, prev_ts=self.decoder[i].ts))
                else:
                    ts_logits = torch.clone(logits[:, self.tokenizer.timestamp_begin:])
                    if self.suppress_word_ts:
                        _suppress_ts(ts_logits, self.suppress_ts_mask)
                    ts = _ts_topk(ts_logits, k=self.ts_num, prev_ts=self.decoder.ts)


                if (len(self.logit_filters) > 0) and (type(self.logit_filters[0]) == list):
                    # for batched case
                    for i, logit_filter_group in enumerate(self.logit_filters):
                        for logit_filter in logit_filter_group:
                            logit_filter.apply(logits[i].unsqueeze(0), tokens[i].unsqueeze(0))
                elif len(self.logit_filters) > 0:
                    for logit_filter in self.logit_filters:
                        logit_filter.apply(logits, tokens)

                if type(self.decoder) == list:
                    # handle batched case
                    completed = []
                    new_tokens = []
                    current_logprobs = []

                    btch_tokens = tokens.view(len(self.decoder), int(tokens.shape[0] / len(self.decoder)),
                                              tokens.shape[1])
                    btch_logits = logits.view(len(self.decoder), int(logits.shape[0] / len(self.decoder)),
                                              logits.shape[1])
                    btch_sum_logprobs = sum_logprobs.view(len(self.decoder), int(sum_logprobs.shape[0] / len(self.decoder)))
                    for i in range(len(self.decoder)):
                        # expand the tokens tensor with the selected next tokens
                        token_slice, comp, logprobs = self.decoder[i].update_with_ts(
                            btch_tokens[i],
                            btch_logits[i],
                            btch_sum_logprobs[i],
                            ts[i]
                        )

                        new_tokens.append(token_slice)
                        completed.append(comp)
                        current_logprobs.append(logprobs)
                        token_confidences[i].append((logprobs.tolist()[0], token_slice.tolist()[0][-1]))
                    tokens = torch.cat(new_tokens, dim=0)
                else:
                    # expand the tokens tensor with the selected next tokens
                    tokens, completed, current_logprobs = self.decoder.update_with_ts(tokens, logits, sum_logprobs, ts)

                    # add confidence to the tokens
                    token_confidences.append((current_logprobs.tolist()[0], tokens.tolist()[0][-1]))

                if type(completed) == list:
                    completed = all(completed)
                    if completed or tokens.shape[-1] > self.n_ctx:
                        break
                else:
                    if completed or tokens.shape[-1] > self.n_ctx:
                        break

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        # print("end", "n_batch", n_batch, "sum_logprobs", dim(sum_logprobs), "no_speech", dim(no_speech_probs),
        #       "confidence", dim(token_confidences))
        return tokens, sum_logprobs, no_speech_probs, token_confidences

    # modified version of whisper.DecodingTask.run
    @torch.no_grad()
    def run(self, mel: Tensor) \
        -> Union[List[DecodingResult], Tuple[List[DecodingResult], List[List[int]], List[int]]]:
        if type(self.decoder) == list:
            _ = [decoder.reset() for decoder in self.decoder]
            tokenizer: List[Tokenizer] = self.tokenizers
        else:
            self.decoder.reset()
            tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]
        self.n_audio = n_audio
        audio_features: Tensor = self._get_audio_features(mel.squeeze(1))  # encoder forward pass
        if type(self.initial_tokens) == list:
            # if batched, then stack prompts together in batch dimension
            tokens = [list(token) for token in self.initial_tokens]
            min_len = min([len(t) for t in tokens])
            tokens = [t[:min_len] for t in tokens]
            tokens: Tensor = torch.tensor(tokens)
        else:
            tokens: Tensor = torch.tensor([self.initial_tokens]).expand(n_audio, -1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(audio_features=features, language=language, language_probs=probs)
                for features, language, probs in zip(audio_features, languages, language_probs)
            ]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        #print("before main_loop", "n_audio", n_audio, "decoders", dim(self.decoder), "audio_features", dim(audio_features), "tokens", dim(tokens))

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs, tc = self._main_loop(audio_features, tokens)

        #print("after main_loop", "n_audio", n_audio, "audio_features", dim(audio_features), "sum_logprobs", dim(sum_logprobs), "no_speech_probs_dim", dim(no_speech_probs), "tokens", dim(tokens), "tc", dim(tc))

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]

        if n_audio <= 1:
            no_speech_probs = no_speech_probs[:: self.n_group]

        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        if type(self.decoder) == list:
            new_tokens = []
            sum_logprobs_new = []
            ts_new = []
            for i in range(len(self.decoder)):
                # get the final candidates for each group, and slice between the first sampled token and EOT
                token_slice, sum_logprob_slice, ts_slice = self.decoder[i].finalize(tokens[i].unsqueeze(0),
                                                                          sum_logprobs[i].unsqueeze(0))
                new_tokens.append(token_slice)
                sum_logprobs_new.append(sum_logprob_slice[0])
                ts_new.append(ts_slice[0])
            tokens = torch.cat(new_tokens, dim=0)
            sum_logprobs = sum_logprobs_new
            ts = ts_new
        else:
            # get the final candidates for each group, and slice between the first sampled token and EOT
            tokens, sum_logprobs, ts = self.decoder.finalize(tokens, sum_logprobs)

        if type(self.sample_begin) == list:
            tokens: List[List[Tensor]] = [
                [t[self.sample_begin[i]: (t == tokenizer[i].eot).nonzero()[0, 0]] for t in s] for i, s in
                enumerate(tokens)
            ]
            # TODO Adapt this to batch inference
            ts: List[List[Tensor]] = [
                [t[:, :tokens[i][j].shape[-1]] for j, t in enumerate(s)] for i, s in enumerate(ts)
            ]
        else:
            tokens: List[List[Tensor]] = [
                [t[self.sample_begin: (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
            ]
            ts: List[List[Tensor]] = [
                [t[:, :tokens[i][j].shape[-1]] for j, t in enumerate(s)] for i, s in enumerate(ts)
            ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        ts: List[List[int]] = [t[i].tolist() for i, t in zip(selected, ts)]
        if type(tokenizer) == list:
            texts: List[str] = [tokenizer[i].decode(t).strip() for i, t in enumerate(tokens)]
        else:
            texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                   audio_features=features,
                   language=language,
                   tokens=tokens,
                   text=text,
                   avg_logprob=avg_logprob,
                   **(dict(no_caption_prob=no_speech_prob) if hasattr(DecodingResult, 'no_caption_prob') else dict(
                       no_speech_prob=no_speech_prob)),
                   temperature=self.options.temperature,
                   compression_ratio=compression_ratio(text),
               )
               for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
        ], ts, tc  # my: tc


# modified version of whisper.decoding.decode
@torch.no_grad()
def decode(model: "Whisper", mel: Tensor, options: DecodingOptions = DecodingOptions(),
                      ts_num: int = None, alpha: float = None, suppress_ts_mask: Tensor = None,
                      suppress_word_ts=False) -> \
        Union[DecodingResult, List[DecodingResult], tuple]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        The Whisper model modified instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    ts_num: int
        Number of additional top timestamp predictions to save for each word for postprocessing stabilization (default: 10)

    alpha: float
        Amount of noise to add to audio to produce slightly difference results.
        audio_features *= torch.rand_like(audio_features) * alpha + 1

    suppress_ts_mask: (list, Tensor)
        Mask suppress to timestamp token(s) for decoding

    suppress_word_ts: bool
        Use suppress_ts_mask to suppress timestamp tokens of words

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    result, ts, tc = DecodingTaskWordLevel(model, options,
                                       ts_num=ts_num,
                                       alpha=alpha,
                                       suppress_ts_mask=suppress_ts_mask,
                                       suppress_word_ts=suppress_word_ts).run(mel)

    if single:
        result = result[0]
        ts_tokens = ts[0][1]
        ts_logits = ts[0][0]
    else:
        ts_tokens = [ts_[1] for ts_ in ts]
        ts_logits = [ts_[0] for ts_ in ts]

    return result, ts_tokens, ts_logits, tc # my: tc


def modify_model(model: Whisper):
    """
    Modifies model instance by:
        -replacing model.decode with the new decode method
        -replacing model.transcribe with the new transcribe method
    """
    model.decode = MethodType(decode, model)
    model.transcribe = MethodType(transcribe, model)


# modified version of whisper.load_model
def load_model(name: str, device: Optional[Union[str, torch.device]] = None,
               download_root: str = None, in_memory: bool = False) -> Whisper:
    """
     Load a modified Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """
    model = load_ori_model(name, device=device, download_root=download_root, in_memory=in_memory)
    modify_model(model)
    return model
