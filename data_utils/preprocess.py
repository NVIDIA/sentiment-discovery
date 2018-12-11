import os
import re
import html
import unicodedata

import unidecode
import torch

try:
    import emoji
except:
    print(Warning("emoji import unavailable"))


HTML_CLEANER_REGEX = re.compile('<.*?>')

def clean_html(text):
    """remove html div tags"""
    text = str(text)
    return re.sub(HTML_CLEANER_REGEX, ' ', text)

def binarize_labels(labels, hard=True):
    """If hard, binarizes labels to values of 0 & 1. If soft thresholds labels to [0,1] range."""
    labels = np.array(labels)
    min_label = min(labels)
    label_range = max(labels)-min_label
    if label_range == 0:
        return labels
    labels = (labels-min_label)/label_range
    if hard:
        labels = (labels > .5).astype(int)
    return labels

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def process_str(text, front_pad='\n ', end_pad=' ', maxlen=None, clean_markup=True,
                clean_unicode=True, encode='utf-8', limit_repeats=3):
    """
    Processes utf-8 encoded text according to the criterion specified in seciton 4 of https://arxiv.org/pdf/1704.01444.pdf (Radford et al).
    We use unidecode to clean unicode text into ascii readable text
    """
    if clean_markup:
        text = clean_html(text)

    if clean_unicode:
        text = unidecode.unidecode(text)

    text = html.unescape(text)
    text = text.split()
    if maxlen is not None:
        len2use = maxlen-len(front_pad)-len(end_pad)
        text = text[:len2use]

    if limit_repeats > 0:
        remove_repeats(text, limit_repeats, join=False)

    text = front_pad+(" ".join(text))+end_pad

    if encode is not None:
        text = text.encode(encoding=encode)
        text = ''.join(chr(c) for c in text)

    return text

def remove_repeats(string, n, join=True):
    count = 0
    output = []
    last = ''
    for c in string:
        if c == last:
            count = count + 1
        else:
            count = 0
            last = c
        if count < n:
            output.append(c)
    if join:
        return "".join(output)
    return output

def tokenize_str_batch(strings, rtn_maxlen=True, process=True, maxlen=None, ids=False, rtn_processed=True):
    """
    Tokenizes a list of strings into a ByteTensor
    Args:
        strings: List of utf-8 encoded strings to tokenize into ByteTensor form
        rtn_maxlen: Boolean with functionality specified in Returns.lens
    Returns:
        batch_tensor: ByteTensor of shape `[len(strings),maxlen_of_strings]`
        lens: Length of each string in strings after being preprocessed with `preprocess` (useful for
            dynamic length rnns). If `rtn_maxlen` is `True` then max(lens) is returned instead.
    """
    if process:
        processed_strings = [process_str(x, maxlen=maxlen) for x in strings]
    else:
        processed_strings = [x.encode('utf-8', 'replace') for x in strings]

    tensor_type = torch.ByteTensor

    lens, batch_tensor = batch_tokens(processed_strings, tensor_type)
    maxlen = max(lens)
    rounded_maxlen = max(lens)

    rtn = []
    if not rtn_maxlen and rtn_maxlen is not None:
        rtn = [batch_tensor, lens]
    elif rtn_maxlen is None:
        rtn = [batch_tensor]
    else:
        rtn = [batch_tensor, rounded_maxlen]
    if rtn_processed:
        rtn += [processed_strings]
    return tuple(rtn)

def batch_tokens(token_lists, tensor_type=torch.LongTensor, fill_value=0):
    lens = list(map(len, token_lists))
    batch_tensor = fill_value * torch.ones(len(lens), max(lens)).type(tensor_type)
    for i, string in enumerate(token_lists):
        _tokenize_str(string, tensor_type, batch_tensor[i])
    return batch_tensor, lens

def _tokenize_str(data, tensor_type, char_tensor=None):
    """
    Parses a utf-8 encoded string and assigns to ByteTensor char_tensor.
    If no char_tensor is provide one is created.
    Typically used internally by `tokenize_str_batch`.
    """
    if char_tensor is None:
        if isinstance(data, str):
            # data could either be a string or a list of ids.
            data = data.encode()
        char_tensor = tensor_type(len(data))
    for i, char in enumerate(data):
        char_tensor[i] = char

EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
BAD_HASHTAG_LOGIC = re.compile(r'(\S+)!!')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

def process_tweet(s, save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    # NOTE: will sometimes need to use Windows encoding here, depending on how CSV is generated.
    # All results saved in UTF-8
    # TODO: Try to get input data in UTF-8 and don't let it touch windows (Excel). That loses emoji, among other things

    # Clean up the text before tokenizing.
    # Why is this necessary?
    # Unsupervised training (and tokenization) is usually on clean, unformatted text.
    # Supervised training/classification may be on tweets -- with non-ASCII, hashtags, emoji, URLs.
    # Not obvious what to do. Two options:
    # A. Rewrite formatting to something in ASCII, then finetune.
    # B. Remove all formatting, keep only the text.
    if save_text_formatting:
        s = re.sub(r'https\S+', r'xxxx', str(s))
    else:
        s = re.sub(r'https\S+', r' ', str(s))
        s = re.sub(r'x{3,5}', r' ', str(s))
    
    # Try to rewrite all non-ASCII if known printable equivalent
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)

    # Rewrite emoji as words? Need to import a function for that.
    # If no formatting, just get the raw words -- else save formatting so model can "learn" emoji
    # TODO: Debug to show differences?
    if save_text_formatting:
        s = emoji.demojize(s)
    elif keep_emoji:
        s = emoji.demojize(s)
        # Transliterating directly is ineffective w/o emoji training. Try to shorten & fix
        s = s.replace('face_with', '')
        s = s.replace('face_', '')
        s = s.replace('_face', '')
        # remove emjoi formatting (: and _)
        # TODO: A/B test -- better to put emoji in parens, or just print to screen?
        #s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' (\1) ', s)
        s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)
        # TODO -- better to replace '_' within the emoji only...
        s = s.replace('(_', '(')
        s = s.replace('_', ' ')

    # Remove all non-printable and non-ASCII characters, including unparsed emoji
    s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)
    # NOTE: We can't use "remove accents" as long as foreign text and emoji gets parsed as characters. Better to delete it.
    # Replace accents with non-accented English letter, if possible.
    # WARNING: Will try to parse corrupted text... (as aAAAa_A)
    s = remove_accents(s)
    # Rewrite or remove hashtag symbols -- important text, but not included in ASCII unsupervised set
    if save_text_formatting:
        s = re.sub(HASHTAG_BEFORE, r'\1!!', s)
    else:
        s = re.sub(HASHTAG_BEFORE, r'\1', s)
        # bad logic in case ^^ done already
        s = re.sub(BAD_HASHTAG_LOGIC, r'\1', s)
    # Keep user names -- or delete them if not saving formatting.
    # NOTE: This is not an obvious choice -- we could also treat mentions vs replies differently. Or we could sub xxx for user name
    # The question is, does name in the @mention matter for category prediction? For emotion, it should not, most likely.
    if save_text_formatting:
        # TODO -- should we keep but anonymize mentions? Same as we rewrite URLs.
        pass
    else:
        # If removing formatting, either remove all mentions, or just the @ sign.
        if keep_usernames:
            # quick cleanup extra spaces
            s = ' '.join(s.split())

            # If keep usernames, *still* remove leading and trailing names in @ mentions (or tail mentions)
            # Why? These are not part of the text -- and should not change sentiment
            s = re.sub(LEADING_NAMES, r' ', s)
            s = re.sub(TAIL_NAMES, r' ', s)

            # Keep remaining mentions, as in "what I like about @nvidia drivers"
            s = re.sub(FIND_MENTIONS, r'\1', s)
        else:
            s = re.sub(FIND_MENTIONS, r' ', s)
    #s = re.sub(re.compile(r'@(\S+)'), r'@', s)
    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    # Final cleanup -- remove extra spaces created by rewrite.
    s = ' '.join(s.split())
    return s
