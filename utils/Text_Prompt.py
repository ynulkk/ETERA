# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
import clip
import numpy as np

def eeg_text_prompt(classed_data): # 0.1shot sub6:87.26%
    text_aug = [f"{{}}",
                f"The human is {{}}",
                f"The video makes the human feel {{}}",
                f"A video of {{}} emotion",
                f"Look, the human is {{}}",
                f"Playing a kind of emotion, {{}}",
                f"Doing a kind of emotion, {{}}",
                f"Does this video convey {{}} emotion?",
                f"What emotion does this video convey: {{}}?",
                f"Identify the emotion in this video: {{}}",
                f"Categorize this video into {{}} emotion",
                f"Can you recognize the emotion of {{}}?",
                f"The human feels {{}} now",
                f"The human looks {{}} about the video",
                f"{{}}, a kind of emotion",
                f"{{}} this is an emotion",
                ]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in classed_data])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict

def eeg_text_prompt1(classed_data): # 0.1shot sub6:87.26%
    text_aug = [f"{{}}",
                f"The human is {{}}",
                f"The video makes the human feel {{}}",
                f"A video of {{}} emotion",
                f"Look, the human is {{}}",
                f"Playing a kind of emotion, {{}}",
                f"Doing a kind of emotion, {{}}",
                f"Does this video convey {{}} emotion?",
                f"What emotion does this video convey: {{}}?",
                f"Identify the emotion in this video: {{}}",
                f"Categorize this video into {{}} emotion",
                f"Can you recognize the emotion of {{}}?",
                f"The human feels {{}} now",
                f"The human looks {{}} about the video",
                f"{{}}, a kind of emotion",
                f"{{}} this is an emotion",
                ]
    text_dict = {}
    num_text_aug = len(text_aug) * len(classed_data)

    for ii in range(len(text_aug)):
        text_dict[ii] = []

    for ii, txt in enumerate(text_aug):
        for jj, classed_data_jj in enumerate(classed_data):
            tokens = torch.cat([clip.tokenize(txt.format(c)) for c in classed_data_jj])
            text_dict[ii].append(tokens)

    classes = torch.cat([torch.cat(text_dict[ii]) for ii in text_dict])

    return classes, num_text_aug, text_dict

def eeg_text_prompt2(classed_data): # 0.1shot sub6:87.26%
    text_aug = [f"{{}}",
                f"{{}} emotion",
                f"the human is {{}}",
                f"the human looks {{}}",
                f"the human seems {{}}",
                f"the human appears {{}}",
                f"the video makes the human feel {{}}",
                f"the movie makes the human feel {{}}",
                f"the film makes the human feel {{}}",
                f"a video of {{}} emotion",
                f"a movie of {{}} emotion",
                f"a film of {{}} emotion",
                f"the video evokes {{}} emotion",
                f"the movie evokes {{}} emotion",
                f"the film evokes {{}} emotion",
                f"the video inspires {{}} emotion",
                f"the movie inspires {{}} emotion",
                f"the film inspires {{}} emotion",
                f"Playing a video of {{}} emotion",
                f"Playing a movie of {{}} emotion",
                f"Playing a film of {{}} emotion",
                f"Look, the human is {{}}",
                f"Playing a kind of emotion, {{}}",
                f"Doing a kind of emotion, {{}}",
                f"Playing an emotion of {{}}",
                f"video classification of {{}} emotion",
                f"movie classification of {{}} emotion",
                f"film classification of {{}} emotion",
                f"Does this video convey {{}} emotion?",
                f"Does this movie convey {{}} emotion?",
                f"Does this film convey {{}} emotion?",
                f"What emotion does this video convey: {{}}?",
                f"What emotion does this movie convey: {{}}?",
                f"What emotion does this film convey: {{}}?",
                f"Identify the emotion in this video: {{}}",
                f"Identify the emotion in this movie: {{}}",
                f"Identify the emotion in this film: {{}}",
                f"Categorize this video into {{}} emotion",
                f"Categorize this movie into {{}} emotion",
                f"Categorize this film into {{}} emotion",
                f"Can you recognize the emotion of {{}}?",
                f"the human feels {{}} now",
                f"the human looks {{}} about the video",
                f"the human looks {{}} about the movie",
                f"the human looks {{}} about the film",
                f"{{}}, a kind of emotion",
                f"{{}}, a video of emotion",
                f"{{}}, a movie of emotion",
                f"{{}}, a film of emotion",
                f"{{}} this is an emotion",
                ]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in classed_data])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict

def eeg_text_prompt3(classed_data):
    text_aug = [f"{{}}",
                f"{{}} emotion",
                f"the human is {{}}",
                f"the human looks {{}}",
                f"the human seems {{}}",
                f"the human appears {{}}",
                f"the video makes the human feel {{}}",
                f"the movie makes the human feel {{}}",
                f"the film makes the human feel {{}}",
                f"a video of {{}} emotion",
                f"a movie of {{}} emotion",
                f"a film of {{}} emotion",
                f"the video evokes {{}} emotion",
                f"the movie evokes {{}} emotion",
                f"the film evokes {{}} emotion",
                f"the video inspires {{}} emotion",
                f"the movie inspires {{}} emotion",
                f"the film inspires {{}} emotion",
                f"Playing a video of {{}} emotion",
                f"Playing a movie of {{}} emotion",
                f"Playing a film of {{}} emotion",
                f"Look, the human is {{}}",
                f"Playing a kind of emotion, {{}}",
                f"Doing a kind of emotion, {{}}",
                f"Playing an emotion of {{}}",
                f"video classification of {{}} emotion",
                f"movie classification of {{}} emotion",
                f"film classification of {{}} emotion",
                f"Does this video convey {{}} emotion?",
                f"Does this movie convey {{}} emotion?",
                f"Does this film convey {{}} emotion?",
                f"What emotion does this video convey: {{}}?",
                f"What emotion does this movie convey: {{}}?",
                f"What emotion does this film convey: {{}}?",
                f"Identify the emotion in this video: {{}}",
                f"Identify the emotion in this movie: {{}}",
                f"Identify the emotion in this film: {{}}",
                f"Categorize this video into {{}} emotion",
                f"Categorize this movie into {{}} emotion",
                f"Categorize this film into {{}} emotion",
                f"Can you recognize the emotion of {{}}?",
                f"the human feels {{}} now",
                f"the human looks {{}} about the video",
                f"the human looks {{}} about the movie",
                f"the human looks {{}} about the film",
                f"{{}}, a kind of emotion",
                f"{{}}, a video of emotion",
                f"{{}}, a movie of emotion",
                f"{{}}, a film of emotion",
                f"{{}} this is an emotion",
                ]
    text_dict = {}
    num_text_aug = len(text_aug) * len(classed_data)

    for ii in range(len(text_aug)):
        text_dict[ii] = []

    for ii, txt in enumerate(text_aug):
        for jj, classed_data_jj in enumerate(classed_data):
            tokens = torch.cat([clip.tokenize(txt.format(c)) for c in classed_data_jj])
            text_dict[ii].append(tokens)

    classes = torch.cat([torch.cat(text_dict[ii]) for ii in text_dict])

    return classes, num_text_aug, text_dict

if __name__ == '__main__':
    data_classes = [['positive', 'neutral', 'negative'], ["happy", "calm", "sad"], ['pleased', 'peaceful', 'unhappy']] # ["neutral", "sad", "fear", "happy"]  #["positive", "neutral", "negative "]
    classes, num_text_aug, text_dict = eeg_text_prompt(data_classes)
    #print(classes)