# HIPPOCAMPAL OSCILLATORY PATTERNS DURING ASSOCIATE RECOGNITION MEMORY

The goal of investigating hippocampal oscillatory patterns to test predictions of dual process theory motivated us to apply the Associate Recognition (AR) paradigm to intracranial EEG patients with depth electrodes inserted into both cerebral hemispheres, including both the anterior and posterior hippocampus simultaneously. Twenty subjects with medication-resistant epilepsy who underwent stereo-electroencephalography (sEEG) surgery with the goal of identifying their ictal onset region(s) performed AR paradigm during their monitoring period at UT Southwestern. Overall, there were 69 anterior and 41 posterior hippocampal electrodes included in the dataset. The subjects view series of word pairs during the encoding (study) phase. During the retrieval (test) phase, the word pairs were shown as intact (studied together), rearranged (both words were studies but on different trials), and new (neither word was studied) for two seconds, then received a cue to respond their decision whether the word pair was intact or rearranged or new. The sEEG signal was sampled at 1 kHz during acquisition and down sampled at 500 Hz offline for processing. Line noise was notch filtered and a kurtosis algorithm (with threshold of 4) was used to exclude abnormal events and interictal activity. The power and phase values (58 frequencies and 900 time steps) were extracted from 1800 ms time windows following the appearance of the study item using Morlet wavelets. 

We aim to:

1)Classify word pairs as either recollected or recognized using encoding signal and compare performance to the classification using retrieval signal.

2)Determine which features are most critical for classification.

3)Use classifier trained on encoding data to classify recognition versus recollection using retrieval data (measure of reinstatement of information).
Classifier can be trained across subjects/electrodes (including anterior and posterior or hemisphere) to increase the number of trials available for classification. 

Team Leads: Bradley Lega, MD and Srinivas Kota, PhD, Neurosurgery, https://www.utsouthwestern.edu/labs/tcm/ 
