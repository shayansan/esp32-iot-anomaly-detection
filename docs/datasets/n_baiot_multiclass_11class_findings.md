# N-BaIoT 11-Class Diagnostic Findings

## Experiment

The initial multiclass experiment used the Danmini Doorbell
device and 11 balanced classes:

- Benign
- 5 Gafgyt/BASHLITE subtypes
- 5 Mirai subtypes

The same 88,000 selected samples were evaluated using two
different split protocols.

## Protocol A: Source-Range-Separated

Samples were taken from separated regions of each source CSV.

Results for the 115-feature MLP:

- Validation accuracy: 58.84%
- Test accuracy: 54.83%

The model showed strong degradation between training and
validation data.

Drift analysis confirmed substantial source-range distribution
shift for several classes.

Examples of validation centroid L2 shift:

- Gafgyt Combo: 14.64
- Mirai ACK: 10.84
- Mirai SYN: 12.55
- Mirai UDP: 17.16
- Mirai UDPplain: 9.87

Some classes had very small shift:

- Gafgyt TCP: 0.0158
- Gafgyt UDP: 0.0245

## Protocol B: Mixed Stratified Diagnostic

The same 88,000 samples were mixed and stratified into new
training, validation, and test sets.

The model architecture was unchanged.

Results:

- Validation accuracy: approximately 89.75%
- Test accuracy: approximately 89.85%

This large increase indicates that source-range distribution
shift is a major factor in the poor performance of Protocol A.

Protocol B is considered a diagnostic IID-style baseline and not
the primary generalization estimate.

## Exact-Vector Duplication

Additional analysis showed severe duplication in some classes.

For Gafgyt TCP:

- 5,600 mixed-training samples
- only 9 unique feature vectors

For Gafgyt UDP:

- 5,600 mixed-training samples
- only 10 unique feature vectors

Exact feature-vector coverage from training was approximately
100% for Gafgyt TCP and UDP in validation and test.

The source-range-separated protocol also contained substantial
exact-vector reuse for these classes, so it should be described
as source-range-separated rather than fully leakage-free.

## Cross-Label Feature Ambiguity

Across the selected 8,000 TCP and 8,000 UDP samples:

- Total samples: 16,000
- Unique feature vectors: 20
- Feature vectors appearing under both labels: 2
- Samples belonging to conflicting vectors: 8,087

The largest conflicting vector occurred as:

- TCP: 96 samples
- UDP: 7,989 samples

A second conflicting vector occurred once under each label.

Therefore, 97 samples are unavoidably ambiguous for any
deterministic classifier using only these 115 features on this
sample pool.

The corresponding exact-vector majority-label ceiling is
approximately 99.39%.

An ExtraTrees diagnostic classifier achieved approximately
99.4% TCP-vs-UDP accuracy, which is very close to this ceiling.
However, because almost all TCP/UDP validation and test vectors
were already present in training, this result should not be
interpreted as strong unseen-pattern generalization.

## Conclusion

The 11-class experiment identified three separate difficulties:

1. Source-range distribution shift.
2. Extreme duplicate concentration in some attack subtypes.
3. Exact feature-vector ambiguity across subtype labels.

For this reason, the 11-class experiment is retained as a
diagnostic study rather than used directly as the next ESP32
deployment target.

The next multiclass experiment will use attack-family labels:

- Benign
- Gafgyt/BASHLITE
- Mirai

This preserves a meaningful multiclass intrusion-detection task
while avoiding artificial conflicts between subtypes that share
nearly identical or identical feature representations.