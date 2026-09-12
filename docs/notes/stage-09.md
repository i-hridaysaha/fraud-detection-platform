# Stage 9: monitoring and the lifecycle

Stage 8 put the model behind an endpoint. This stage watches it and replaces it. The two things
that can go wrong have different clocks: the inputs can move the day they move, and the model's
accuracy can only be read once the labels for a batch have arrived, which on this kind of data is
weeks later. The design keeps them apart: a drift job that reads no labels, a decay job that
reads only mature ones, a retrain decision that takes decay as its primary trigger and sustained
drift as an early one, a challenger through the same pipeline, a gate with a margin above the
stage 6 noise band, and a rollback.

## Expectations, written before the adversarial validation ran

This section was committed before `scripts/adversarial_validation.py` existed. The stage brief
asks which features a train-versus-test classifier finds most discriminative; here is what I
expected, so that the artifact can disagree with me on the record.

**The classifier separates the windows easily.** Out-of-fold ROC-AUC above 0.90 on the shipped
model's 186 columns. Stage 2 measured 22 raw columns above PSI 0.25 between train and test, and
the loudest were clocks: `card_start_day` at 1.841 is derived from `TransactionDT` and moves by
construction, `id_31` at 1.500 has 19 browser versions test has and train never saw. A classifier
with those to hand does not need the rest.

**The top ten by gain, in roughly this order.** The stage 3 encodings of the identifiers first:
`card1_te`, `card1_freq`, `addr1_te`, `addr1_freq`, `card2_te`. Not because the cards changed
but because of how the encodings are built: a train row reads the target table at its own lagged
day, so early train rows read a near-empty table and fall to the prior or to null, while every
test row reads the table frozen at the end of the training window. The train distribution of a
`_te` column is a smear across the window; the test distribution is one snapshot. A classifier
told to tell the windows apart will find that first, and it says nothing about the cards. Then
the timedelta columns that count days since something (`D15`, `D10`, `D4`, `D11`), then the
identity fields stage 2 flagged (`id_13`, `id_31_norm_cat`, `id_19`, `id_20`), then the first
V block mean (`vblockmean_V1`, from V1 to V11 at PSI 0.29 to 0.30 in stage 2).

**The raw identifiers themselves, `card1` and `addr1`, in the top thirty but not the top ten.**
As integer codes they carry population turnover (0.6598 of test rows sit on entities train never
saw, stage 1) but a tree on a code has to carve it into ranges, and the encodings above carry the
same fact more cleanly.

**The character of the shift: population turnover, not a change in what fraud looks like.** If
the top of the ranking is identifiers, their encodings and clocks, then the classifier is
recognising *who* and *when*, not *how*. That agrees with stage 2, which found the target rate
non-stationary (Mann-Kendall tau 0.399, weekly range 0.0207 to 0.0507) but the intersection
between the 22 high-PSI columns and the 59 time-inconsistent columns empty: the columns that
drift are not the columns whose relationship with the label inverted.

**Dropping the identifiers, their encodings and the D columns leaves a separable problem.**
Out-of-fold AUC still 0.75 to 0.85, carried by `id_31_norm_cat`, the device fields and the V
block means. Browser versions are a clock that no entity key removes.

**What would change my reading.** If the top features were the C columns, the amount, or the
V block means with the identifiers well down the list, the windows would differ in transaction
behaviour rather than in population, and the stage document would have to say the drift is
closer to concept drift than the PSI baseline suggested.
