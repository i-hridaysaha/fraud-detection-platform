# Stage 10: release

Nine stages produced the artifacts; this one writes the front door. The README, a methodology
document, an architecture diagram, a licence, a changelog, a verification pass over every number
in the repository, and the release tag. Nothing here fits a model or reads the data. Everything
here is the kind of writing that goes wrong quietly, so most of the stage was spent building the
checks that make it go wrong loudly instead.

## What changed my mind

**The README is not written by hand.** The first draft was, and it took a test to show me why
that was a mistake. The label latency table had one cell retyped from memory, `-0.0706` where
the artifact says `-0.0613`, in a row of fourteen numbers I had read minutes earlier. The rule
for this repo is "never retype a number from memory into prose", and it turns out to be a rule
about tables as much as sentences. `scripts/readme_numbers.py` now prints the three tables and
every inline number; the README carries its output pasted; `tests/test_readme.py` fails if the
two differ by a character. The verification script found one more of the same kind in
`docs/eda.md`, `0.2330` for a top-value share the artifact holds as `0.23285`. Two last-place
errors in a repository that had a verification table at every stage close is a small count, and
neither would have been found by reading.

**A number I sourced wrongly while writing the verification document.** The residue table
explains `244,197` as a derived difference. My first explanation named two counts that do not
subtract to it, and the second named the right counts and the wrong file. The scan of the
verification document against the artifacts caught the first; reading the sentence in
`docs/eda.md` that the number comes from caught the second. The document that accounts for the
numbers is subject to the same rule as the numbers, and I nearly wrote it as if it were not.

**The README is longer than the standard wants.** The target is 120 to 200 lines; the README is
241. The three results tables cost 28 lines and the scope statement 10, and after two trimming
passes the prose under them is as short as I can make it without dropping a table. ADR 0041
records this as a deliberate deviation rather than drift: the first table is at line 50, and
the standard's own ceiling clause allows 200 to 300 when the top carries everything. I would
rather a reviewer see the leakage table than a shorter file.

## Dead ends

- The first architecture diagram had labels sitting on the boxes they belonged beside, because
  a 50-pixel gap does not hold "get_features, then commit" at 11 points. The gaps are 80 now
  and the two-way arrows carry their labels in the channels above and below the row.
- The first draft of the label-latency paragraph said the target encodings and the D columns
  carry the immature regime. Stage 7's own artifact says no single block is the door and which
  columns carry it is not established. The sentence now says that.
- The Model section said the C block's construction is "not published". What is established is
  narrower: nothing this repository could fetch states it. The sentence now says that.
- The ADR numbers in the stage brief, 0038 and 0039, were both taken, as in every stage since 4.
  This stage's are 0040 and 0041.

## Surprises

- All 24 committed figures re-render byte for byte from the artifacts. I expected at least one
  to differ on a font or a version string.
- The ablation figure, the one whose numbers decided the shipped stack, was the only model
  comparison without an interval on it. The rule reads the paired interval per seed; the figure
  drew the seed points and a mean. It has the intervals now.
- The whole-repository numeric scan runs in under a second and matches 5,209 of 5,271 tokens
  against an artifact leaf. The 62 it does not match are all derived, external, assumed or
  session-measured, and every one had already been listed as such in an earlier stage's
  verification table. The stages were more careful than I remembered.

## Outside this repository

The case study on the site was written against an earlier build of this platform and quotes
numbers this repository does not produce (a searched XGBoost at 0.5274 on 308 features under
class weighting; here the default configuration ships at 0.5451 on 186 columns with no
reweighting, and the search lost). The README links the case study as the narrative and quotes
nothing from it. Bringing the case study into line with `reports/` is the next piece of work and
is not part of this release.

## Not established

- Whether `make reproduce` gives the same scores on a machine that is not this laptop. The
  clean-clone run in the build log is on the same machine as every other number here.
- What the README reads like to the reader it is written for. The standard's 30-seconds-to-five-
  minutes reader has not read it.
