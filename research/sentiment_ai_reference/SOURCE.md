# Source of these labels: AI reference, not human labels

The 20 videos in `videos.csv` were labeled with an AI tool (GPT or MAI; the exact model and
version are not recorded), from the transcripts exported by `quant sentiment gold-export`,
on 2026-10-13. A person reviewed the format and fixed the coded values; the opinions
themselves were not checked against the videos by a person.

What this means:

- It is a **reference from a different model**, not ground truth. Scores against it measure
  **agreement** between the extractor and this model, not accuracy.
- It is not from Claude or Gemini (the two extraction backends being compared), so it does
  not simply echo either of them.
- Where the extractor and this reference disagree, either one may be wrong; only a person
  reading the passage can say which.

Mechanical fixes after labeling (2026-10-13):

- reasons moved from `condition` to `note` (`reason: ...`); times set to the first mention
  where they were 00:00 / 00:20; rows for BE and MU without a mention removed
  (backup of the first version: `data/sentiment_gold/opinions.backup-2026-10-13.csv`);
- `LGUewAMj5H8` AMD: action BUY -> NONE (a recommendation, not the creator's own trade);
- `chOchL89sw8` CEG: time 16:38 (after the end of the video) -> 06:12, where CEG is discussed.
