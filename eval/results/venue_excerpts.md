# Venue excerpts benchmark (real recording, blind 7-day search)

Reference: ELEXON (15 s cadence), 2026-11-05T00:00:00Z to 2026-11-12T00:00:00Z. Rule: answer only if score >= 0.8, margin over anywhere else >= 0.05, halves agree within 10 s. Correct = within +-5 s of the camera clock.

```
            excerpts  answered  correct  wrong  top1_correct  median_abs_error_s
length_min                                                                      
2                 11         0        0      0             0            164358.0
3                 11         0        0      0             2            143028.0
5                  9         1        1      0             8                 2.0
8                  7         2        2      0             7                 1.0
12                 5         5        5      0             5                 1.0
```

Totals: 8/43 answered, 8 correct, 0 wrong. Excerpts overlap (step 90 s) and all come from ONE recording.