# The demo case: what is real and how it was made

The car-park alibi is a reconstruction of a real kind of dispute. In UK Financial Ombudsman decision [DRN4869503](https://www.financial-ombudsman.org.uk/decision/DRN4869503.pdf), a driver's own CCTV showed his car at home at the time of an alleged collision, and the insurer decided his CCTV clock might be wrong. The real footage from that case is not public, so we rebuilt the situation.

## What the agent is given

`data/media/carpark_cam3.mp4`: six minutes of car-park footage with a recorder overlay reading 20:33:00 on 17 September 2026. The case file states that time and the alleged collision time (20:36). Nothing tells the agent when the clip was really recorded.

## How the clip was made

- **Footage:** licensed stock footage of an underground car park, with a recorder-style timestamp and camera label added.
- **Flicker:** the mains lighting flicker is simulated with the model in `tools/flicker_sim/banding.py`:
  - each light fitting sits on one of three mains phases, so bands shift between lighting zones;
  - the lamp waveform is a rectified sine with phosphor afterglow, integrated over the camera's exposure time;
  - each sensor row is exposed at its own instant (rolling shutter), which turns flicker into bands;
  - steady light sources such as headlights are left unmodulated;
  - sensor noise and H.264 compression are applied afterwards.
- **The hidden time:** the flicker's frequency follows the real Great Britain grid record (NESO, one reading per second) starting at **20:30:50 BST on 17 September 2026**. That is 2 minutes 10 seconds before the time on the overlay.

## What the agent recovers

Working only from the video, the agent extracts the flicker, searches the grid record, and reports a true start of **20:30:51**, within one second of the hidden time, with a clock offset of 2 min 9 s. A blind search of the whole of September 2026 (30 days, 2.6 million candidate start times) finds the same second, with the best wrong answer scoring 0.79 against 0.98.

## The real recordings

The simulated clip tests the agent on a known answer. The evidence that the method works on real cameras and real lights is separate:

- our own 18-minute phone video, dated to within 2 seconds ([PROOF.md](PROOF.md));
- 43 excerpts of that video, searched blind across a week, with no wrong answers;
- the live agent runs on those recordings in [`runs/`](../runs/).
